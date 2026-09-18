"""Every shared module an entrypoint imports flatly must actually be shipped to that entrypoint.

There are THREE separate, independently ENUMERATED shipping mechanisms, each serving different steps:

  1. src/vla_pipeline/common/sourcedir.py  -- a tuple of module filenames, for train/eval sourcedirs
  2. src/vla_pipeline/runner.py            -- base64-embeds modules into a bootstrap string, because
                                              SageMaker's Validate step downloads ONE file
  3. scripts/build_arena_connector.py      -- _required_zip_members(), for the Arena container image

None of them globs, and nothing tied them to the imports they exist to satisfy. That is why the
runner bootstrap stayed hardcoded to two files across three additions of shared code: adding a module
and forgetting a mechanism raises ImportError INSIDE a SageMaker job while the local suite stays
green, because locally the module is importable from src/.

The module list here is DERIVED from the actual import statements. A hand-written list would be the
same defect one level up.
"""
import ast
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _shared_module_names() -> set:
    return {p.stem for p in (_ROOT / "src/vla_pipeline/common").glob("*.py")
            if p.stem != "__init__"}


def _flat_imports() -> dict:
    """{module_name: {importing entrypoint paths}} for flat (non-package) imports."""
    shared = _shared_module_names()
    found: dict = {}
    for path in sorted((_ROOT / "entrypoints").rglob("*.py")):
        if "__pycache__" in str(path):
            continue
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module in shared:
                names = [node.module]
            elif isinstance(node, ast.Import):
                names = [a.name for a in node.names if a.name in shared]
            for name in names:
                found.setdefault(name, set()).add(path.relative_to(_ROOT).as_posix())
    return found


def test_flat_imports_are_staged_into_the_sourcedir():
    staged = (_ROOT / "src/vla_pipeline/common/sourcedir.py").read_text()
    for module, importers in sorted(_flat_imports().items()):
        assert f'"{module}.py"' in staged, (
            f"{module} is imported flatly by {sorted(importers)} but sourcedir.py never stages it, "
            f"so the import fails inside the SageMaker job")


def test_modules_the_validate_step_imports_are_in_the_runner_bootstrap():
    """The Validate step gets its modules ONLY through the base64 bootstrap."""
    runner = (_ROOT / "src/vla_pipeline/runner.py").read_text()
    for module, importers in sorted(_flat_imports().items()):
        if "entrypoints/validate_entry.py" not in importers:
            continue
        assert f"{module}.py" in runner, (
            f"validate_entry.py imports {module} flatly, but runner.py's bootstrap does not carry "
            f"it -- the Validate step would raise ImportError with the suite still green")


def test_modules_the_arena_image_imports_are_in_the_zip():
    """The Arena image gets its modules only through the enumerated zip member list."""
    builder = (_ROOT / "scripts/build_arena_connector.py").read_text()
    for module, importers in sorted(_flat_imports().items()):
        if not any(i.startswith("entrypoints/eval/isaac_arena/") for i in importers):
            continue
        assert f"{module}.py" in builder, (
            f"an isaac_arena entrypoint imports {module} flatly, but _required_zip_members() omits "
            f"it, so it silently never reaches CodeBuild")


def test_modules_the_arena_image_imports_are_copied_into_the_image():
    """The FOURTH shipping boundary: a filename in the zip is not a file in the image.

    _required_zip_members() only guarantees a file reaches CodeBuild's build CONTEXT. The Dockerfile
    then COPYs an explicitly enumerated subset of that context into the image. A module present in
    the zip but absent from the COPY lines is missing at runtime and NOTHING else observes it: the
    local suite is green, the zip test above is green, the build succeeds (the file is in the
    context, merely never copied), and the failure appears only when a real Arena job executes the
    import -- which for capped_reader is inside extract_checkpoint, so it strands the job after the
    image has already pulled and the checkpoint has already downloaded.

    This is the only shipping mechanism whose omission is invisible to every other check, which is
    why it is the one that broke.
    """
    dockerfile = _ROOT.parent / "containers/isaac-lab-arena/Dockerfile"
    assert dockerfile.is_file(), (
        f"the Arena connector Dockerfile is missing at {dockerfile}; build_arena_connector.py "
        f"names it as a required zip member, so the image cannot build without it")

    # Destination basenames the image ends up carrying, from COPY <src>... <dst>.
    copied = set()
    for line in dockerfile.read_text().splitlines():
        stripped = line.strip()
        if not stripped.upper().startswith("COPY "):
            continue
        parts = stripped.split()[1:]
        parts = [p for p in parts if not p.startswith("--")]
        if len(parts) < 2:
            continue
        dst = parts[-1]
        copied.add(dst.rstrip("/").rsplit("/", 1)[-1] if not dst.endswith("/") else None)
        if dst.endswith("/"):
            # a directory destination carries every source's basename
            copied.update(p.rsplit("/", 1)[-1] for p in parts[:-1])
    copied.discard(None)

    for module, importers in sorted(_flat_imports().items()):
        arena = [i for i in importers if i.startswith("entrypoints/eval/isaac_arena/")]
        if not arena:
            continue
        assert f"{module}.py" in copied, (
            f"{sorted(arena)} imports {module} flatly and _required_zip_members() ships it to the "
            f"build context, but {dockerfile.name} never COPYs it into the image -- the import "
            f"raises ModuleNotFoundError inside a real Arena job while every local check stays green")


def test_the_derivation_actually_finds_the_known_modules():
    """Guards the guard: a derivation returning nothing would satisfy every assertion above."""
    found = _flat_imports()
    assert {"digest", "validator", "rlds_validator"} <= set(found), (
        f"expected the known shared modules among the flat imports, got {sorted(found)}")
    assert len(found["digest"]) >= 5, "digest should have many importers; the walk is missing some"


def test_no_function_uses_a_flat_import_it_cannot_see():
    """C1 (cycle 15): validate_entry.main() used capped_tar_open with the import in a DIFFERENT
    function, so every production Validate raised NameError while the suite stayed green.

    Function-local imports are required here (the module is not on sys.path until staging), which
    makes them easy to add to one caller and miss another. This resolves scope with the AST instead
    of trusting that a symbol imported somewhere is visible everywhere.
    """
    for path in sorted((_ROOT / "entrypoints").rglob("*.py")):
        if "__pycache__" in str(path) or path.name in ("capped_reader.py", "digest.py",
                                                       "validator.py", "rlds_validator.py"):
            continue
        tree = ast.parse(path.read_text())
        shared = _shared_module_names()
        exported: dict = {}
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module in shared:
                exported.update({a.asname or a.name: "module" for a in node.names})
        for fn in [n for n in ast.walk(tree)
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
            local = dict(exported)
            for node in ast.walk(fn):
                if isinstance(node, ast.ImportFrom) and node.module in shared:
                    local.update({a.asname or a.name: fn.name for a in node.names})
            used = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
            # any name that some OTHER function imports from a shared module, used here unimported
            for name in used:
                owners = set()
                for other in [n for n in ast.walk(tree)
                              if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
                    for node in ast.walk(other):
                        if isinstance(node, ast.ImportFrom) and node.module in shared:
                            if name in {a.asname or a.name for a in node.names}:
                                owners.add(other.name)
                if owners and name not in local:
                    raise AssertionError(
                        f"{path.relative_to(_ROOT).as_posix()}: {fn.name}() uses {name!r}, imported "
                        f"only inside {sorted(owners)} -- this raises NameError at runtime")


def test_the_bootstrap_actually_carries_each_module_byte_for_byte():
    """I5 (cycle 15): the tests above assert filename STRINGS occur in the shipping code, which is not
    delivery. Astra removed the two bootstrap lines that WRITE capped_reader.py and the runner test
    still passed, because the filename survived elsewhere in the file.

    This decodes every base64 payload the generated bootstrap carries and requires each module's exact
    bytes to be among them. A payload that is named but not embedded, or embedded stale, fails.
    """
    import base64
    import re
    from vla_pipeline.runner import stage_validate_code

    # stage_validate_code returns the PATH of the staged entry file, not its text.
    generated = pathlib.Path(stage_validate_code(str(_ROOT))).read_text()
    # Two embedding forms: decoded inline, and assigned to a variable first (validator uses
    # _validator_b64 = "..."). Matching only the inline form missed validator entirely and the test
    # failed for its own reason rather than a delivery defect.
    payloads = set()
    for blob in re.findall(r'(?:_b64\.b64decode\(|_\w+_b64\s*=\s*)"([A-Za-z0-9+/=]{64,})"',
                           generated):
        try:
            payloads.add(base64.b64decode(blob))
        except Exception:
            continue
    assert payloads, "the bootstrap embeds no decodable payloads at all"

    required = {m for m, importers in _flat_imports().items()
                if "entrypoints/validate_entry.py" in importers}
    assert required, "validate_entry imports no shared modules; the derivation is broken"
    for module in sorted(required):
        source = (_ROOT / "src/vla_pipeline/common" / f"{module}.py").read_bytes()
        assert source in payloads, (
            f"{module}.py is named in the bootstrap but its BYTES are not embedded, so the Validate "
            f"step would import a module that never arrives (or an older copy)")


def test_the_bootstrap_writes_every_module_it_embeds():
    """Naming and embedding are not enough -- the generated code must also write each file."""
    import re
    from vla_pipeline.runner import stage_validate_code

    # stage_validate_code returns the PATH of the staged entry file, not its text.
    generated = pathlib.Path(stage_validate_code(str(_ROOT))).read_text()
    written = set(re.findall(r'_module_dir\.name,\s*"(\w+\.py)"', generated))
    required = {f"{m}.py" for m, importers in _flat_imports().items()
                if "entrypoints/validate_entry.py" in importers}
    missing = required - written
    assert not missing, (
        f"the bootstrap embeds but never writes {sorted(missing)}, so the module is not on sys.path "
        f"when validate_entry imports it")


def test_the_bootstrap_writes_the_right_bytes_to_the_right_filename():
    """Cycle-16 I5: the two tests above compare SETS -- one of decoded payloads, one of filenames --
    and establish no mapping between them. Astra swapped the output filenames for digest.py and
    capped_reader.py, left the payloads intact, and both tests still passed.

    This EXECUTES the generated bootstrap and reads what it actually wrote, which is the only check
    that pins filename -> bytes. Same lesson as cycle-15 I6: a set of correct names is not a correct
    mapping.
    """
    import pathlib
    import subprocess
    import sys
    from vla_pipeline.runner import stage_validate_code

    generated = pathlib.Path(stage_validate_code(str(_ROOT))).read_text()
    marker = "import sys as _sys"
    assert marker in generated, "the bootstrap no longer has the expected shape"
    bootstrap = generated[:generated.index(marker)]

    # Run in a subprocess: the bootstrap creates a TemporaryDirectory and mutates sys.path, and it
    # must not be able to affect the test process either way.
    probe = (bootstrap
             + "\nimport json, os\n"
             + "print(json.dumps({n: open(os.path.join(_module_dir.name, n), 'rb').read().hex()\n"
             + "                  for n in sorted(os.listdir(_module_dir.name))}))\n")
    # The staged source contains SDK wheels and exceeds the OS argv size limit.
    result = subprocess.run([sys.executable, "-"], input=probe, capture_output=True, text=True)
    assert result.returncode == 0, f"the bootstrap does not run: {result.stderr[-600:]}"
    import json as _json
    written = {name: bytes.fromhex(body)
               for name, body in _json.loads(result.stdout.strip().splitlines()[-1]).items()}
    assert written, "the bootstrap wrote no modules at all"

    required = {m for m, importers in _flat_imports().items()
                if "entrypoints/validate_entry.py" in importers}
    for module in sorted(required):
        filename = f"{module}.py"
        assert filename in written, (
            f"the bootstrap never writes {filename}, so validate_entry cannot import it")
        expected = (_ROOT / "src/vla_pipeline/common" / filename).read_bytes()
        assert written[filename] == expected, (
            f"{filename} is written but with the WRONG bytes -- the payload delivered under this name "
            f"is not this module, so the Validate step imports something else entirely")


def test_the_documented_inventory_lists_every_shared_module_the_image_copies():
    """Documentation that under-reports the baked set is how a rebuild gets skipped.

    Arena is `delivery_mode: baked`: its code is COPYed into the image, so a source fix with no
    connector rebuild leaves the running image unchanged while results are attributed to current code.
    That happened -- three executions ran a day-old image. The docs telling a reader which files
    require a rebuild is therefore load-bearing, and both inventories omitted `capped_reader.py` and
    `source_identity.py`. A missing `capped_reader.py` COPY is separately how the image once shipped
    without it at all.

    The Dockerfile is the source of truth; the docs must not under-report it.
    """
    import re
    root = pathlib.Path(__file__).resolve().parents[1]
    dockerfile = (root.parent / "containers/isaac-lab-arena/Dockerfile").read_text()
    # Key on the BASENAME, from any COPY source. The Dockerfile copies digest.py from the _shared/
    # mirror but validator.py, capped_reader.py and source_identity.py from src/vla_pipeline/common/,
    # so a pattern anchored on "_shared/" captures ONE module. My first version did exactly that and
    # passed while checking a quarter of the set -- the mutation that should have failed it did not,
    # which is the only reason I looked.
    # DERIVED from the Dockerfile, not a hardcoded tuple. The tuple version listed four modules and a
    # fifth (checkpoint_compat.py) was baked into the image without ever entering it, so this test
    # passed while the inventory it checks omitted a module -- the very drift it exists to catch. A
    # hardcoded list needs a manual update that nothing forces.
    copied = {pathlib.Path(m).name for m in re.findall(
        r"^COPY\s+\S*?((?:src/vla_pipeline/common|_shared)/[a-z_]+\.py)\s", dockerfile, re.M)}
    assert len(copied) >= 5, (
        f"the Arena Dockerfile copies only {sorted(copied)}; five shared modules are known to be "
        f"baked. Either a COPY line was lost -- which is how the image once shipped without "
        f"capped_reader.py -- or this pattern stopped matching the Dockerfile's paths.")

    text = (root / "README.md").read_text().split(
        "### Baked inputs and rebuilds\n", 1)[1].split("\n### ", 1)[0]
    missing = sorted(m for m in copied if m not in text)
    assert not missing, (
        f"README baked-input inventory does not list {missing}, which the Arena Dockerfile "
        f"COPYs into the image. A reader would miss the required connector rebuild.")
