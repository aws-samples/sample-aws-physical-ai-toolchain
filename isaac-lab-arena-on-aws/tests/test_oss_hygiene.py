"""Check selected publication-hygiene patterns in component and connector files.

The scan uses ``git ls-files`` to include tracked and visible untracked text.
Generated artifacts follow the repository's ignore rules. These checks
supplement review; they are not a complete detector of private information.
"""
import os
import re
import subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)

# Binary extensions to skip. Everything else (including extensionless files
# such as Dockerfile) is scanned; genuine binaries that slip through are caught
# by the UnicodeDecodeError guard in the scan loop.
_BINARY_EXT = (
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".whl", ".tar", ".gz",
    ".zip", ".woff", ".woff2", ".pyc", ".so", ".bin", ".pt", ".pth", ".npy",
)
_THIS = os.path.basename(__file__)

# (label, compiled regex). Patterns are high-signal internal markers.
FORBIDDEN = [
    # AWS's public DLC account and synthetic documentation/test accounts are allowed.
    ("unrecognized account number", re.compile(
        r"\b(?!(?:763104351884|123456789012|111122223333|999988887777|0{12})\b)\d{12}\b")),
    ("internal cycle code (Stage N / Stage N step M)", re.compile(r"\bStage \d")),
    ("internal milestone code (M0.x / M1 milestone)", re.compile(r"\bM0\.\d\b")),
    ("internal priority code (P0 #N)", re.compile(r"\bP0 #\d")),
    ("internal fix code (An FIX)", re.compile(r"\bA\d FIX\b", re.IGNORECASE)),
    ("internal increment code", re.compile(r"\bINCREMENT \d", re.IGNORECASE)),
    ("internal roadmap ref (North Star)", re.compile(r"North Star")),
    ("internal design-change ref", re.compile(r"design change #\d")),
    ("proposal ref ('the pitch')", re.compile(r"\bthe pitch\b")),
    ("dev-account narration", re.compile(r"\bdev account\b|Discovered live")),
    ("placeholder-in-narration", re.compile(r"<your-account-id>")),
    ("dated internal review note", re.compile(r"(design|reasoning)[- ]review 20\d\d")),
    ("internal milestone/North-Star code (NS1/NS2)", re.compile(r"\bNS[12]\b")),
    ("internal milestone code (M3 …)", re.compile(r"\bM3 (fix|milestone|SimEval)\b|reach M3\b")),
    ("internal work-item code (W1.1)", re.compile(r"\bW\d\.\d\b")),
    ("internal issue-tracker ref (#N vN / #N fix)", re.compile(r"#\d{1,3} (v\d|fix)\b")),
    ("internal image-iteration version (vNN SimEval/build/hung/Arena)",
     re.compile(r"\bv\d{1,2} (SimEval|build|hung|Arena)\b")),
    ("retired hand-built image tag (:vNN)", re.compile(r"isaac-arena:v\d+")),
    ("internal dose/anchor run narration", re.compile(r"dose=\d+ anchor|anchor completed \d+ ep")),
    ("internal 'port blocker' framing", re.compile(r"port blocker")),
    ("internal review-process residue", re.compile(r"\bsubagent\b|reviewers? concur", re.IGNORECASE)),
    # Internal review-loop residue (severity-tagged review notes / delta reviews).
    ("internal review-severity label",
     re.compile(r"\breview (HIGH|MED|MEDIUM|LOW)\b|\bdelta-review (HIGH|MED|MEDIUM|LOW|#\d)")),
    # Internal phase/PR-cycle codes (PR1..PR9) used during staged development.
    # Narrowed to the internal "PR<n> <descriptor>" style so it does not collide
    # with ordinary "PR" (pull request) references once the repo is public.
    ("internal phase/PR code (PRn)",
     re.compile(r"\bPR[1-9]\b(?=\s+(?:layout|invariant|Option|option|move|refactor|phase|scope|gate|baseline))")),
]


def _iter_files():
    toolchain_root = os.path.dirname(_REPO)
    proc = subprocess.run(
        ["git", "-C", toolchain_root, "ls-files", "--cached", "--others",
         "--exclude-standard", "-z", "--", os.path.basename(_REPO),
         "containers/isaac-lab-arena"],
        capture_output=True, text=True, check=True)
    paths = sorted(set(proc.stdout.split("\0")) - {""})
    if not paths:
        raise AssertionError("Publication scan discovered no component files")
    for rel in paths:
        if os.path.basename(rel) == _THIS or rel.endswith(_BINARY_EXT):
            continue
        yield os.path.join(toolchain_root, rel)


def test_no_internal_tokens_in_shipped_files():
    hits = []
    for path in _iter_files():
        try:
            with open(path, encoding="utf-8") as fh:
                lines = fh.readlines()
        except (UnicodeDecodeError, OSError):
            continue
        for i, line in enumerate(lines, 1):
            for label, rx in FORBIDDEN:
                if rx.search(line):
                    rel = os.path.relpath(path, _REPO)
                    hits.append(f"{rel}:{i}  [{label}]  {line.strip()[:100]}")
    assert not hits, "internal tokens found in shipped files:\n" + "\n".join(hits)


def test_vendored_upstream_copies_carry_their_permission_notice():
    """I13: this repository redistributes upstream source, publicly.

    The MIT licence requires the copyright notice AND the permission notice to accompany copies or
    substantial portions. Naming the licence in NOTICE is attribution, not the notice the licence
    asks for. A fixture is still a copy.
    """
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    vendored = sorted((root / "tests/data").glob("*_pinned.py"))
    assert vendored, "no vendored upstream fixtures found"
    for path in vendored:
        text = path.read_text()
        # Licence-aware: MIT requires its permission notice verbatim in copies, Apache-2.0
        # requires retained copyright and attribution notices. My first version demanded MIT's
        # text from every copy, which the Apache-licensed fixtures cannot satisfy -- and the
        # test caught that I had only supplied a notice for one of the three.
        assert "Licence:" in text or "MIT License" in text, (
            f"{path.name} does not state the upstream licence")
        if "MIT License" in text:
            assert "Permission is hereby granted" in text, (
                f"{path.name} is an MIT copy without its permission notice")
        assert "Copyright" in text, f"{path.name} lacks the upstream copyright line"
        # The provenance must be exact enough to re-fetch and diff.
        assert "github.com/" in text and len([c for c in text.split()
                                              if len(c) == 40 and c.isalnum()]) >= 1, (
            f"{path.name} does not record the upstream repo and pinned commit")


def test_the_readme_does_not_deny_resources_the_infra_creates():
    """I9: the README told operators no component bucket or SageMaker role is created.

    Terraform creates two buckets and three SageMaker execution roles. That is a materially wrong
    statement, in a public-bound repository, about what `terraform apply` does to someone's
    account -- and it drifted because the role split and the handoff bucket were added without
    revisiting the prose. Pinned so the same drift fails a test instead of shipping.
    """
    import pathlib
    import re
    root = pathlib.Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text()
    infra = "\n".join(p.read_text() for p in (root / "infra").glob("*.tf"))

    buckets = set(re.findall(r'resource "aws_s3_bucket" "(\w+)"', infra))
    roles = set(re.findall(r'resource "aws_iam_role" "(\w+)"', infra))
    sagemaker_roles = roles - {"codebuild"}

    if buckets:
        assert "No component\nbucket" not in readme and "No component bucket" not in readme, (
            f"the README denies creating a component bucket, but infra creates {sorted(buckets)}")
    if sagemaker_roles:
        assert "SageMaker execution role is created" not in readme.replace(
            "three SageMaker execution roles", ""), (
            f"the README denies creating a SageMaker execution role, but infra creates "
            f"{sorted(sagemaker_roles)}")
    # Every SageMaker execution role must be named, so an operator can audit what they granted.
    for role in sorted(sagemaker_roles):
        assert role in readme, f"infra creates the {role!r} role but the README never names it"


def test_ci_runs_every_standing_gate():
    """I9: the gates existed and CI never invoked them.

    A gate that only runs when a maintainer remembers it is not a gate. Each of these covers a
    defect class the pytest suite structurally cannot reach, and each of those defects reached a
    commit before its gate existed -- so the suite passing is not evidence about them.
    """
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    buildspec = (root / "buildspec_tests.yml").read_text()
    gates = sorted(p.name for p in (root / "scripts").glob("gate_*.py"))
    assert gates, "no standing gates found"
    for gate in gates:
        assert gate in buildspec, (
            f"{gate} is never run by CI, so it protects nothing that is not checked by hand")


def test_teardown_documents_every_bucket_the_component_owns():
    """I10: teardown described only ECR, CodeBuild and the HF grant.

    The component owns two buckets holding immutable promoted artifacts and attestations. S3 will
    not delete a non-empty bucket, so following the old procedure stops part-way -- after the ECR
    repositories and CodeBuild project are already gone. Worse, those buckets are the provenance of
    every model already registered, so removing them is a decision rather than a step.
    """
    import pathlib
    import re
    root = pathlib.Path(__file__).resolve().parents[1]
    teardown = (root / "README.md").read_text().split(
        "## Resource ownership and permanent teardown\n", 1)[1].split("\n## ", 1)[0]
    infra = "\n".join(p.read_text() for p in (root / "infra").glob("*.tf"))
    for bucket in sorted(set(re.findall(r'resource "aws_s3_bucket" "(\w+)"', infra))):
        assert bucket in teardown, (
            f"infra creates the {bucket!r} bucket but teardown never mentions it, so the "
            f"documented procedure cannot complete")
    # The consequence must be stated, not just the resource listed.
    assert "provenance" in teardown, "teardown does not say what deleting the evidence costs"


def test_every_standing_gate_runs_from_any_directory(tmp_path):
    """C ycle-15 I4: the gates used CWD-relative paths, so CI -- which invokes them from the repo
    root -- got FileNotFoundError from all three. A gate that fails for the wrong reason is worse
    than no gate: the obvious response is to delete the invocation.

    Runs each gate from a directory unrelated to the repo, which is the case CI actually exercises.
    """
    import pathlib
    import subprocess
    import sys
    root = pathlib.Path(__file__).resolve().parents[1]
    gates = sorted((root / "scripts").glob("gate_*.py"))
    assert gates, "no standing gates found"
    for gate in gates:
        result = subprocess.run([sys.executable, str(gate)], cwd=tmp_path,
                                capture_output=True, text=True)
        assert result.returncode == 0, (
            f"{gate.name} fails when run from {tmp_path} rather than the component directory:\n"
            f"{result.stdout[-800:]}{result.stderr[-800:]}")


def test_the_readme_names_the_schema_version_the_code_requires():
    """Cycle-15 S2: the README described SimEval output as schema-v2 while the validator requires 3.

    The README is the public contract for a repo bound for aws-samples, so a stale version there is a
    documented interface that does not exist. Derived from the validator rather than hardcoded, so
    bumping the schema fails this test until the contract is updated with it.
    """
    import pathlib
    import re
    root = pathlib.Path(__file__).resolve().parents[1]
    validator = (root / "src/vla_pipeline/common/validator.py").read_text()
    match = re.search(r'report\["schema_version"\]\s*!=\s*(\d+)', validator)
    assert match, "the required schema_version no longer parses from the validator"
    required = match.group(1)
    readme = (root / "README.md").read_text()
    assert f"schema-v{required}" in readme, (
        f"the validator requires schema_version {required} but the README does not say schema-v"
        f"{required}; the published contract names a schema the code rejects")
    for stale in range(1, int(required)):
        assert f"schema-v{stale}" not in readme, (
            f"the README still advertises schema-v{stale}, which the validator rejects")
