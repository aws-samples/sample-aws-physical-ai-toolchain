"""Isaac Lab Arena runtime resolution: suite manifest -> the eval's knobs.

ONE implementation, shared by both Arena launchers (``scripts/run_arena.py`` and
``scripts/submit_simeval.py``). They previously each carried their own CLI
defaults, which drifted apart -- ``submit_simeval.py`` defaulted to
``--suite libero_spatial`` alongside ``--task-name gr1_open_microwave`` (a
mutually incoherent pair on an Arena-only script) and to a different
``--eval-seed`` than ``run_arena.py``. Resolving in one place is what makes
"the suite defines the cell" true for every entry point.

The output is the ``EVAL_SIM_CONFIG`` payload: exactly the six keys
``entrypoints/eval/isaac_arena/gr00t/eval_entry.py`` unpacks into its discrete
``SM_HP_*`` / ``EVAL_*`` env keys. Adding or dropping a key here is a hard failure
in the eval entry (it rejects unknown keys), which is intentional.
"""
from __future__ import annotations

from .registry import ResolvedSuite

#: The eval entry's "omit this flag" sentinel. Honoured ONLY for ``EVAL_OBJECT``
#: and ``EVAL_ARENA_EMBODIMENT`` -- see :func:`_sentinel`.
OMIT = "NONE"


class ArenaConfigError(ValueError):
    """A suite cannot produce a runnable Arena configuration. Never swallowed."""


def _sentinel(value) -> str:
    """``None``/``""`` -> ``"NONE"``, the eval entry's "omit this flag" sentinel.

    Only for fields the eval entry actually treats as optional (``EVAL_OBJECT``,
    ``EVAL_ARENA_EMBODIMENT``). It must NOT be applied to ``embodiment_tag``: that
    value is passed straight through to the GR00T server's ``--embodiment-tag``,
    which has no "omit" contract, so ``"NONE"`` would launch the server with a
    bogus tag and die on an opaque tyro enum error after paying for the venv build.
    """
    return OMIT if value in (None, "") else str(value)


def resolve_runtime(
    suite: ResolvedSuite,
    family: str,
    family_version: str,
    *,
    embodiment_tag: str | None = None,
    policy_config: str | None = None,
    arena_embodiment: str | None = None,
    obj: str | None = None,
) -> dict:
    """Resolve the Arena runtime knobs: suite manifest, with explicit overrides.

    Every override defaults to ``None`` meaning "no override, use the manifest".
    Raises :class:`ArenaConfigError` at SUBMIT time -- rather than letting the run
    fail inside a GPU job -- when the suite cannot produce a valid configuration.
    """
    # Already guaranteed by registry._parse_arena (an isaac_arena suite without an
    # arena block does not resolve). Retained because resolve_runtime is public and
    # may be called with a non-Arena suite; do not "test" it as a live path.
    if suite.arena is None:
        raise ArenaConfigError(
            f"suite {suite.name!r} has no arena block (simulator="
            f"{suite.simulator!r}); it cannot produce an Arena configuration.")
    arena = suite.arena

    tag = embodiment_tag
    if tag is None:
        tag = suite.family_override(family, family_version).get("embodiment_tag")
    if family_version == "n16" and tag and tag != "GR1":
        raise ArenaConfigError(
            f"N1.6 server hardcodes --embodiment-tag GR1; the requested "
            f"override {tag!r} would be recorded but not applied. "
            f"Use --embodiment-tag GR1 or omit it for the N1.6 route.")
    if not tag:
        # family_override() documents an absent entry as "not an error" -- it means
        # "use the family's defaults.json", which is right for LIBERO. An Arena run
        # must name a tag, so fail here rather than emit an unconsumable sentinel.
        raise ArenaConfigError(
            f"suite {suite.name!r} declares no family_overrides.{family}."
            f"{family_version}.embodiment_tag, and the Arena eval entry has no "
            f"'omit' sentinel for the GR00T server --embodiment-tag.\n"
            f"  Add it to config/suites/{suite.name}.yaml, or pass --embodiment-tag "
            f"for this run.")

    cfg = policy_config if policy_config is not None else arena.policy_config
    if not cfg:
        raise ArenaConfigError(
            f"suite {suite.name!r} has no Arena policy config.\n"
            f"  config/suites/{suite.name}.yaml declares arena.policy_config: null.\n"
            f"  Set it to the container path of an embodiment-matching closed-loop\n"
            f"  config, or pass --policy-config-yaml <container-path> for this run.\n"
            f"  There is deliberately no default: Isaac Lab Arena's packaged example\n"
            f"  configs are embodiment PLACEHOLDERS and would yield an invalid result.\n"
            f"  See README.md#arena-images-and-connector-contract.")

    return {
        # GR00T inference server --embodiment-tag. Never sentinelled.
        "embodiment_tag": str(tag),
        # policy_runner's positional `task`.
        "task_name": arena.task,
        "policy_config_yaml": cfg,
        "arena_embodiment": _sentinel(
            arena.embodiment if arena_embodiment is None else arena_embodiment),
        "object": _sentinel(arena.object if obj is None else obj),
        # NO budget key. The sample size is the episode count and travels as
        # EvalTrials -> EVAL_TRIALS -> --num_episodes. It used to be carried here as a
        # step budget resolved independently of EvalTrials, and nothing reconciled the
        # two: a 280-step budget against a 500-step episode length silently produced
        # zero episodes. One knob, one path.
    }


def gate_suite(suite: ResolvedSuite, *, allow_experimental: bool) -> None:
    """Reject a suite that cannot legitimately run on Arena. Fail loud.

    Both launchers apply this before submitting, so a wrong-simulator or
    never-proven suite costs a second rather than a GPU job (and, for an
    experimental suite, a run that Validate would later reject as an unknown suite).
    """
    from .registry import suites_for_simulator
    if suite.simulator != "isaac_arena":
        raise ArenaConfigError(
            f"suite {suite.name!r} runs on simulator {suite.simulator!r}, not "
            f"'isaac_arena'. Arena suites: "
            f"{', '.join(suites_for_simulator('isaac_arena', include_experimental=True))}")
    if not suite.supported and not allow_experimental:
        raise ArenaConfigError(
            f"suite {suite.name!r} is status: experimental -- it is excluded from "
            f"the validator's suite allowlist, so SimEval would run and Validate "
            f"would then reject the report as an unknown suite. Pass "
            f"--allow-experimental to submit anyway (diagnostic only).")
