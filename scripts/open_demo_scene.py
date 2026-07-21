"""
Open Isaac Sim with a non-empty demo scene instead of a blank stage.

Workstation users connecting via DCV otherwise land on an empty Isaac Sim stage.
This standalone script loads an NVIDIA sample environment (a warehouse) from the
Isaac Sim cloud asset root so there's something meaningful on screen out of the box.

Pattern + APIs are taken verbatim from NVIDIA's own standalone examples
(IsaacSim repo: source/standalone_examples/api/isaacsim.simulation_app/load_stage.py):
  - `from isaacsim import SimulationApp` (instantiate FIRST, before other imports)
  - `from isaacsim.storage.native import get_assets_root_path, is_file`
  - `omni.usd.get_context().open_stage(...)`

The asset root resolves to NVIDIA's hosted asset server, so NO local Nucleus
server is required. Run with the pip-installed isaacsim env active:

    source ~/isaac-env/bin/activate
    python scripts/open_demo_scene.py
    # or a different sample:
    python scripts/open_demo_scene.py --usd-path /Isaac/Environments/Grid/default_environment.usd
"""

import argparse
import sys

from isaacsim import SimulationApp

# Default: an NVIDIA-provided warehouse environment (on-theme for a robotics demo).
DEFAULT_USD = "/Isaac/Environments/Simple_Warehouse/warehouse.usd"

parser = argparse.ArgumentParser("Open Isaac Sim with a demo scene")
parser.add_argument("--usd-path", default=DEFAULT_USD,
                    help="USD path relative to the Isaac Sim assets root")
parser.add_argument("--headless", action="store_true", help="Run without a UI window")
args, _ = parser.parse_known_args()

# SimulationApp must be created before any omni/isaacsim imports.
kit = SimulationApp({"width": 1280, "height": 720, "headless": args.headless})

import carb  # noqa: E402
import omni  # noqa: E402
from isaacsim.storage.native import get_assets_root_path, is_file  # noqa: E402

assets_root = get_assets_root_path()
if assets_root is None:
    carb.log_error("Could not resolve the Isaac Sim assets root (network/asset server unreachable).")
    kit.close()
    sys.exit(1)

usd_path = assets_root + args.usd_path
if not is_file(usd_path):
    carb.log_error(f"Sample USD not found: {usd_path}")
    kit.close()
    sys.exit(1)

print(f"Opening demo scene: {usd_path}")
omni.usd.get_context().open_stage(usd_path)

# Let the stage finish loading before handing control to the UI.
kit.update()
kit.update()
from isaacsim.core.experimental.utils.stage import is_stage_loading  # noqa: E402
while is_stage_loading():
    kit.update()
print("Demo scene loaded.")

# In GUI mode, keep the app open for interaction; in headless, just exit cleanly.
if args.headless:
    kit.close()
else:
    while kit.is_running():
        kit.update()
    kit.close()
