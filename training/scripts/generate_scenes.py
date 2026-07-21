"""
Scene Generation with NVIDIA Cosmos

Generates diverse, photorealistic bin-picking scenes for training.
Uses Cosmos world foundation model to create visual variations that
improve sim-to-real transfer.

Generates:
  - Varied lighting conditions (warehouse, factory, outdoor)
  - Different bin fill levels and object arrangements
  - Random object textures and materials
  - Background clutter and environmental context

Output: USD scene files saved to S3 for use in training pipeline.

Usage:
    python generate_scenes.py --output-dir s3://bucket/generated/ --num-scenes 1000
    python generate_scenes.py --output-dir ./scenes/ --num-scenes 10 --no-cosmos
"""

import argparse
import os
import json
import random
from pathlib import Path
from typing import List, Dict

# Isaac Sim for USD scene construction
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import omni.usd
from pxr import Usd, UsdGeom, UsdLux, UsdShade, Gf, Sdf


# ============================================================
# Scene Templates
# ============================================================

# Object catalog for bin picking
OBJECTS = [
    {"name": "cube_small", "size": (0.03, 0.03, 0.03), "mass": 0.05},
    {"name": "cube_medium", "size": (0.05, 0.05, 0.05), "mass": 0.15},
    {"name": "cylinder", "size": (0.03, 0.06), "mass": 0.1},
    {"name": "sphere", "size": (0.025,), "mass": 0.08},
    {"name": "box_flat", "size": (0.06, 0.04, 0.02), "mass": 0.12},
    {"name": "irregular", "size": (0.04, 0.03, 0.05), "mass": 0.2},
]

# Lighting presets
LIGHTING_PRESETS = [
    {"name": "warehouse_overhead", "intensity": 3000, "color": (1.0, 0.95, 0.9), "angle": 60},
    {"name": "factory_fluorescent", "intensity": 2500, "color": (0.9, 0.95, 1.0), "angle": 45},
    {"name": "dim_ambient", "intensity": 1000, "color": (1.0, 0.9, 0.8), "angle": 90},
    {"name": "harsh_spotlight", "intensity": 5000, "color": (1.0, 1.0, 1.0), "angle": 30},
    {"name": "mixed_natural", "intensity": 2000, "color": (1.0, 0.98, 0.95), "angle": 75},
]

# Material variations
MATERIALS = [
    {"name": "plastic_red", "color": (0.8, 0.2, 0.1), "roughness": 0.4},
    {"name": "plastic_blue", "color": (0.1, 0.3, 0.8), "roughness": 0.4},
    {"name": "metal_silver", "color": (0.7, 0.7, 0.7), "roughness": 0.2},
    {"name": "rubber_black", "color": (0.1, 0.1, 0.1), "roughness": 0.8},
    {"name": "wood_light", "color": (0.7, 0.5, 0.3), "roughness": 0.6},
    {"name": "cardboard", "color": (0.6, 0.5, 0.3), "roughness": 0.7},
    {"name": "plastic_white", "color": (0.9, 0.9, 0.9), "roughness": 0.3},
    {"name": "plastic_green", "color": (0.2, 0.7, 0.2), "roughness": 0.4},
]


# ============================================================
# Scene Generation (Procedural — no Cosmos)
# ============================================================

def generate_procedural_scene(scene_id: int, output_dir: str, config: Dict) -> str:
    """Generate a single scene with randomized objects, lighting, and materials."""

    scene_path = os.path.join(output_dir, f"scene_{scene_id:05d}.usd")

    # Create new USD stage
    stage = Usd.Stage.CreateNew(scene_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    # Root prim
    root = UsdGeom.Xform.Define(stage, "/World")

    # Ground plane
    ground = UsdGeom.Mesh.Define(stage, "/World/Ground")
    ground.CreatePointsAttr([(-2, -2, 0), (2, -2, 0), (2, 2, 0), (-2, 2, 0)])
    ground.CreateFaceVertexCountsAttr([4])
    ground.CreateFaceVertexIndicesAttr([0, 1, 2, 3])

    # Bin
    bin_prim = UsdGeom.Cube.Define(stage, "/World/Bin")
    bin_prim.CreateSizeAttr(0.3)  # 30cm bin
    bin_xform = UsdGeom.Xformable(bin_prim)
    bin_xform.AddTranslateOp().Set(Gf.Vec3d(0.5, 0.0, 0.15))

    # Randomize number of objects
    num_objects = random.randint(config['min_objects'], config['max_objects'])

    for i in range(num_objects):
        obj_template = random.choice(OBJECTS)
        material = random.choice(MATERIALS)

        # Random position within bin
        x = 0.5 + random.uniform(-0.1, 0.1)
        y = random.uniform(-0.1, 0.1)
        z = 0.05 + i * 0.03 + random.uniform(0, 0.02)  # Stack slightly

        obj_path = f"/World/Object_{i}"
        obj_prim = UsdGeom.Cube.Define(stage, obj_path)
        obj_prim.CreateSizeAttr(obj_template['size'][0])

        obj_xform = UsdGeom.Xformable(obj_prim)
        obj_xform.AddTranslateOp().Set(Gf.Vec3d(x, y, z))

        # Random rotation
        obj_xform.AddRotateXYZOp().Set(Gf.Vec3f(
            random.uniform(0, 360),
            random.uniform(0, 360),
            random.uniform(0, 360),
        ))

        # Apply material color (simplified — full version uses UsdShade)
        obj_prim.CreateDisplayColorAttr([Gf.Vec3f(*material['color'])])

    # Lighting
    lighting = random.choice(LIGHTING_PRESETS)
    light = UsdLux.DistantLight.Define(stage, "/World/Light")
    light.CreateIntensityAttr(lighting['intensity'])
    light.CreateColorAttr(Gf.Vec3f(*lighting['color']))

    # Add slight random rotation to light
    light_xform = UsdGeom.Xformable(light)
    light_xform.AddRotateXYZOp().Set(Gf.Vec3f(
        -lighting['angle'] + random.uniform(-10, 10),
        random.uniform(-30, 30),
        0,
    ))

    # Save stage
    stage.GetRootLayer().Save()

    return scene_path


# ============================================================
# Scene Generation (Cosmos-enhanced)
# ============================================================

def generate_cosmos_scene(scene_id: int, output_dir: str, config: Dict) -> str:
    """
    Generate scene with Cosmos world foundation model for photorealistic variation.

    Cosmos adds:
    - Realistic textures and materials (scratched metal, dusty surfaces)
    - Environmental context (warehouse background, conveyor belts nearby)
    - Physically plausible lighting (bounced light, shadows, reflections)
    """
    # First generate base procedural scene
    base_scene = generate_procedural_scene(scene_id, output_dir, config)

    try:
        # Use Cosmos NIM API to enhance the scene
        # Reference: https://docs.nvidia.com/nim/cosmos/latest/api-reference.html
        import requests

        cosmos_endpoint = os.environ.get('COSMOS_NIM_ENDPOINT',
                                          'http://localhost:8000/v1/cosmos/transfer')

        # Cosmos Transfer model: takes a base scene and adds photorealistic detail
        payload = {
            "input_scene": base_scene,
            "prompt": (
                "Industrial warehouse environment with concrete floor, "
                "metal shelving in background, overhead LED panel lighting, "
                "slight dust particles in air, realistic shadows"
            ),
            "style": "photorealistic",
            "preserve_geometry": True,  # Keep object positions, just enhance visuals
            "output_format": "usd",
        }

        response = requests.post(cosmos_endpoint, json=payload, timeout=120)

        if response.status_code == 200:
            enhanced_path = base_scene.replace('.usd', '_cosmos.usd')
            with open(enhanced_path, 'wb') as f:
                f.write(response.content)
            print(f"    Cosmos-enhanced: {enhanced_path}")
            return enhanced_path
        else:
            print(f"    Cosmos unavailable (status {response.status_code}), using procedural scene")
            return base_scene

    except Exception as e:
        print(f"    Cosmos unavailable ({e}), using procedural scene")
        return base_scene


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Generate training scenes')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for USD scenes')
    parser.add_argument('--num-scenes', type=int, default=1000,
                        help='Number of scenes to generate')
    parser.add_argument('--task', type=str, default='pick-and-place')
    parser.add_argument('--robot', type=str, default='ur3')
    parser.add_argument('--gripper', type=str, default='robotiq-2f85')
    parser.add_argument('--min-objects', type=int, default=1)
    parser.add_argument('--max-objects', type=int, default=8)
    parser.add_argument('--no-cosmos', action='store_true',
                        help='Skip Cosmos enhancement (procedural only)')
    parser.add_argument('--randomize-lighting', action='store_true', default=True)
    parser.add_argument('--randomize-objects', action='store_true', default=True)
    parser.add_argument('--randomize-bin-fill', action='store_true', default=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    config = {
        'min_objects': args.min_objects,
        'max_objects': args.max_objects,
        'task': args.task,
        'robot': args.robot,
        'gripper': args.gripper,
    }

    print(f"{'='*60}")
    print(f"  Scene Generation")
    print(f"  Output: {args.output_dir}")
    print(f"  Scenes: {args.num_scenes}")
    print(f"  Objects per scene: {args.min_objects}-{args.max_objects}")
    print(f"  Cosmos enhancement: {'disabled' if args.no_cosmos else 'enabled'}")
    print(f"{'='*60}")

    generated_scenes = []

    for i in range(args.num_scenes):
        if args.no_cosmos:
            scene_path = generate_procedural_scene(i, args.output_dir, config)
        else:
            scene_path = generate_cosmos_scene(i, args.output_dir, config)

        generated_scenes.append(scene_path)

        if (i + 1) % 100 == 0:
            print(f"  Generated {i+1}/{args.num_scenes} scenes")

    # Save manifest
    manifest_path = os.path.join(args.output_dir, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump({
            'num_scenes': len(generated_scenes),
            'scenes': generated_scenes,
            'config': config,
            'cosmos_enhanced': not args.no_cosmos,
        }, f, indent=2)

    print(f"\n  Done! Generated {len(generated_scenes)} scenes")
    print(f"  Manifest: {manifest_path}")

    simulation_app.close()


if __name__ == '__main__':
    main()
