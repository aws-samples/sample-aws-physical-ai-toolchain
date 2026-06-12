"""
UR3 Pick-and-Place Isaac Lab Environment

Defines the RL task for a UR3 + Robotiq 2F-85 gripper picking objects from a bin.

Observation space:
  - Wrist camera RGB (64x64x3)
  - Joint positions (6)
  - Joint velocities (6)
  - Gripper state (1 — open/closed)
  - Object pose relative to end-effector (7 — pos + quat)

Action space:
  - End-effector delta pose (6 — dx, dy, dz, droll, dpitch, dyaw)
  - Gripper command (1 — open/close)

Reward:
  - Approach reward (distance to object decreasing)
  - Grasp reward (object lifted)
  - Success reward (object above threshold height)
  - Penalties for drops, collisions, time
"""

import torch
import math
from dataclasses import dataclass

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, RigidObjectCfg
from omni.isaac.lab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import (
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import CameraCfg
from omni.isaac.lab.utils import configclass


# ============================================================
# Scene Configuration
# ============================================================

@configclass
class PickAndPlaceSceneCfg(InteractiveSceneCfg):
    """Scene with UR3 robot, Robotiq gripper, bin, and objects."""

    # Ground plane
    ground = sim_utils.GroundPlaneCfg()

    # UR3 robot arm with Robotiq 2F-85 gripper
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="omniverse://localhost/NVIDIA/Assets/Isaac/Robots/UniversalRobots/ur3e/ur3e_robotiq_2f85.usd",
            activate_contact_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "shoulder_pan_joint": 0.0,
                "shoulder_lift_joint": -math.pi / 2,
                "elbow_joint": math.pi / 2,
                "wrist_1_joint": -math.pi / 2,
                "wrist_2_joint": -math.pi / 2,
                "wrist_3_joint": 0.0,
                # Robotiq gripper joints (open position)
                "finger_joint": 0.0,
                "left_inner_knuckle_joint": 0.0,
                "right_inner_knuckle_joint": 0.0,
                "right_outer_knuckle_joint": 0.0,
                "left_inner_finger_joint": 0.0,
                "right_inner_finger_joint": 0.0,
            },
        ),
    )

    # Bin (container for objects)
    bin = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Bin",
        spawn=sim_utils.UsdFileCfg(
            usd_path="omniverse://localhost/NVIDIA/Assets/Isaac/Props/Bins/plastic_bin.usd",
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5, 0.0, 0.0),  # 50cm in front of robot base
        ),
    )

    # Target object to pick
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.CuboidCfg(
            size=(0.04, 0.04, 0.04),  # 4cm cube
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=1.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),  # 100g
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8,
                dynamic_friction=0.6,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.2, 0.2),  # Red cube
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5, 0.0, 0.05),  # Inside bin, slightly above bottom
        ),
    )

    # Wrist camera (mounted on UR3 tool flange)
    wrist_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/tool0/WristCamera",
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.05),  # 5cm below tool flange
            rot=(0.0, 0.0, 0.0, 1.0),  # Looking down
        ),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.5,
            horizontal_aperture=3.6,  # Approximate UR wrist camera FOV
        ),
        width=64,
        height=64,
        data_types=["rgb"],
        update_period=0.02,  # 50Hz (matches control rate)
    )

    # Lighting
    light = sim_utils.DistantLightCfg(
        prim_path="/World/Light",
        intensity=3000.0,
        color=(1.0, 1.0, 1.0),
    )


# ============================================================
# Environment Configuration
# ============================================================

@configclass
class PickAndPlaceUR3EnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for UR3 pick-and-place environment."""

    # Scene
    scene: PickAndPlaceSceneCfg = PickAndPlaceSceneCfg(num_envs=4096, env_spacing=1.5)

    # Simulation parameters
    sim = sim_utils.SimulationCfg(
        dt=0.02,  # 50Hz physics
        render_interval=1,
    )

    # Episode length
    episode_length_s = 4.0  # 4 seconds = 200 steps at 50Hz

    # --- Observations ---
    observations = ObservationGroupCfg(
        policy=ObservationGroupCfg(
            joint_pos=ObservationTermCfg(
                func=lambda env: env.scene["robot"].data.joint_pos[:, :6],
            ),
            joint_vel=ObservationTermCfg(
                func=lambda env: env.scene["robot"].data.joint_vel[:, :6],
            ),
            gripper_state=ObservationTermCfg(
                func=lambda env: env.scene["robot"].data.joint_pos[:, 6:7],  # finger_joint
            ),
            object_pos_relative=ObservationTermCfg(
                func=_get_object_pose_relative,
            ),
            wrist_camera=ObservationTermCfg(
                func=lambda env: env.scene["wrist_camera"].data.output["rgb"].float() / 255.0,
            ),
        ),
    )

    # --- Actions ---
    # Delta end-effector pose + gripper command
    action_space = 7  # dx, dy, dz, droll, dpitch, dyaw, gripper

    # --- Rewards ---
    rewards = {
        "approach": RewardTermCfg(
            func=_reward_approach,
            weight=1.0,
        ),
        "grasp": RewardTermCfg(
            func=_reward_grasp,
            weight=2.0,
        ),
        "lift_success": RewardTermCfg(
            func=_reward_lift_success,
            weight=10.0,
        ),
        "time_penalty": RewardTermCfg(
            func=lambda env: torch.full((env.num_envs,), -0.01, device=env.device),
            weight=1.0,
        ),
        "collision_penalty": RewardTermCfg(
            func=_reward_collision_penalty,
            weight=1.0,
        ),
    }

    # --- Terminations ---
    terminations = {
        "time_out": TerminationTermCfg(
            func=lambda env: env.episode_length_buf >= env.max_episode_length,
        ),
        "success": TerminationTermCfg(
            func=_check_success,
        ),
    }

    # --- Events (domain randomization) ---
    events = {
        "reset_object_pose": EventTermCfg(
            func=_randomize_object_pose,
            mode="reset",
        ),
        "randomize_lighting": EventTermCfg(
            func=_randomize_lighting,
            mode="reset",
        ),
    }


# ============================================================
# Helper Functions
# ============================================================

def _get_object_pose_relative(env) -> torch.Tensor:
    """Get object pose relative to end-effector (7D: pos + quat)."""
    ee_pos = env.scene["robot"].data.body_pos_w[:, -1, :]  # Tool flange position
    obj_pos = env.scene["object"].data.root_pos_w
    obj_quat = env.scene["object"].data.root_quat_w

    # Relative position
    rel_pos = obj_pos - ee_pos

    return torch.cat([rel_pos, obj_quat], dim=-1)


def _reward_approach(env) -> torch.Tensor:
    """Reward for moving end-effector closer to object."""
    ee_pos = env.scene["robot"].data.body_pos_w[:, -1, :]
    obj_pos = env.scene["object"].data.root_pos_w

    distance = torch.norm(ee_pos - obj_pos, dim=-1)

    # Shaped reward: closer = higher reward (exponential decay)
    return torch.exp(-5.0 * distance)


def _reward_grasp(env) -> torch.Tensor:
    """Reward for successfully grasping (object between fingers and lifted)."""
    obj_pos = env.scene["object"].data.root_pos_w
    obj_height = obj_pos[:, 2]  # Z coordinate

    # Object is grasped if it's above the bin surface (0.05m) and moving with EE
    gripper_closed = env.scene["robot"].data.joint_pos[:, 6] > 0.5
    lifted = obj_height > 0.08  # Above bin rim

    return (gripper_closed & lifted).float()


def _reward_lift_success(env) -> torch.Tensor:
    """Large reward for lifting object above success threshold."""
    obj_pos = env.scene["object"].data.root_pos_w
    obj_height = obj_pos[:, 2]

    success = obj_height > 0.15  # 15cm above ground = clear of bin
    return success.float()


def _reward_collision_penalty(env) -> torch.Tensor:
    """Penalty for colliding with bin walls."""
    # Check contact forces on robot links
    contact_forces = env.scene["robot"].data.net_contact_forces_w
    force_magnitude = torch.norm(contact_forces, dim=-1).sum(dim=-1)

    # Penalize excessive contact (threshold filters normal grasping forces)
    excessive_contact = (force_magnitude > 50.0).float()
    return -excessive_contact


def _check_success(env) -> torch.Tensor:
    """Episode terminates successfully when object is lifted above threshold."""
    obj_pos = env.scene["object"].data.root_pos_w
    return obj_pos[:, 2] > 0.15


def _randomize_object_pose(env, env_ids: torch.Tensor):
    """Randomize object position within bin on episode reset."""
    num_resets = len(env_ids)

    # Random position within bin bounds (±8cm from center, slight height variation)
    random_pos = torch.zeros(num_resets, 3, device=env.device)
    random_pos[:, 0] = 0.5 + (torch.rand(num_resets, device=env.device) - 0.5) * 0.16
    random_pos[:, 1] = (torch.rand(num_resets, device=env.device) - 0.5) * 0.16
    random_pos[:, 2] = 0.05 + torch.rand(num_resets, device=env.device) * 0.02

    env.scene["object"].write_root_pose_to_sim(
        torch.cat([random_pos, torch.tensor([[0, 0, 0, 1]], device=env.device).expand(num_resets, -1)], dim=-1),
        env_ids,
    )


def _randomize_lighting(env, env_ids: torch.Tensor):
    """Randomize light intensity and color temperature on reset."""
    # This is a simplified version — full implementation would modify USD light prims
    pass


# ============================================================
# Environment Class
# ============================================================

class PickAndPlaceUR3Env(ManagerBasedRLEnv):
    """UR3 Pick-and-Place reinforcement learning environment."""

    cfg: PickAndPlaceUR3EnvCfg

    def __init__(self, cfg: PickAndPlaceUR3EnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self.success_threshold = 0.15  # meters above ground

    def _apply_action(self, action: torch.Tensor):
        """Apply delta end-effector pose + gripper command."""
        # Split action into EE delta and gripper
        ee_delta = action[:, :6]  # dx, dy, dz, droll, dpitch, dyaw
        gripper_cmd = action[:, 6:7]  # 0=open, 1=close

        # Scale actions
        pos_delta = ee_delta[:, :3] * 0.01  # Max 1cm per step
        rot_delta = ee_delta[:, 3:6] * 0.05  # Max ~3 degrees per step

        # Convert EE delta to joint commands via differential IK
        # (Isaac Lab provides this utility)
        from omni.isaac.lab.controllers import DifferentialIKController, DifferentialIKControllerCfg

        ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=True,
            ik_method="dls",  # Damped least squares
        )
        ik_controller = DifferentialIKController(ik_cfg, num_envs=self.num_envs, device=self.device)

        # Compute joint targets
        joint_targets = ik_controller.compute(
            ee_pos_delta=pos_delta,
            ee_rot_delta=rot_delta,
            current_joint_pos=self.scene["robot"].data.joint_pos[:, :6],
        )

        # Apply joint position targets to robot
        self.scene["robot"].set_joint_position_target(joint_targets, joint_ids=list(range(6)))

        # Apply gripper command (binary: open or closed)
        gripper_target = torch.where(gripper_cmd > 0.5, 0.8, 0.0)  # 0.8 = closed
        self.scene["robot"].set_joint_position_target(
            gripper_target.expand(-1, 6),  # All finger joints
            joint_ids=list(range(6, 12)),
        )
