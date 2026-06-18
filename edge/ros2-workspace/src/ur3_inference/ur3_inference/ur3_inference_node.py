"""
UR3 TensorRT Inference Node (ROS 2)

Subscribes to wrist camera images and robot joint states,
runs the trained pick-and-place policy via TensorRT,
and publishes joint position commands to the UR3.

Runs at 200+ Hz on Jetson Orin or GPU PC.

ROS 2 Topics:
  Subscriptions:
    /ur3/wrist_camera/image_raw (sensor_msgs/Image) — RGB from wrist camera
    /ur3/joint_states (sensor_msgs/JointState) — Current joint positions/velocities

  Publications:
    /ur3/joint_commands (trajectory_msgs/JointTrajectoryPoint) — Target joint positions
    /ur3/gripper_command (std_msgs/Float64) — Gripper open/close command

Usage:
    ros2 run ur3_inference inference_node --ros-args \
        -p model_path:=/model/policy.trt \
        -p inference_rate:=200
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectoryPoint
from std_msgs.msg import Float64
from cv_bridge import CvBridge
import cv2


class UR3InferenceNode(Node):
    """Real-time inference node for UR3 pick-and-place policy."""

    def __init__(self):
        super().__init__('ur3_inference_node')

        # Parameters
        self.declare_parameter('model_path', '/model/policy.trt')
        self.declare_parameter('inference_rate', 200)  # Hz
        self.declare_parameter('action_scale_pos', 0.01)  # meters per step
        self.declare_parameter('action_scale_rot', 0.05)  # radians per step

        model_path = self.get_parameter('model_path').value
        self.inference_rate = self.get_parameter('inference_rate').value
        self.action_scale_pos = self.get_parameter('action_scale_pos').value
        self.action_scale_rot = self.get_parameter('action_scale_rot').value

        # Load TensorRT engine
        self.engine = self._load_tensorrt_engine(model_path)
        self.get_logger().info(f'Loaded TensorRT engine: {model_path}')

        # State buffers
        self.latest_image = None
        self.latest_joint_pos = np.zeros(6)
        self.latest_joint_vel = np.zeros(6)
        self.latest_gripper_state = 0.0
        self.bridge = CvBridge()

        # QoS for real-time (best effort, small queue)
        rt_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=1,
        )

        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/ur3/wrist_camera/image_raw',
            self._image_callback,
            rt_qos,
        )
        self.joint_sub = self.create_subscription(
            JointState,
            '/ur3/joint_states',
            self._joint_state_callback,
            rt_qos,
        )

        # Publishers
        self.joint_cmd_pub = self.create_publisher(
            JointTrajectoryPoint,
            '/ur3/joint_commands',
            rt_qos,
        )
        self.gripper_cmd_pub = self.create_publisher(
            Float64,
            '/ur3/gripper_command',
            rt_qos,
        )

        # Inference timer (runs at specified rate)
        timer_period = 1.0 / self.inference_rate
        self.inference_timer = self.create_timer(timer_period, self._inference_step)

        self.get_logger().info(
            f'UR3 inference node started at {self.inference_rate} Hz'
        )

    def _load_tensorrt_engine(self, model_path: str):
        """Load and initialize TensorRT engine."""
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit  # noqa: F401

            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(TRT_LOGGER)

            with open(model_path, 'rb') as f:
                engine = runtime.deserialize_cuda_engine(f.read())

            context = engine.create_execution_context()

            # Allocate GPU buffers
            # Input: flattened observation (6+6+1+7+64*64*3 = 12308)
            # Output: action (7)
            input_size = 12308 * 4  # float32
            output_size = 7 * 4

            d_input = cuda.mem_alloc(input_size)
            d_output = cuda.mem_alloc(output_size)
            stream = cuda.Stream()

            return {
                'engine': engine,
                'context': context,
                'd_input': d_input,
                'd_output': d_output,
                'stream': stream,
                'h_output': np.empty(7, dtype=np.float32),
            }

        except ImportError:
            self.get_logger().warn(
                'TensorRT not available — falling back to ONNX Runtime'
            )
            return self._load_onnx_fallback(model_path.replace('.trt', '.onnx'))

    def _load_onnx_fallback(self, onnx_path: str):
        """Fallback: use ONNX Runtime if TensorRT unavailable."""
        import onnxruntime as ort

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(onnx_path, providers=providers)

        return {'type': 'onnx', 'session': session}

    def _image_callback(self, msg: Image):
        """Store latest camera image."""
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        # Resize to 64x64 and normalize
        self.latest_image = cv2.resize(cv_image, (64, 64)).astype(np.float32) / 255.0

    def _joint_state_callback(self, msg: JointState):
        """Store latest joint state."""
        if len(msg.position) >= 6:
            self.latest_joint_pos = np.array(msg.position[:6], dtype=np.float32)
        if len(msg.velocity) >= 6:
            self.latest_joint_vel = np.array(msg.velocity[:6], dtype=np.float32)
        if len(msg.position) > 6:
            self.latest_gripper_state = float(msg.position[6])

    def _inference_step(self):
        """Run one inference step and publish commands."""
        if self.latest_image is None:
            return  # Wait for first image

        # Build observation vector
        obs = self._build_observation()

        # Run inference
        action = self._run_inference(obs)

        # Parse action into EE delta + gripper
        ee_delta = action[:6]
        gripper_cmd = action[6]

        # Convert EE delta to joint commands (simplified — production uses IK)
        # For now, publish delta as joint velocity-scaled targets
        joint_target = self.latest_joint_pos + ee_delta[:6] * self.action_scale_pos

        # Publish joint command
        cmd_msg = JointTrajectoryPoint()
        cmd_msg.positions = joint_target.tolist()
        cmd_msg.time_from_start.sec = 0
        cmd_msg.time_from_start.nanosec = int(1e9 / self.inference_rate)
        self.joint_cmd_pub.publish(cmd_msg)

        # Publish gripper command
        gripper_msg = Float64()
        gripper_msg.data = 1.0 if gripper_cmd > 0.5 else 0.0
        self.gripper_cmd_pub.publish(gripper_msg)

    def _build_observation(self) -> np.ndarray:
        """Flatten all observations into single vector for policy input."""
        # Object pose relative to EE — in production this comes from perception
        # For now, use a placeholder (real system would use pose estimation)
        object_pose_relative = np.zeros(7, dtype=np.float32)
        object_pose_relative[6] = 1.0  # Identity quaternion w component

        # Flatten camera to CHW format then to 1D
        camera_flat = self.latest_image.transpose(2, 0, 1).flatten()

        obs = np.concatenate([
            self.latest_joint_pos,           # 6
            self.latest_joint_vel,           # 6
            np.array([self.latest_gripper_state], dtype=np.float32),  # 1
            object_pose_relative,            # 7
            camera_flat,                     # 64*64*3 = 12288
        ])

        return obs.astype(np.float32)

    def _run_inference(self, obs: np.ndarray) -> np.ndarray:
        """Run policy inference (TensorRT or ONNX fallback)."""
        if isinstance(self.engine, dict) and self.engine.get('type') == 'onnx':
            # ONNX Runtime fallback
            session = self.engine['session']
            input_name = session.get_inputs()[0].name
            result = session.run(None, {input_name: obs.reshape(1, -1)})
            return result[0].flatten()

        # TensorRT inference
        import pycuda.driver as cuda

        ctx = self.engine['context']
        d_input = self.engine['d_input']
        d_output = self.engine['d_output']
        stream = self.engine['stream']
        h_output = self.engine['h_output']

        # Copy input to GPU
        cuda.memcpy_htod_async(d_input, obs, stream)

        # Execute
        ctx.execute_async_v2([int(d_input), int(d_output)], stream.handle)

        # Copy output from GPU
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()

        return h_output.copy()


def main(args=None):
    rclpy.init(args=args)
    node = UR3InferenceNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
