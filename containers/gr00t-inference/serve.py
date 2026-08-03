#!/usr/bin/env python3
"""SageMaker real-time inference server for a fine-tuned GR00T N1.6 policy.

Serves the model SageMaker extracts to /opt/ml/model (the model.tar.gz from a
Lab 1 training job) over the BYOC contract: GET /ping (health) + POST /invocations
on port 8080. Loads the policy once at startup and predicts an action from an
observation (wrist image + robot state + task string).

The custom UR3 embodiment (NEW_EMBODIMENT) needs its modality config registered and
the processor's modality_configs + norm_params patched from the checkpoint's
dataset_statistics.json — the base model's processor only knows pre-trained
embodiments. /opt/ml/model is read-only on SageMaker, so the model is merged into a
writable dir first.

Request (POST /invocations, JSON):
    {"state": [j0..j5, gripper], "task": "pick up the cube",
     "image": "<base64>"}            # or "images": [<b64>,...] / {cam: <b64>}
Response: {"actions": [[...]], "action_dim": 7}
"""
import base64, io, json, logging, os
import numpy as np
import torch
from flask import Flask, Response, request
from PIL import Image

# If flash_attn is installed but broken, drop it so transformers doesn't try to
# load a broken .so and crash at import.
try:
    import flash_attn  # noqa: F401
except Exception as e:
    import sys as _sys
    print(f"WARNING: flash_attn not usable ({e}), cleaning up", flush=True)
    for _k in list(_sys.modules.keys()):
        if "flash_attn" in _k:
            del _sys.modules[_k]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("groot-serve")
app = Flask(__name__)
policy = None
model_config = None
modality_cfg = None


def load_model():
    global policy, model_config, modality_cfg
    model_path = os.environ.get("GROOT_MODEL_PATH", "nvidia/GR00T-N1.6-3B")
    action_dim = int(os.environ.get("GROOT_ACTION_DIM", "7"))
    state_dim = int(os.environ.get("GROOT_STATE_DIM", "7"))
    embodiment = os.environ.get("GROOT_EMBODIMENT", "NEW_EMBODIMENT")

    # If SageMaker extracted a fine-tuned model, use it. /opt/ml/model is read-only,
    # so merge it into a writable dir (and overlay onto the base model's processor
    # files if the fine-tune didn't include them).
    sm_model_dir = "/opt/ml/model"
    if os.path.isdir(sm_model_dir) and any(
        f.endswith((".bin", ".safetensors", "config.json")) for f in os.listdir(sm_model_dir)
    ):
        logger.info(f"Using SageMaker model dir: {sm_model_dir}")
        import shutil
        writable_dir = "/tmp/groot-model"
        if os.path.exists(writable_dir):
            shutil.rmtree(writable_dir)

        processor_markers = ["preprocessor_config.json", "processor_config.json", "tokenizer_config.json"]
        has_processor = any(os.path.exists(os.path.join(sm_model_dir, f)) for f in processor_markers)

        if not has_processor:
            base_model_id = os.environ.get("GROOT_BASE_MODEL", "nvidia/GR00T-N1.6-3B")
            logger.info(f"Fine-tuned model missing processor files, copying from {base_model_id}...")
            from huggingface_hub import snapshot_download
            base_path = snapshot_download(base_model_id)
            shutil.copytree(base_path, writable_dir, symlinks=False)
            logger.info(f"Overlaying fine-tuned weights from {sm_model_dir}...")
            for item in os.listdir(sm_model_dir):
                src = os.path.join(sm_model_dir, item)
                dst = os.path.join(writable_dir, item)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
                elif os.path.isdir(src):
                    if os.path.exists(dst):
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)
        else:
            shutil.copytree(sm_model_dir, writable_dir)

        model_path = writable_dir

    logger.info(f"Loading GR00T from {model_path} (embodiment={embodiment}); "
                f"cuda={torch.cuda.is_available()}")

    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.types import ModalityConfig, ActionConfig, ActionRepresentation, ActionType, ActionFormat
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from gr00t.model.gr00t_n1d6.processing_gr00t_n1d6 import Gr00tN1d6Processor
    from gr00t.configs.data.embodiment_configs import register_modality_config

    tag = EmbodimentTag[embodiment]

    camera_names = os.environ.get("GROOT_CAMERAS", "wrist").split(",")
    max_video_frames = int(os.environ.get("GROOT_MAX_VIDEO_FRAMES", "1"))
    video_delta_indices = list(range(0, -max_video_frames, -1)) or [0]
    logger.info(f"Cameras: {camera_names}, frames/camera: {max_video_frames}")

    # Matches the UR3 embodiment used at training time (training/gr00t/convert_zarr_to_lerobot.py
    # + containers/gr00t-training): 6-DOF arm + gripper, EEF velocity-delta actions.
    ur3_config = {
        "video": ModalityConfig(delta_indices=video_delta_indices, modality_keys=camera_names),
        "state": ModalityConfig(delta_indices=[0], modality_keys=["arm", "gripper"]),
        "action": ModalityConfig(
            delta_indices=list(range(0, 16)),
            modality_keys=["arm", "gripper"],
            action_configs=[
                ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.EEF, format=ActionFormat.DEFAULT),
                ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.EEF, format=ActionFormat.DEFAULT),
            ],
        ),
        "language": ModalityConfig(delta_indices=[0], modality_keys=["annotation.human.action.task_description"]),
    }

    register_modality_config(ur3_config, embodiment_tag=tag)

    # The base model's processor only has pre-trained embodiments — inject ours for
    # construction, then restore the original method.
    _orig = Gr00tN1d6Processor.get_modality_configs
    def _patched(self):
        configs = _orig(self)
        if tag.value not in configs:
            configs[tag.value] = ur3_config
        return configs
    Gr00tN1d6Processor.get_modality_configs = _patched

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = Gr00tPolicy(embodiment_tag=tag, model_path=model_path, device=device)
    Gr00tN1d6Processor.get_modality_configs = _orig

    # The state_action_processor keeps its own modality_configs + norm_params dicts.
    sap = policy.processor.state_action_processor
    sap.use_relative_action = False  # actions are velocity deltas, decoded directly
    policy.processor.use_relative_action = False

    if hasattr(sap, "modality_configs") and tag.value not in sap.modality_configs:
        sap.modality_configs[tag.value] = ur3_config

    if hasattr(sap, "norm_params") and tag.value not in sap.norm_params:
        stats_path = os.path.join(model_path, "experiment_cfg", "dataset_statistics.json")

        def _params(s, dim):
            return {"min": np.array(s["min"], dtype=np.float32), "max": np.array(s["max"], dtype=np.float32),
                    "mean": np.array(s["mean"], dtype=np.float32), "std": np.array(s["std"], dtype=np.float32),
                    "dim": np.array([dim])}

        def _rel_params(s, dim):
            first = lambda v: v[0] if isinstance(v[0], list) else v
            return {"min": np.array(first(s["min"]), dtype=np.float32), "max": np.array(first(s["max"]), dtype=np.float32),
                    "mean": np.array(first(s["mean"]), dtype=np.float32), "std": np.array(first(s["std"]), dtype=np.float32),
                    "dim": np.array([dim])}

        if os.path.exists(stats_path):
            with open(stats_path) as f:
                all_stats = json.load(f)
            emb_key = embodiment.lower()
            if emb_key not in all_stats:
                emb_key = list(all_stats.keys())[0]
            stats = all_stats[emb_key]
            logger.info(f"Loaded dataset statistics (key={emb_key})")
            sap.norm_params[tag.value] = {
                "state": {"arm": _params(stats["state"]["arm"], 6), "gripper": _params(stats["state"]["gripper"], 1)},
                "action": {"arm": _params(stats["action"]["arm"], 6), "gripper": _params(stats["action"]["gripper"], 1)},
                "relative_action": {"arm": _rel_params(stats["relative_action"]["arm"], 6),
                                     "gripper": _params(stats["action"]["gripper"], 1)},
            }
        else:
            logger.warning(f"No dataset_statistics.json at {stats_path}; using identity norm")
            def _ident(dim):
                return {"min": np.zeros(dim, dtype=np.float32), "max": np.ones(dim, dtype=np.float32),
                        "mean": np.zeros(dim, dtype=np.float32), "std": np.ones(dim, dtype=np.float32),
                        "dim": np.array([dim])}
            sap.norm_params[tag.value] = {
                "state": {"arm": _ident(6), "gripper": _ident(1)},
                "action": {"arm": _ident(6), "gripper": _ident(1)},
                "relative_action": {"arm": _ident(6), "gripper": _ident(1)},
            }

    modality_cfg = policy.get_modality_config()
    model_config = {"action_dim": action_dim, "state_dim": state_dim, "model_path": model_path}
    logger.info("GR00T model loaded and ready")


@app.route("/ping", methods=["GET"])
def ping():
    return Response(status=200 if policy else 503)


def _decode_image(b64):
    return np.array(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"))


@app.route("/invocations", methods=["POST"])
def invocations():
    data = json.loads(request.data)
    state = np.array(data["state"], dtype=np.float32)
    task = data.get("task", "")

    video_keys = modality_cfg["video"].modality_keys if modality_cfg else ["wrist"]
    expected_t = len(modality_cfg["video"].delta_indices) if modality_cfg else 1

    def _pad(frames):
        if len(frames) < expected_t:
            return [frames[0]] * (expected_t - len(frames)) + frames
        return frames[-expected_t:]

    video_dict = {}
    if "image" in data and "images" not in data:
        vt = np.stack(_pad([_decode_image(data["image"])]), axis=0)[np.newaxis].astype(np.uint8)
        for vk in video_keys:
            video_dict[vk] = vt
    elif "images" in data and isinstance(data["images"], list):
        vt = np.stack(_pad([_decode_image(i) for i in data["images"]]), axis=0)[np.newaxis].astype(np.uint8)
        for vk in video_keys:
            video_dict[vk] = vt
    elif "images" in data and isinstance(data["images"], dict):
        for vk in video_keys:
            cam = data["images"].get(vk)
            if cam is None:
                frames = [np.zeros((224, 224, 3), dtype=np.uint8)] * expected_t
            elif isinstance(cam, str):
                frames = _pad([_decode_image(cam)])
            else:
                frames = _pad([_decode_image(i) for i in cam])
            video_dict[vk] = np.stack(frames, axis=0)[np.newaxis].astype(np.uint8)
    else:
        return Response(json.dumps({"error": "Provide 'image' (base64) or 'images' (list or dict)"}),
                        status=400, mimetype="application/json")

    state_keys = modality_cfg["state"].modality_keys if modality_cfg else ["arm", "gripper"]
    arm_dim = len(state) - 1
    state_arrays = {}
    if len(state_keys) == 1:
        state_arrays[state_keys[0]] = state.reshape(1, 1, -1)
    else:
        state_arrays[state_keys[0]] = state[:arm_dim].reshape(1, 1, -1)
        state_arrays[state_keys[-1]] = state[arm_dim:].reshape(1, 1, -1)

    lang_keys = modality_cfg["language"].modality_keys if modality_cfg else ["annotation.human.action.task_description"]
    obs = {"video": video_dict, "state": state_arrays, "language": {lk: [[task]] for lk in lang_keys}}

    with torch.no_grad():
        action, _info = policy.get_action(obs)
    action_np = np.concatenate([np.array(v) for v in action.values()], axis=-1) if isinstance(action, dict) else np.array(action)

    return Response(json.dumps({"actions": action_np.tolist(), "action_dim": model_config["action_dim"]}),
                    status=200, mimetype="application/json")


if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=8080)
