# Cosmos Sample Videos

Pre-generated demo outputs so you can see the results without GPU infrastructure.

---

## Lab 3: Cosmos World Generation (Predict mode — Cosmos 3 Super 64B)

Generates **new** synthetic demonstrations from a reference video + text prompt.
Different trajectories, new content. Uses `vllm/vllm-omni:cosmos3` on p5.48xlarge.

| File | What | Duration |
|------|------|----------|
| `original_episode_000000.mp4` | Input: UR3 wrist camera, pick-and-place (81 frames, 5fps, 640x480) | 16 sec |
| `augmented_specific_prompt.mp4` | Output: Generated pick-and-place (189 frames, 24fps, 1280x720) | 8 sec |
| `augmented_blue_block.mp4` | Output: Same task, blue block variant | 8 sec |
| `augmented_dim_lighting.mp4` | Output: Same task, dim lighting | 8 sec |

### Configuration used:
```
Model: nvidia/Cosmos3-Super (64B)
Server: vllm/vllm-omni:cosmos3
Instance: p5.48xlarge (8x H100 80GB), EC2 Capacity Block
Endpoint: POST /v1/videos/sync
Parameters:
  size: 1280x720
  num_frames: 189
  fps: 24
  num_inference_steps: 35
  guidance_scale: 6.0
  max_sequence_length: 4096
  flow_shift: 10.0
  extra_params: {"condition_frame_indexes_vision":[0,1],"condition_video_keep":"first"}
Generation time: ~5 min per video
```

### Prompts used:

**augmented_specific_prompt.mp4** (seed=100):
> A UR3 robot arm with a Robotiq gripper reaches down to a dark matte table, grasps a small red wooden block, lifts it slowly, and places it onto a yellow sticky note approximately 6 inches away. Top-down wrist camera view. Colorful wooden blocks are scattered on the table. Smooth deliberate motion.

**augmented_blue_block.mp4** (seed=200):
> A UR3 robot arm with a Robotiq gripper reaches down to a dark matte table, grasps a small blue wooden block, lifts it slowly, and places it onto a yellow sticky note. Top-down wrist camera view. Colorful wooden blocks scattered on the table. Smooth deliberate motion.

**augmented_dim_lighting.mp4** (seed=300):
> A UR3 robot arm with a Robotiq gripper picks up a small red wooden block from a dark table and places it on a yellow sticky note. Top-down wrist camera view. Dim overhead lighting with strong shadows. Colorful blocks on the table.

---

## Lab 4: Cosmos Transfer (Transfer mode — Cosmos Transfer 2.5 NIM)

Restyles **existing** video preserving geometry/motion. Same frames, different visual
appearance. Uses `nvcr.io/nim/nvidia/cosmos-transfer2.5-2b` NIM on p5.48xlarge.

| File | What | Duration |
|------|------|----------|
| `transfer_original_episode009.mp4` | Input: UR3 wrist camera, full episode (242 frames, 5fps, 640x480) | 48 sec |
| `transfer_result_episode009.mp4` | Output: Restyled first 93 frames (factory lighting/surface) | ~6 sec |

### Configuration used:
```
Model: nvidia/Cosmos-Transfer2.5-2B (NIM container)
Container: nvcr.io/nim/nvidia/cosmos-transfer2.5-2b:latest
Instance: p5.48xlarge (8x H100 80GB), Capacity Block us-west-2
NIM Profile: auto-selected H100 latency FP8
Endpoint: POST /v1/infer
Parameters:
  edge: {}              (auto-extracts edge map from input for geometry control)
  num_steps: 35
  guidance: 3
Input: first 93 frames of episode_000009.mp4 at native 5fps
Output: base64 JSON (decode data["b64_video"], then ffmpeg -c:v libx264 for Mac playback)
Generation time: ~8-10 min
```

### Prompt used:
> Industrial factory with scratched metal table and fluorescent lighting

### Notes on the output:
- Geometry/block positions are preserved correctly
- Colors shift toward the prompted environment (lighter table, different lighting)
- Output plays at 16fps (NIM default) — re-encode to 5fps to match original timing
- Quality is from the FP8 "latency" profile — BF16 "throughput" profile should be better

---

## Next Steps (for contributors improving Transfer quality)

The Transfer NIM is functional but output quality needs tuning. The instance
(`i-0b57fd45b65895ff0`, us-west-2, p5.48xlarge) has the NIM loaded and ready
for experimentation. Things to try:

1. **Force BF16 throughput profile:** Restart container with `NIM_MODEL_PROFILE=throughput`
   (more compute per frame but should produce sharper, more color-accurate output)

2. **Try `depth` control mode:** Replace `"edge": {}` with `"depth": {}` in the payload.
   Depth maps preserve 3D structure better than edge maps and may give more faithful colors.

3. **Increase guidance:** Try `"guidance": 7` or `"guidance": 10` to make the prompt
   more dominant (may produce stronger style but better structure preservation).

4. **Combine controls:** Try `"edge": {"control_weight": 0.5}, "vis": {"control_weight": 0.5}`
   for multi-modal control (edge for structure + vis for appearance).

5. **Fix framerate:** Ensure input is encoded at 16fps before sending (the NIM's native rate).
   The output then plays correctly without re-encoding. Formula: take your 5fps input,
   re-encode to 16fps (duplicating frames), send 93 frames, get 93 frames back at 16fps.

6. **Resolution:** Try adding `"resolution": "720"` to the payload for 720p output.

### Reproduction steps (if you have a p5.48xlarge):
```bash
# 1. Pull NIM (requires NGC API key)
docker pull nvcr.io/nim/nvidia/cosmos-transfer2.5-2b:latest

# 2. Run (no NIM_MODEL_PROFILE = auto-selects based on GPU)
docker run -d --name cosmos-transfer-nim --gpus all --ipc=host --shm-size=64g \
  -p 8000:8000 -e NGC_API_KEY=<key> -e HF_TOKEN=<token> \
  nvcr.io/nim/nvidia/cosmos-transfer2.5-2b:latest

# 3. Wait ~15 min for TRT engine download + model load
curl http://localhost:8000/v1/health/ready  # 200 = ready

# 4. Run inference (Python)
python3 -c "
import base64, requests, json
with open('input_video.mp4', 'rb') as f:
    video_b64 = base64.b64encode(f.read()).decode()
payload = {
    'prompt': 'Industrial factory with scratched metal table and fluorescent lighting',
    'video': video_b64,
    'edge': {},
    'num_steps': 35,
    'guidance': 3
}
resp = requests.post('http://localhost:8000/v1/infer', json=payload, timeout=900)
data = resp.json()
video = base64.b64decode(data['b64_video'])
with open('output_raw.mp4', 'wb') as f:
    f.write(video)
print(f'Output: {len(video)} bytes')
"

# 5. Convert for Mac playback
ffmpeg -y -i output_raw.mp4 -r 5 -c:v libx264 -crf 18 output_final.mp4
```

### HuggingFace licenses required:
- nvidia/Cosmos-Guardrail1
- nvidia/Cosmos-1.0-Guardrail
- nvidia/Cosmos-Predict2.5-2B
- nvidia/Cosmos-Transfer2.5-2B
