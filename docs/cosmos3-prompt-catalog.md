# Cosmos 3 Prompt Catalog

Prompts tested against `nvidia/Cosmos3-Super` via vLLM-Omni V2V endpoint.
All use the same input reference: `episode_000000.mp4` (UR3 wrist camera, 81 frames,
5fps, 640x480 — pick-and-place of red block onto yellow post-it).

**S3 location:** `s3://<DATASETS_BUCKET>/cosmos-samples/`

---

## Generation Parameters (constant across all runs)

```
model: nvidia/Cosmos3-Super
size: 1280x720
num_frames: 189
fps: 24
num_inference_steps: 35
guidance_scale: 6.0
max_sequence_length: 4096
flow_shift: 10.0
extra_params: {"condition_frame_indexes_vision":[0,1],"condition_video_keep":"first"}
input_reference: episode_000000.mp4 (type=video/mp4)
```

---

## Tested Prompts

### 1. Generic warehouse (first attempt — too vague)

| | |
|---|---|
| **File** | `augmented_episode_000000.mp4` |
| **Seed** | 42 |
| **Prompt** | `A robot arm performing pick and place in a photorealistic industrial warehouse with fluorescent lighting and scratched metal surfaces` |
| **Result** | ❌ Generated hallucinated scene — objects moved, hand appeared. Too much creative freedom. |
| **Lesson** | Don't describe actions vaguely. Don't mix environment description with action description loosely. |

### 2. Environment-only description (second attempt — still too loose)

| | |
|---|---|
| **File** | `augmented_episode_000000_prompt2.mp4` |
| **Seed** | 42 |
| **Prompt** | `Dark matte table with subtle wear marks and fine scratches, colorful wooden blocks, overhead workshop lighting` |
| **Result** | ❌ Added scratches to table (good!) but hallucinated rolling blocks and a hand. Objects not preserved. |
| **Lesson** | Pure environment descriptions cause the model to hallucinate new actions since no action was specified. |

### 3. Specific task description (SUCCESS)

| | |
|---|---|
| **File** | `augmented_specific_prompt.mp4` |
| **Seed** | 100 |
| **Prompt** | `A UR3 robot arm with a Robotiq gripper reaches down to a dark matte table, grasps a small red wooden block, lifts it slowly, and places it onto a yellow sticky note approximately 6 inches away. Top-down wrist camera view. Colorful wooden blocks are scattered on the table. Smooth deliberate motion.` |
| **Result** | ✅ Coherent pick-and-place trajectory. Robotiq gripper visible, grasps red/orange block, places on yellow target. Realistic motion. |
| **Lesson** | Specific robot + specific action + specific target + camera angle + motion style = coherent generation. |

### 4. Blue block variant (SUCCESS)

| | |
|---|---|
| **File** | `augmented_blue_block.mp4` |
| **Seed** | 200 |
| **Prompt** | `A UR3 robot arm with a Robotiq gripper reaches down to a dark matte table, grasps a small blue wooden block, lifts it slowly, and places it onto a yellow sticky note. Top-down wrist camera view. Colorful wooden blocks scattered on the table. Smooth deliberate motion.` |
| **Result** | ✅ Coherent pick-and-place of blue block. Gripper visible, correct motion. |
| **Lesson** | Changing one word (red→blue) produces a valid variant with the new block color. |

### 5. Dim lighting variant (SUCCESS)

| | |
|---|---|
| **File** | `augmented_dim_lighting.mp4` |
| **Seed** | 300 |
| **Prompt** | `A UR3 robot arm with a Robotiq gripper picks up a small red wooden block from a dark table and places it on a yellow sticky note. Top-down wrist camera view. Dim overhead lighting with strong shadows. Colorful blocks on the table.` |
| **Result** | ✅ (uploaded, pending visual review) |
| **Lesson** | Lighting variation with specific task description. |

---

## Suggested Additional Prompts (not yet tested)

### Approach angle variation
```
A UR3 robot arm with a Robotiq gripper approaches from the left side, grasps a small
red wooden block from a dark matte table, and places it on a yellow sticky note to the
right. Top-down wrist camera view. Multiple colorful blocks visible. Smooth motion.
```

### Speed variation
```
A UR3 robot arm with a Robotiq gripper quickly grasps a red wooden block from a dark
table, lifts it high, then precisely places it on a yellow sticky note. Top-down wrist
camera view. Colorful blocks on table. Swift confident motion.
```

### Different target
```
A UR3 robot arm with a Robotiq gripper picks up a small green wooden block from a dark
matte table and places it into a small cardboard box on the right side. Top-down wrist
camera view. Colorful blocks scattered on table. Careful deliberate motion.
```

### Cluttered environment
```
A UR3 robot arm with a Robotiq gripper navigates around several obstacles, grasps a
red wooden block from a crowded dark table, and places it on a yellow sticky note.
Top-down wrist camera view. Many colorful blocks closely packed. Precise cautious motion.
```

### Failure case (for negative training data)
```
A UR3 robot arm with a Robotiq gripper attempts to grasp a small red wooden block but
the block slips from the gripper and falls back onto the dark table. Top-down wrist
camera view. Colorful blocks on table. The grasp fails.
```

---

## Prompting Best Practices (learned from testing)

1. **Be specific about the robot:** "UR3 robot arm with a Robotiq gripper" — not just "robot arm"
2. **Describe the complete action sequence:** "reaches → grasps → lifts → places" — step by step
3. **Specify the target precisely:** "yellow sticky note approximately 6 inches away" — concrete
4. **State the camera angle:** "Top-down wrist camera view" — matches your actual data
5. **Describe the scene context:** "Colorful wooden blocks scattered on the table" — grounds it
6. **Specify motion style:** "Smooth deliberate motion" or "Swift confident motion"
7. **DON'T** use vague environment descriptions without an action — model hallucinates
8. **DON'T** give editing instructions ("add scratches", "leave blocks unchanged") — it's a generator, not an editor
9. **DO** change seed for different trajectories of the same prompt
10. **DO** vary one element at a time (block color, lighting, speed) to build a diverse dataset

---

## How This Fits the Pipeline

Cosmos 3 Super generates **novel synthetic demonstrations** — new plausible trajectories
of the same task. Each generated video is a synthetic training episode.

**The trade-off:** These videos have no action labels (joint positions/velocities).
They're useful for:
- Vision backbone pre-training (teach the model what "pick and place" looks like)
- Data diversity (vary conditions the policy trains against)
- Evaluation (does a generated rollout look plausible?)

For **action-paired** augmentation (same video restyled with actions preserved), use
Cosmos Transfer 2.5 (different model, different setup — see `docs/cosmos3-validated-runbook.md`).
