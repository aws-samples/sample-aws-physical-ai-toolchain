"""Fail the image build if real AV1 CPU decoding or frame selection is broken.

av1-two-frames.mp4 contains two generated 64x64 frames: black, then white.
No external dataset, model, GPU, network or token is needed for this check.
"""
import json
from pathlib import Path

from torchcodec.decoders import VideoDecoder

video = Path(__file__).with_name("av1-two-frames.mp4")
decoder = VideoDecoder(str(video), device="cpu", dimension_order="NHWC")
if decoder.metadata.codec != "av1":
    raise RuntimeError(f"Expected AV1 regression fixture, got {decoder.metadata.codec}")
# Out-of-order, repeated indices exercise the same get_frames_at API as training.
frames = decoder.get_frames_at(indices=[1, 0, 1]).data
if tuple(frames.shape) != (3, 64, 64, 3):
    raise RuntimeError(f"Unexpected decoded shape: {tuple(frames.shape)}")
if not (frames[0].min().item() >= 250 and frames[1].max().item() <= 5
        and frames[2].min().item() >= 250):
    raise RuntimeError("AV1 frame pixels or requested frame order are incorrect")
print(json.dumps({"av1_cpu_decode": "passed", "frame_indices": [1, 0, 1],
                  "shape": list(frames.shape)}))
