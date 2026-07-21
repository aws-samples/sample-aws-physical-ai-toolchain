# Isaac Sim Container for Scene Generation
#
# Used by the scene-generation stage of the OSMO workflow.
# Runs Cosmos-enhanced or procedural scene generation headless.
#
# Build:
#   docker build -t isaac-sim-scenegen:latest .
#
# Run:
#   docker run --gpus all -v /data:/storage isaac-sim-scenegen:latest \
#     python /workspace/scripts/generate_scenes.py --output-dir /storage/scenes/ --num-scenes 1000

# Base: Official Isaac Sim container from NGC
FROM nvcr.io/nvidia/isaac-sim:4.5.0

# Isaac Sim ships its own Python at /isaac-sim/kit/python/bin/python3 — there is
# no system `pip`/`python` on PATH. Use the bundled interpreter explicitly.
RUN /isaac-sim/kit/python/bin/python3 -m pip install --no-cache-dir \
    requests>=2.31 \
    boto3>=1.34 \
    pyyaml>=6.0

# Make Isaac Sim's Python available as system python3/python (for the entrypoint).
RUN ln -sf /isaac-sim/kit/python/bin/python3 /usr/local/bin/python3 \
    && ln -sf /isaac-sim/kit/python/bin/python3 /usr/local/bin/python

# Copy scene generation scripts
COPY training/scripts/generate_scenes.py /workspace/scripts/generate_scenes.py

WORKDIR /workspace

ENTRYPOINT ["python"]
CMD ["/workspace/scripts/generate_scenes.py", "--help"]
