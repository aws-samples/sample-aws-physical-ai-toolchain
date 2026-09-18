"""Enforce GPU-job deadlines that the pinned local SDK otherwise ignores."""
from __future__ import annotations

import signal
import time


def install(client, seconds):
    original = client.create_training_job

    def create_training_job(*args, **kwargs):
        # LocalPipelineSession executes synchronously on the worker's main thread.
        # Preserve the worker's total deadline while bounding this individual job.
        previous, interval = signal.getitimer(signal.ITIMER_REAL)
        limit = min(seconds, previous) if previous else seconds
        started = time.monotonic()
        print(f"[local] GPU job deadline: {limit:.0f}s "
              f"({kwargs.get('TrainingJobName') or (args[0] if args else 'local training job')})",
              flush=True)
        signal.setitimer(signal.ITIMER_REAL, limit)
        try:
            return original(*args, **kwargs)
        finally:
            elapsed = time.monotonic() - started
            signal.setitimer(signal.ITIMER_REAL,
                            max(0.001, previous - elapsed) if previous else 0, interval)

    client.create_training_job = create_training_job
