"""vla_pipeline -- an account-portable SageMaker Pipeline for VLA capability eval.

Public surface:
  load_config / PipelineConfig / ConfigError  (config.py)
  build_pipeline / build_parameters           (pipeline.py)
"""
from __future__ import annotations

from .config import ConfigError, PipelineConfig, load_config

__all__ = ["ConfigError", "PipelineConfig", "load_config"]
__version__ = "0.1.0"
