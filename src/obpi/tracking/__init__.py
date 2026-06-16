"""Tracking-data adapter stub.

This module provides a placeholder interface for integrating
StatsBomb tracking data (or other provider) into the OBPI pipeline.

Planned functionality:
- Load tracking frames aligned to event timestamps
- Infer velocity from high-frequency (25 Hz) positional data
- Compute movement metrics directly from tracking instead of event-derived estimates
"""

from obpi.tracking.stub import TrackingAdapter

__all__ = ["TrackingAdapter"]
