from ai_simple_engine.models.info.abstract import ModelInfo
from dataclasses import dataclass


@dataclass(frozen = True)
class MusicgenModelInfo(
    ModelInfo
):
    sample_rate: int
    frame_rate: int