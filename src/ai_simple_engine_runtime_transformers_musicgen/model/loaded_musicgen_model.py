from transformers import AutoProcessor, MusicgenForConditionalGeneration
from dataclasses import dataclass


@dataclass(frozen = True)
class LoadedMusicgenModel:
    """
    The `musicgen` model, loaded, that uses
    transformers.
    """

    processor: AutoProcessor
    network: MusicgenForConditionalGeneration