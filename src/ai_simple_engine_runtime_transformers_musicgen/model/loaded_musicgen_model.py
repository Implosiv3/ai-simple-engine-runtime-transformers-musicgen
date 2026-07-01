from transformers import AutoProcessor, MusicgenForConditionalGeneration
from dataclasses import dataclass


@dataclass(frozen = True)
class LoadedMusicgenModel:

    processor: AutoProcessor
    network: MusicgenForConditionalGeneration