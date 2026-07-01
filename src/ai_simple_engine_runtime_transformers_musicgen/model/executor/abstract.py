from ai_simple_engine_runtime_transformers_musicgen.model.loaded_musicgen_model import LoadedMusicgenModel
from ai_simple_engine.models.loaded_model import LoadedModel
from ai_simple_engine.models.executor.abstract import ModelExecutor
from ai_simple_engine.types.audio import Audio
from abc import ABC, abstractmethod


class MusicgenModelExecutorAbstract(
    ModelExecutor,
    ABC
):

    @abstractmethod
    async def generate(
        self,
        model: LoadedModel[LoadedMusicgenModel],
        prompt: str,
        *,
        max_new_tokens: int,
        guidance_scale: float
    ) -> Audio:
        ...