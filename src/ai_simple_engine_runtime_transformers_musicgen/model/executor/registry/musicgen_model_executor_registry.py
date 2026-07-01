from ai_simple_engine_runtime_transformers_musicgen.model.loaded_musicgen_model import LoadedMusicgenModel
from ai_simple_engine_runtime_transformers_musicgen.model.executor.abstract import MusicgenModelExecutorAbstract
from ai_simple_engine.models.loaded_model import LoadedModel
from ai_simple_engine.models.executor.registry.family_model_executor_registry import FamilyModelExecutorRegistry


class MusicgenModelExecutorRegistry(
    FamilyModelExecutorRegistry[
        LoadedModel[LoadedMusicgenModel],
        MusicgenModelExecutorAbstract
    ]
):

    pass
    

