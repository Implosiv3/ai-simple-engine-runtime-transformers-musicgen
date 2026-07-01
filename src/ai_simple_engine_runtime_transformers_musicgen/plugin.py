from ai_simple_engine_runtime_transformers_musicgen.model.executor.transformers_musicgen_model_executor import TransformersMusicgenModelExecutor
from ai_simple_engine.models.executor.registry.family_model_executor_registry import FamilyModelExecutorRegistry
from ai_simple_engine_runtime_transformers_musicgen.model.loader.musicgen_model_loader import MusicgenModelLoader
from ai_simple_engine_runtime_transformers_musicgen.consts import MUSICGEN_MODEL_FAMILY
from ai_simple_engine.engine_builder import EngineBuilder
from ai_simple_engine.plugins.plugin import Plugin


class TransformersRuntimeMusicgenPlugin(
    Plugin
):
    """
    The plugin to add the `musicgen` model
    functionality.

    This plugin includes:
    - `MusicgenModelLoader`
    - `TransformersMusicgenModelExecutor`
    """

    def register(
        self,
        builder: EngineBuilder
    ):
        (
            builder
            .add_model_loader(MusicgenModelLoader())
        )

        """
        We obtain the registry that handles the
        model executors by the model's family,
        and we register our specific model
        executor that uses transformers.
        """
        registry = builder.get_or_add_service(FamilyModelExecutorRegistry)

        registry.register(
            MUSICGEN_MODEL_FAMILY,
            TransformersMusicgenModelExecutor()
        )
