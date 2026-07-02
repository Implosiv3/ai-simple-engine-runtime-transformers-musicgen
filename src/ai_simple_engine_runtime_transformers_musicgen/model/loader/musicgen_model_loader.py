from ai_simple_engine_runtime_transformers_musicgen.model.loaded_musicgen_model import LoadedMusicgenModel
from ai_simple_engine_runtime_transformers_musicgen.consts import MUSICGEN_MODEL_FAMILY
from ai_simple_engine_runtime_transformers_musicgen.model.info.musicgen_model_info import MusicgenModelInfo
from ai_simple_engine.models.loaded_model import LoadedModel
from ai_simple_engine.models.loaders.abstract import ModelLoader
from ai_simple_engine.models.installed_model import InstalledModel
from ai_simple_engine.device.base import Device
from transformers import AutoProcessor, MusicgenForConditionalGeneration


class MusicgenModelLoader(
    ModelLoader
):
    """
    The `musicgen` model loader that uses
    transformers.
    """
    
    @property
    def family(
        self
    ) -> str:
        return MUSICGEN_MODEL_FAMILY

    async def load(
        self,
        installed_model: InstalledModel,
        *,
        device: Device
    ) -> LoadedModel[LoadedMusicgenModel, MusicgenModelInfo]:
        processor = AutoProcessor.from_pretrained(installed_model.path)
        network = MusicgenForConditionalGeneration.from_pretrained(installed_model.path)
        
        network.to(device.identifier)

        return LoadedModel(
            installed_model = installed_model,
            instance = LoadedMusicgenModel(
                processor = processor,
                network = network
            ),
            info = MusicgenModelInfo(
                sample_rate = processor.feature_extractor.sampling_rate,
                frame_rate = installed_model.audio_encoder.config.frame_rate
            )
        )

    async def unload(
        self,
        model: LoadedModel
    ) -> None:
        del model