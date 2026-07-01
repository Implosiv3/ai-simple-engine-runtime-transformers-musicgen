from ai_simple_engine_runtime_transformers_musicgen.model.executor.abstract import MusicgenModelExecutorAbstract
from ai_simple_engine_runtime_transformers_musicgen.model.loaded_musicgen_model import LoadedMusicgenModel
from ai_simple_engine.models.loaded_model import LoadedModel
from ai_simple_engine.types.audio import Audio


class TransformersMusicgenModelExecutor(
    MusicgenModelExecutorAbstract
):

    async def generate(
        self,
        model: LoadedModel[LoadedMusicgenModel],
        prompt: str,
        *,
        max_new_tokens: int,
        guidance_scale: float
    ) -> Audio:
        processor = model.processor
        network = model.network

        inputs = processor(
            text = [prompt],
            return_tensors = 'pt'
        )

        inputs = {
            k: v.to(network.device)
            for k, v in inputs.items()
        }

        output = network.generate(
            **inputs,
            max_new_tokens = max_new_tokens,
            guidance_scale = guidance_scale
        )

        samples = output.cpu().numpy()[0, 0]

        return Audio(
            samples = samples,
            sample_rate = 32000
        )