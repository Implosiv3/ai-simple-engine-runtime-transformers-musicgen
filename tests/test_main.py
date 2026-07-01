"""
A simple test to verify that pytes is working and
the tests are being detected.
"""
import pytest


@pytest.mark.mandatory
def test_main_flow():
    from ai_simple_engine_runtime_transformers_musicgen.plugin import TransformersRuntimeMusicgenPlugin
    from ai_simple_engine.engine_builder import EngineBuilder

    engine_builder = EngineBuilder()
    engine_builder.add_plugin(TransformersRuntimeMusicgenPlugin())