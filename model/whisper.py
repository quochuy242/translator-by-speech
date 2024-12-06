from transformers import (
    WhisperForConditionalGeneration,
    WhisperFeatureExtractor,
    WhisperTokenizer,
)


def load_model(model_id: str = "tiny"):
    model = WhisperForConditionalGeneration.from_pretrained(
        f"openai/whisper-{model_id}", use_cache=False
    )
    model.config.forced_decoder_ids = None
    model.suppress_tokens = None
    return model


def load_feature_extractor(model_id: str = "tiny"):
    return WhisperFeatureExtractor.from_pretrained(f"openai/whisper-{model_id}")


def load_tokenizer(model_id: str = "tiny"):
    return WhisperTokenizer.from_pretrained(
        f"openai/whisper-{model_id}", language="vi", task="transcribe"
    )
