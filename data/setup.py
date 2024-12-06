from utils import const
from dataclasses import dataclass
from typing import Any


def prepare_ds(batch, feature_extractor, tokenizer):
    audio = batch["audio"]

    # Compute input length
    batch["input_length"] = len(audio["array"])

    # Compute log-Mel input features
    batch["input_features"] = feature_extractor(
        audio["array"], sampling_rate=const.SAMPLE_RATE
    ).input_features[0]

    # Compute labels and labels length
    batch["labels"] = tokenizer(batch["transcription"]).input_ids
    batch["labels_length"] = len(batch["labels"])

    return batch


def filter_audios(input_length):
    # Filter out zero audio and too long audios
    return 0 < input_length < const.SAMPLE_RATE * const.MAX_AUDIO_LENGTH


def filter_labels(labels_length):
    # Filter out too long labels
    return labels_length < const.MAX_LABEL_LENGTH


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    feature_extractor: Any
    tokenizer: Any

    def __call__(self, features):
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), const.DUMMY_TOKEN
        )

        if (labels[:, 0] == self.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch
