"""Dataset configuration helping to load
path: subset of URL on hugging face,
split: choose the type of dataset (e.g., train, test)
mode:
    if 0, normal case,
    if 1, just select audio and transcription fields,
    if 2, select audio, sentence field and rename the sentence field to transcription
"""

DATA_CONFIG = [
    {
        "path": "doof-ferb/infore1_25hours",
        "split": "train",
        "mode": 0,
    },
    {
        "path": "linhtran92/viet_bud500",
        "split": "train",
        "mode": 0,
    },
    {
        "path": "AILAB-VNUHCM/vivos",
        "split": "test",
        "mode": 2,
    },
    {
        "path": "mozilla-foundation/common_voice_12_0",
        "name": "vi",
        "split": "train",
        "mode": 2,
    },
    {
        "path": "NhutP/VSV-1100",
        "split": "train",
        "mode": 0,
    },
    {
        "path": "google/fleurs",
        "split": "train",
        "name": "vi_vn",
        "mode": 1,
    },
]


STREAMING = True
SAMPLE_RATE = 16000
WHISPER_ID = [
    "tiny",
    "base",
    "small",
    "medium",
    "large",
    "large-v2",
]
DUMMY_TOKEN = -100
MAX_LABEL_LENGTH = 448
MAX_AUDIO_LENGTH = 30  # second


# Training configuration
BATCH_SIZE = 16
LR = 3.75e-5
WARMUP_RATIO = 0.05  # between 5-15%

