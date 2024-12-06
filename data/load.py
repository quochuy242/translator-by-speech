import datasets as hugDS
from utils import const


def load_my_dataset(mode: int, **kwargs):
    ds = hugDS.load_dataset(
        **kwargs, trust_remote_code=True, streaming=const.STREAMING
    ).cast_column("audio", hugDS.Audio(sampling_rate=const.SAMPLE_RATE))

    match mode:
        case 0:
            return ds
        case 1:
            return ds.select_columns(["audio", "transcription"])
        case 2:
            return ds.select_columns(["audio", "sentence"]).rename_column(
                "sentence", "transcription"
            )
        case _:
            raise ValueError("mode must be 0, 1 or 2")
