import torch
import numpy as np
from typing import Optional, Union, Dict, Any, List, Tuple
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import soundfile as sf
from translator_by_speech.translator import (
    create_vi2en_translator,
    create_en2vi_translator,
    TranslationModel,
)


class ASRModel:
    """Speech recognition model that transcribes audio to text."""

    def __init__(
        self,
        model_id: str = "suzii/vi-whisper-large-v3-turbo-v1",
        device: Optional[torch.device] = None,
        torch_dtype: torch.dtype = torch.float16,
    ):
        """
        Initialize the ASR model.

        Args:
            model_id: The Hugging Face model ID for the ASR model
            device: The device to run the model on (defaults to CUDA if available)
            torch_dtype: Datatype to use for model parameters
        """
        self.model_id = model_id
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        # Use float32 if on CPU to avoid issues
        if self.device == torch.device("cpu"):
            torch_dtype = torch.float32

        self.torch_dtype = torch_dtype

        # Load processor and model
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        self.model.to(self.device)

    def transcribe_audio_file(
        self,
        file_path: str,
        language: Optional[str] = "vi",
        return_timestamps: bool = False,
    ) -> Dict[str, Any]:
        """
        Transcribe audio from a file.

        Args:
            file_path: Path to the audio file
            language: Language code (default is Vietnamese)
            return_timestamps: Whether to return word timestamps

        Returns:
            Dictionary containing transcription and metadata
        """
        # Load audio file
        audio_array, sampling_rate = sf.read(file_path)

        # Convert to mono if stereo
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)

        return self.transcribe_audio(
            audio_array, sampling_rate, language, return_timestamps
        )

    def transcribe_audio(
        self,
        audio_array: np.ndarray,
        sampling_rate: int,
        language: Optional[str] = "vi",
        return_timestamps: bool = False,
    ) -> Dict[str, Any]:
        """
        Transcribe audio from a numpy array.

        Args:
            audio_array: Numpy array of audio samples
            sampling_rate: Sampling rate of the audio
            language: Language code (default is Vietnamese)
            return_timestamps: Whether to return word timestamps

        Returns:
            Dictionary containing transcription and metadata
        """
        # Ensure proper dtype
        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)

        # Normalize if not already normalized
        if np.abs(audio_array).max() > 1.0:
            audio_array = audio_array / np.abs(audio_array).max()

        # Process audio with the model's processor
        inputs = self.processor(
            audio=audio_array, sampling_rate=sampling_rate, return_tensors="pt"
        ).to(self.device)

        # Generate transcription
        with torch.no_grad():
            generation_config = {
                "max_new_tokens": 256,
                "return_timestamps": return_timestamps,
            }

            # Add language forcing if provided
            if language:
                generation_config["language"] = language

            outputs = self.model.generate(**inputs, **generation_config)

        # Decode the output
        transcription = self.processor.batch_decode(outputs, skip_special_tokens=True)[
            0
        ]

        # Process timestamps if requested
        timestamps = None
        if return_timestamps and hasattr(self.processor, "decode_with_timestamps"):
            timestamps = self.processor.decode_with_timestamps(outputs[0].tolist())

        return {"text": transcription, "language": language, "timestamps": timestamps}
