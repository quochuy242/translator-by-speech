from typing import Dict, Optional

import numpy as np

from translator_by_speech.speech_recognition import ASRModel
from translator_by_speech.translator import TranslationModel


class SpeechTranslationPipeline:
    """Pipeline that combines ASR and translation for end-to-end speech translation."""

    def __init__(
        self,
        asr_model: Optional[ASRModel] = None,
        translator_model: Optional[TranslationModel] = None,
        source_lang: str = "vi",
        target_lang: str = "en_XX",
    ):
        """
        Initialize the speech translation pipeline.

        Args:
            asr_model: ASR model instance (creates a new one if None)
            translator_model: Translation model instance (creates a new one if None)
            source_lang: Source language code for ASR
            target_lang: Target language code for translation
        """
        # Import here to avoid circular imports
        from translator_by_speech.translator import (
            create_en2vi_translator,
            create_vi2en_translator,
        )

        self.source_lang = source_lang
        self.target_lang = target_lang

        # Initialize ASR model if not provided
        self.asr_model = asr_model or ASRModel()

        # Initialize translation model if not provided
        if translator_model is None:
            if source_lang == "vi" and target_lang == "en_XX":
                translator_model = create_vi2en_translator()
            elif source_lang == "en" and target_lang == "vi_VN":
                translator_model = create_en2vi_translator()
            else:
                raise ValueError(
                    f"Unsupported language pair: {source_lang} -> {target_lang}"
                )

        self.translator_model = translator_model

    def translate_speech_from_file(self, file_path: str) -> Dict[str, str]:
        """
        Process audio file through ASR and translation.

        Args:
            file_path: Path to the audio file

        Returns:
            Dictionary with original transcription and translation
        """
        # Transcribe audio to text
        asr_result = self.asr_model.transcribe_audio_file(
            file_path, language=self.source_lang
        )

        # Translate the transcribed text
        transcription = asr_result["text"]
        translation = self.translator_model.translate(transcription)

        return {
            "source_text": transcription,
            "source_lang": self.source_lang,
            "translated_text": translation,
            "target_lang": self.target_lang,
        }

    def translate_speech(
        self, audio_array: np.ndarray, sampling_rate: int
    ) -> Dict[str, str]:
        """
        Process audio array through ASR and translation.

        Args:
            audio_array: Numpy array of audio samples
            sampling_rate: Sampling rate of the audio

        Returns:
            Dictionary with original transcription and translation
        """
        # Transcribe audio to text
        asr_result = self.asr_model.transcribe_audio(
            audio_array, sampling_rate, language=self.source_lang
        )

        # Translate the transcribed text
        transcription = asr_result["text"]
        translation = self.translator_model.translate(transcription)

        return {
            "source_text": transcription,
            "source_lang": self.source_lang,
            "translated_text": translation,
            "target_lang": self.target_lang,
        }
