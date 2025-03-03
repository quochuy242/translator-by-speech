import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Optional, Union


class TranslationModel:
    """A class to handle translation models between English and Vietnamese."""

    def __init__(
        self,
        model_name: str,
        src_lang: str,
        tgt_lang: str,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the translation model.

        Args:
            model_name: The name of the pretrained model
            src_lang: Source language code
            tgt_lang: Target language code
            device: The device to run the model on (defaults to CUDA if available)
        """
        self.model_name = model_name
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=src_lang)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)

    def translate(
        self,
        text: Union[str, List[str]],
        num_beams: int = 5,
        early_stopping: bool = True,
    ) -> Union[str, List[str]]:
        """
        Translate text from source to target language.

        Args:
            text: Input text or list of texts to translate
            num_beams: Number of beams for beam search
            early_stopping: Whether to stop beam search when first complete candidate is found

        Returns:
            Translated text or list of translated texts
        """
        is_single_text = isinstance(text, str)
        texts = [text] if is_single_text else text

        # Tokenize input text
        inputs = self.tokenizer(texts, padding=True, return_tensors="pt").to(
            self.device
        )

        # Generate translation
        outputs = self.model.generate(
            **inputs,
            decoder_start_token_id=self.tokenizer.lang_code_to_id[self.tgt_lang],
            num_return_sequences=1,
            num_beams=num_beams,
            early_stopping=early_stopping,
        )

        # Decode output tokens
        translated_texts = self.tokenizer.batch_decode(
            outputs, skip_special_tokens=True
        )

        return translated_texts[0] if is_single_text else translated_texts


# Factory functions for convenience
def create_en2vi_translator() -> TranslationModel:
    """Create an English to Vietnamese translator."""
    return TranslationModel(
        model_name="vinai/vinai-translate-en2vi-v2", src_lang="en_XX", tgt_lang="vi_VN"
    )


def create_vi2en_translator() -> TranslationModel:
    """Create a Vietnamese to English translator."""
    return TranslationModel(
        model_name="vinai/vinai-translate-vi2en-v2", src_lang="vi_VN", tgt_lang="en_XX"
    )
