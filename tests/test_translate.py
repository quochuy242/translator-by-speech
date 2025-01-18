from unittest.mock import MagicMock

import pytest
import torch

from src.translator import (
  load_en2vi_model,
  load_vi2en_model,
  translate_en2vi,
  translate_vi2en,
)


@pytest.fixture
def mock_models():
  # Mock the tokenizer and model for en2vi
  tokenizer_en2vi = MagicMock()
  model_en2vi = MagicMock()
  device_en2vi = torch.device("cpu")

  # Mock the tokenizer and model for vi2en
  tokenizer_vi2en = MagicMock()
  model_vi2en = MagicMock()
  device_vi2en = torch.device("cpu")

  return {
    "tokenizer_en2vi": tokenizer_en2vi,
    "model_en2vi": model_en2vi,
    "device_en2vi": device_en2vi,
    "tokenizer_vi2en": tokenizer_vi2en,
    "model_vi2en": model_vi2en,
    "device_vi2en": device_vi2en,
  }


def test_load_en2vi_model(mock_models):
  # Mock the AutoTokenizer and AutoModelForSeq2SeqLM methods
  tokenizer_en2vi, model_en2vi = load_en2vi_model()

  # Check if the tokenizer and model are returned
  assert tokenizer_en2vi is not None
  assert model_en2vi is not None

  # Check if the model is moved to the correct device
  device_en2vi = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  )
  assert model_en2vi.device == device_en2vi


def test_load_vi2en_model(mock_models):
  # Mock the AutoTokenizer and AutoModelForSeq2SeqLM methods
  tokenizer_vi2en, model_vi2en = load_vi2en_model()

  # Check if the tokenizer and model are returned
  assert tokenizer_vi2en is not None
  assert model_vi2en is not None

  # Check if the model is moved to the correct device
  device_vi2en = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  )
  assert model_vi2en.device == device_vi2en


def test_translate_en2vi(mock_models):
  tokenizer_en2vi = mock_models["tokenizer_en2vi"]
  model_en2vi = mock_models["model_en2vi"]
  device_en2vi = mock_models["device_en2vi"]

  # Mock the tokenizer and model behaviors
  tokenizer_en2vi.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
  model_en2vi.generate.return_value = torch.tensor([[4, 5, 6]])
  tokenizer_en2vi.batch_decode.return_value = ["translated text"]

  result = translate_en2vi(tokenizer_en2vi, model_en2vi, device_en2vi, "Hello")

  assert result == ["translated text"]
  tokenizer_en2vi.assert_called_once()
  model_en2vi.generate.assert_called_once()


def test_translate_vi2en(mock_models):
  tokenizer_vi2en = mock_models["tokenizer_vi2en"]
  model_vi2en = mock_models["model_vi2en"]
  device_vi2en = mock_models["device_vi2en"]

  # Mock the tokenizer and model behaviors
  tokenizer_vi2en.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
  model_vi2en.generate.return_value = torch.tensor([[4, 5, 6]])
  tokenizer_vi2en.batch_decode.return_value = ["translated text"]

  result = translate_vi2en(tokenizer_vi2en, model_vi2en, device_vi2en, "Xin chào")

  assert result == ["translated text"]
  tokenizer_vi2en.assert_called_once()
  model_vi2en.generate.assert_called_once()
