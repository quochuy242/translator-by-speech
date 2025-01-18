import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from typing import List, Optional, Union


def load_en2vi_model():
  tokenizer_en2vi = AutoTokenizer.from_pretrained(
    "vinai/vinai-translate-en2vi-v2", src_lang="en_XX"
  )
  model_en2vi = AutoModelForSeq2SeqLM.from_pretrained("vinai/vinai-translate-en2vi-v2")
  device_en2vi = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  )
  model_en2vi.to(device_en2vi)
  return tokenizer_en2vi, model_en2vi


def translate_en2vi(
  tokenizer_en2vi,
  model_en2vi,
  device_en2vi: torch.device = torch.device("cpu"),
  vi_texts: str = "Hello, how are you?",
) -> str:
  input_ids = tokenizer_en2vi(vi_texts, padding=True, return_tensors="pt").to(
    device_en2vi
  )
  output_ids = model_en2vi.generate(
    **input_ids,
    decoder_start_token_id=tokenizer_en2vi.lang_code_to_id["vi_VN"],
    num_return_sequences=1,
    num_beams=5,
    early_stopping=True,
  )
  en_texts = tokenizer_en2vi.batch_decode(output_ids, skip_special_tokens=True)
  return en_texts


def load_vi2en_model():
  tokenizer_vi2en = AutoTokenizer.from_pretrained(
    "vinai/vinai-translate-vi2en-v2", src_lang="vi_VN"
  )
  model_vi2en = AutoModelForSeq2SeqLM.from_pretrained("vinai/vinai-translate-vi2en-v2")
  device_vi2en = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  )
  model_vi2en.to(device_vi2en)
  return tokenizer_vi2en, model_vi2en


def translate_vi2en(
  tokenizer_vi2en,
  model_vi2en,
  device_vi2en: torch.device = torch.device("cpu"),
  vi_texts: str = "Xin chào, bạn khỏe không?",
) -> str:
  input_ids = tokenizer_vi2en(vi_texts, padding=True, return_tensors="pt").to(
    device_vi2en
  )
  output_ids = model_vi2en.generate(
    **input_ids,
    decoder_start_token_id=tokenizer_vi2en.lang_code_to_id["en_XX"],
    num_return_sequences=1,
    num_beams=5,
    early_stopping=True,
  )
  en_texts = tokenizer_vi2en.batch_decode(output_ids, skip_special_tokens=True)
  return en_texts
