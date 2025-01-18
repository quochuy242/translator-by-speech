import wave

import numpy as np
import pyaudio
import torch


def record_audio(duration: int, sample_rate: int, chunk_size: int) -> np.ndarray:
  """
  Record audio and return a numpy array

  Args:
      duration (int): recording duration in seconds
      sample_rate (int): sample rate in Hz
      chunk_size (int): chunk size in bytes
  """

  # Initialize PyAudio
  audio = pyaudio.PyAudio()

  # Start recording
  stream = audio.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=sample_rate,
    input=True,
    frames_per_buffer=chunk_size,
  )
  frames = []
  for _ in range(0, int(sample_rate / chunk_size * duration)):
    data = stream.read(chunk_size)
    frames.append(data)

  # Stop recording
  stream.stop_stream()
  stream.close()
  audio.terminate()

  # Convert frames to numpy array
  audio_data = np.frombuffer(b"".join(frames), dtype=np.int16)

  return audio_data


def save_audio(audio_data: np.ndarray, filename: str, sample_rate: int) -> None:
  with wave.open(filename, "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)  # 16-bit audio
    wf.setframerate(sample_rate)
    wf.writeframes(audio_data.tobytes())
