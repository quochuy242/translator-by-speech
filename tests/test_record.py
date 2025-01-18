from src.helpers import record_audio
import numpy as np
import pytest
import pyaudio

duration = 5
sample_rate = 16000
chunk_size = 1024


@pytest.fixture
def mock_pyaudio(mocker):
  """
  Mock the PyAudio library to avoid actual audio recording.
  """
  mock_audio = mocker.MagicMock(spec=pyaudio.PyAudio)
  mock_stream = mocker.MagicMock()
  mock_audio.open.return_value = mock_stream

  # Simulate generated audio data (e.g., silence)
  total_chunks = int(sample_rate / chunk_size * duration)
  silent_audio = b"\x00\x00" * chunk_size  # Silent audio bytes

  mock_stream.read.side_effect = [silent_audio] * total_chunks
  return mock_audio, mock_stream


def test_record_audio(mocker, mock_pyaudio):
  """
  Test the record_audio function.
  """
  # Arrange
  mock_audio, mock_stream = mock_pyaudio
  mocker.patch("pyaudio.PyAudio", return_value=mock_audio)

  # Simulate silent audio
  silent_audio = b"\x00\x00" * chunk_size
  total_chunks = int(sample_rate / chunk_size * duration)
  mock_stream.read.side_effect = [silent_audio] * total_chunks

  # Act
  audio_data = record_audio(duration, sample_rate, chunk_size)

  # Assert
  expected_samples = total_chunks * chunk_size
  assert isinstance(audio_data, np.ndarray), "Audio data should be a numpy array"
  assert len(audio_data) == expected_samples, (
    f"Audio data length does not match expected duration ({len(audio_data)} != {expected_samples})"
  )
  assert audio_data.dtype == np.int16, "Audio data should be 16-bit PCM format"

  # Ensure PyAudio methods were called as expected
  mock_audio.open.assert_called_once_with(
    format=pyaudio.paInt16,
    channels=1,
    rate=sample_rate,
    input=True,
    frames_per_buffer=chunk_size,
  )
  mock_stream.read.assert_called()

  print("Test passed successfully!")
