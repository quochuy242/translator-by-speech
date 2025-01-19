from unittest.mock import MagicMock, patch

import pytest

from translator_by_speech.record import record_audio


# Mock logger to prevent output during testing
class MockLogger:
  def warning(self, msg):
    pass

  def info(self, msg):
    pass

  def error(self, msg):
    pass


@pytest.fixture
def mock_logger():
  return MockLogger()


@pytest.fixture
def mock_tempfile():
  with patch("tempfile.NamedTemporaryFile") as mock_temp:
    mock_temp.return_value.name = "/assets/tmp/test_audio.wav"
    yield mock_temp


@pytest.fixture
def mock_input_stream():
  with patch("sounddevice.InputStream") as mock_stream:
    yield mock_stream


@pytest.fixture
def mock_soundfile():
  with patch("soundfile.SoundFile") as mock_sf:
    yield mock_sf


@pytest.fixture
def mock_queue():
  with patch("queue.Queue") as mock_q:
    yield mock_q


@pytest.mark.parametrize(
  "filename, expected_filename",
  [
    ("/assets/tmp/output.wav", "/assets/tmp/output.wav"),  # Test with explicit filename
    (
      None,
      "/assets/tmp/test_audio.wav",
    ),  # Test with None, temporary file should be created
  ],
)
def test_record_audio(
  mock_logger,
  mock_tempfile,
  mock_input_stream,
  mock_soundfile,
  mock_queue,
  filename,
  expected_filename,
):
  # Setup mocks
  mock_sf = mock_soundfile.return_value.__enter__.return_value
  mock_sf.write = MagicMock()

  mock_stream = mock_input_stream.return_value.__enter__.return_value
  mock_stream.read = MagicMock(return_value=(b"fake_audio_data", 1024))

  # Call the function
  with patch("sys.stderr.write"):  # Suppress stderr output
    result = record_audio(filename=filename, sample_rate=16000)

  # Assertions
  mock_tempfile.assert_called_once()  # Ensure tempfile was called when filename is None
  mock_sf.write.assert_called()  # Ensure that the write method of SoundFile was called
  assert result == expected_filename  # Ensure the correct filename is returned
  mock_input_stream.assert_called_once()  # Ensure InputStream was initialized


def test_record_audio_keyboard_interrupt(
  mock_logger, mock_input_stream, mock_soundfile, mock_queue
):
  # Test KeyboardInterrupt handling
  mock_sf = mock_soundfile.return_value.__enter__.return_value
  mock_sf.write = MagicMock()

  mock_stream = mock_input_stream.return_value.__enter__.return_value
  mock_stream.read = MagicMock(return_value=(b"fake_audio_data", 1024))

  with (
    patch("sys.stderr.write"),
    patch("builtins.input", side_effect=KeyboardInterrupt),
  ):
    _ = record_audio(filename="/assets/tmp/output.wav", sample_rate=16000)

  # Ensure the function handles KeyboardInterrupt gracefully
  mock_logger.info.assert_called_with(
    "Recording stopped. File saved in /assets/tmp/output.wav."
  )
