import os
import queue
import sys
import tempfile

import numpy as np
import sounddevice as sd
import soundfile as sf

from translator_by_speech.default import logger, SAMPLE_RATE


def record_audio(filename: str, sample_rate: int = SAMPLE_RATE) -> str:
  """
  Record audio with arbitrary duration

  Args:
      filename (str): output filename, if None, a temporary file will be created
      sample_rate (int): sample rate in Hz, defaults to 16000
  """

  # Initialize the queue
  audio_queue = queue.Queue()

  def callback(indata, frames, time, status):
    if status:
      print(status, file=sys.stderr)
    audio_queue.put(indata.copy())

  if filename is None:
    # Create a temporary file
    os.makedirs("/assets/tmp", exist_ok=True)
    with tempfile.NamedTemporaryFile(
      delete=False, dir="/assets/tmp", suffix=".wav"
    ) as temp_file:
      filename = temp_file.name

  try:
    # Open a SoundFile for writing and start an InputStream for recording
    with sf.SoundFile(
      filename, mode="x", samplerate=sample_rate, channels=1
    ) as audio_file:
      with sd.InputStream(samplerate=sample_rate, channels=1, callback=callback):
        logger.warning("Recording... (Press Ctrl+C to stop)")
        while True:
          audio_file.write(audio_queue.get())
  except KeyboardInterrupt:
    logger.info(f"Recording stopped. File saved in {filename}.")
  except Exception as e:
    logger.error(f"type: {type(e)} | value: {e}")
    raise e  # Reraise the exception for proper error handling
  finally:
    return filename
