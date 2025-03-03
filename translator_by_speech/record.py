import pyaudio
import wave
import os
import time
from datetime import datetime


class AudioRecorder:
    def __init__(self, output_directory="recordings"):
        """Initialize the audio recorder with configuration parameters."""
        # Audio recording parameters
        self.format = pyaudio.paInt16  # 16-bit resolution
        self.channels = 1  # Mono audio
        self.rate = 44100  # 44.1kHz sampling rate
        self.chunk = 1024  # Number of frames per buffer

        # Output configuration
        self.output_directory = output_directory
        os.makedirs(output_directory, exist_ok=True)

        # PyAudio instance
        self.audio = pyaudio.PyAudio()

    def record(self, duration=5, filename=None):
        """
        Record audio for a specified duration.

        Args:
            duration (int): Recording duration in seconds
            filename (str, optional): Output filename. If None, generates a timestamped filename.

        Returns:
            str: Path to the saved audio file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{timestamp}.wav"

        filepath = os.path.join(self.output_directory, filename)

        # Open audio stream
        stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk,
        )

        print(f"Recording started for {duration} seconds...")
        frames = []

        # Record audio in chunks
        for i in range(0, int(self.rate / self.chunk * duration)):
            data = stream.read(self.chunk)
            frames.append(data)

        print("Recording finished.")

        # Stop and close the stream
        stream.stop_stream()
        stream.close()

        # Save the recorded audio to a WAV file
        self._save_audio(filepath, frames)

        return filepath

    def record_until_silence(
        self, silence_threshold=1000, silence_duration=2, max_duration=60, filename=None
    ):
        """
        Record audio until silence is detected or max duration is reached.

        Args:
            silence_threshold (int): Amplitude threshold to consider as silence
            silence_duration (int): Consecutive seconds of silence to stop recording
            max_duration (int): Maximum recording duration in seconds
            filename (str, optional): Output filename. If None, generates a timestamped filename.

        Returns:
            str: Path to the saved audio file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{timestamp}.wav"

        filepath = os.path.join(self.output_directory, filename)

        # Open audio stream
        stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk,
        )

        print("Recording started (speak into the microphone)...")
        print("Recording will stop after silence or maximum duration...")

        frames = []
        silence_count = 0
        start_time = time.time()

        # Record audio until silence is detected or max duration is reached
        while True:
            data = stream.read(self.chunk)
            frames.append(data)

            # Check for silence (simplified approach)
            amplitude = max(
                abs(int.from_bytes(data[i : i + 2], byteorder="little", signed=True))
                for i in range(0, len(data), 2)
            )

            if amplitude < silence_threshold:
                silence_count += 1
            else:
                silence_count = 0

            # Check if silence duration threshold is reached
            if silence_count >= int(silence_duration * self.rate / self.chunk):
                print("Silence detected, stopping recording.")
                break

            # Check if maximum duration is reached
            if time.time() - start_time > max_duration:
                print("Maximum duration reached, stopping recording.")
                break

        # Stop and close the stream
        stream.stop_stream()
        stream.close()

        # Save the recorded audio to a WAV file
        self._save_audio(filepath, frames)

        return filepath

    def _save_audio(self, filepath, frames):
        """Save recorded frames to a WAV file."""
        wf = wave.open(filepath, "wb")
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.audio.get_sample_size(self.format))
        wf.setframerate(self.rate)
        wf.writeframes(b"".join(frames))
        wf.close()
        print(f"Audio saved to: {filepath}")

    def __del__(self):
        """Clean up resources when the object is deleted."""
        self.audio.terminate()
