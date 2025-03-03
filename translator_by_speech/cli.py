import argparse
import os
import sys
from typing import Dict, Any, Optional, List


# Import our custom modules
# Assuming these modules are in the same directory or properly installed
try:
    from translator_by_speech.record import AudioRecorder
    from translator_by_speech.speech_recognition import ASRModel
    from translator_by_speech.translator import (
        create_vi2en_translator,
        create_en2vi_translator,
        TranslationModel,
    )
    from translator_by_speech.pipeline import SpeechTranslationPipeline
except ImportError:
    print("Error: Required modules not found.")
    print(
        "Make sure translator_by_speech having record, speech_recognition and translator modules."
    )
    sys.exit(1)


class TranslationCLI:
    """Command Line Interface for audio recording, transcription and translation."""

    def __init__(self):
        """Initialize the CLI application with all required components."""
        # Initialize audio recorder
        self.recorder = AudioRecorder(output_directory="recordings")

        # Initialize models and pipeline with lazy loading (will be loaded when needed)
        self._asr_model = None
        self._vi2en_translator = None
        self._en2vi_translator = None
        self._vi2en_pipeline = None
        self._en2vi_pipeline = None

        # Default language settings
        self.source_lang = "vi"
        self.target_lang = "en"

        # Create output directories
        os.makedirs("recordings", exist_ok=True)
        os.makedirs("transcripts", exist_ok=True)

        print("Translation CLI initialized. Type 'help' for available commands.")

    @property
    def asr_model(self) -> ASRModel:
        """Lazy-loaded ASR model."""
        if self._asr_model is None:
            print("Loading ASR model (this may take a moment)...")
            self._asr_model = ASRModel(model_id="suzii/vi-whisper-large-v3-turbo-v1")
        return self._asr_model

    @property
    def vi2en_translator(self) -> TranslationModel:
        """Lazy-loaded Vietnamese to English translator."""
        if self._vi2en_translator is None:
            print("Loading Vietnamese to English translation model...")
            self._vi2en_translator = create_vi2en_translator()
        return self._vi2en_translator

    @property
    def en2vi_translator(self) -> TranslationModel:
        """Lazy-loaded English to Vietnamese translator."""
        if self._en2vi_translator is None:
            print("Loading English to Vietnamese translation model...")
            self._en2vi_translator = create_en2vi_translator()
        return self._en2vi_translator

    @property
    def vi2en_pipeline(self) -> SpeechTranslationPipeline:
        """Lazy-loaded Vietnamese to English speech translation pipeline."""
        if self._vi2en_pipeline is None:
            self._vi2en_pipeline = SpeechTranslationPipeline(
                asr_model=self.asr_model,
                translator_model=self.vi2en_translator,
                source_lang="vi",
                target_lang="en_XX",
            )
        return self._vi2en_pipeline

    @property
    def en2vi_pipeline(self) -> SpeechTranslationPipeline:
        """Lazy-loaded English to Vietnamese speech translation pipeline."""
        if self._en2vi_pipeline is None:
            self._en2vi_pipeline = SpeechTranslationPipeline(
                asr_model=self.asr_model,
                translator_model=self.en2vi_translator,
                source_lang="en",
                target_lang="vi_VN",
            )
        return self._en2vi_pipeline

    def record_audio(
        self, duration: Optional[int] = None, silence_detection: bool = True
    ) -> str:
        """
        Record audio from microphone.

        Args:
            duration: Fixed duration in seconds (if None, uses silence detection)
            silence_detection: Whether to use silence detection

        Returns:
            Path to the recorded audio file
        """
        try:
            if silence_detection and duration is None:
                print("Recording... (speak now, will stop after silence)")
                file_path = self.recorder.record_until_silence(
                    silence_threshold=500, silence_duration=1.5, max_duration=60
                )
            else:
                duration = duration or 5  # Default to 5 seconds if not specified
                print(f"Recording for {duration} seconds...")
                file_path = self.recorder.record(duration=duration)

            print(f"Recording saved to: {file_path}")
            return file_path

        except Exception as e:
            print(f"Error during recording: {str(e)}")
            return ""

    def transcribe_audio(self, file_path: str) -> Dict[str, Any]:
        """
        Transcribe audio file to text.

        Args:
            file_path: Path to audio file

        Returns:
            Dictionary with transcription results
        """
        try:
            print(f"Transcribing audio file: {file_path}")
            result = self.asr_model.transcribe_audio_file(
                file_path, language=self.source_lang
            )

            # Save transcript to file
            transcript_path = os.path.join(
                "transcripts", os.path.basename(file_path).replace(".wav", ".txt")
            )

            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(result["text"])

            print(f"Transcript saved to: {transcript_path}")
            return result

        except Exception as e:
            print(f"Error during transcription: {str(e)}")
            return {"text": "", "language": self.source_lang}

    def translate_text(self, text: str) -> str:
        """
        Translate text between languages.

        Args:
            text: Text to translate

        Returns:
            Translated text
        """
        try:
            if self.source_lang == "vi" and self.target_lang == "en":
                translation = self.vi2en_translator.translate(text)
            elif self.source_lang == "en" and self.target_lang == "vi":
                translation = self.en2vi_translator.translate(text)
            else:
                print(
                    f"Unsupported language pair: {self.source_lang} → {self.target_lang}"
                )
                return ""

            return translation

        except Exception as e:
            print(f"Error during translation: {str(e)}")
            return ""

    def process_audio_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process existing audio file through transcription and translation.

        Args:
            file_path: Path to audio file

        Returns:
            Dictionary with results
        """
        try:
            if not os.path.exists(file_path):
                print(f"Error: File not found: {file_path}")
                return {}

            # Choose the appropriate pipeline based on source/target languages
            if self.source_lang == "vi" and self.target_lang == "en":
                pipeline = self.vi2en_pipeline
            elif self.source_lang == "en" and self.target_lang == "vi":
                pipeline = self.en2vi_pipeline
            else:
                print(
                    f"Unsupported language pair: {self.source_lang} → {self.target_lang}"
                )
                return {}

            # Process through pipeline
            print(f"Processing audio file: {file_path}")
            result = pipeline.translate_speech_from_file(file_path)

            # Save results to files
            basename = os.path.basename(file_path).replace(".wav", "")

            # Save transcript
            transcript_path = os.path.join("transcripts", f"{basename}_transcript.txt")
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(result["source_text"])

            # Save translation
            translation_path = os.path.join(
                "transcripts", f"{basename}_translation.txt"
            )
            with open(translation_path, "w", encoding="utf-8") as f:
                f.write(result["translated_text"])

            print(f"Transcript saved to: {transcript_path}")
            print(f"Translation saved to: {translation_path}")

            return result

        except Exception as e:
            print(f"Error processing audio file: {str(e)}")
            return {}

    def record_and_process(self) -> Dict[str, Any]:
        """
        Record audio and process it through transcription and translation.

        Returns:
            Dictionary with results
        """
        file_path = self.record_audio(silence_detection=True)
        if not file_path:
            return {}

        return self.process_audio_file(file_path)

    def switch_languages(self) -> None:
        """Switch source and target languages."""
        self.source_lang, self.target_lang = self.target_lang, self.source_lang
        print(f"Languages switched: {self.source_lang} → {self.target_lang}")

    def set_languages(self, source: str, target: str) -> None:
        """
        Set source and target languages.

        Args:
            source: Source language code ('en' or 'vi')
            target: Target language code ('en' or 'vi')
        """
        if source not in ["en", "vi"] or target not in ["en", "vi"]:
            print("Error: Supported languages are 'en' (English) and 'vi' (Vietnamese)")
            return

        self.source_lang = source
        self.target_lang = target
        print(f"Languages set to: {self.source_lang} → {self.target_lang}")

    def print_help(self) -> None:
        """Print help information."""
        print("\n=== Audio Translation CLI Help ===")
        print("Available commands:")
        print("  record [duration]       - Record audio (optional duration in seconds)")
        print("  transcribe <file>       - Transcribe an audio file")
        print("  translate <text>        - Translate text")
        print("  process <file>          - Process audio file (transcribe + translate)")
        print("  speak                   - Record and process audio")
        print("  switch                  - Switch languages")
        print("  lang <src> <tgt>        - Set languages (en/vi)")
        print("  status                  - Show current status")
        print("  help                    - Show this help message")
        print("  exit                    - Exit the application")
        print("\nCurrent language setting:")
        print(f"  {self.source_lang} → {self.target_lang}")
        print("===============================\n")

    def print_status(self) -> None:
        """Print current status."""
        print("\n=== Current Status ===")
        print(f"Source language: {self.source_lang}")
        print(f"Target language: {self.target_lang}")
        print(f"ASR model loaded: {self._asr_model is not None}")
        print(f"VI→EN translator loaded: {self._vi2en_translator is not None}")
        print(f"EN→VI translator loaded: {self._en2vi_translator is not None}")
        print(f"Output directories:")
        print(f"  - Recordings: {os.path.abspath('recordings')}")
        print(f"  - Transcripts: {os.path.abspath('transcripts')}")
        print("====================\n")

    def handle_command(self, command: str, args: List[str]) -> bool:
        """
        Handle a user command.

        Args:
            command: Command name
            args: Command arguments

        Returns:
            False if the application should exit, True otherwise
        """
        if command == "exit":
            print("Exiting...")
            return False

        elif command == "help":
            self.print_help()

        elif command == "status":
            self.print_status()

        elif command == "record":
            duration = int(args[0]) if args else None
            self.record_audio(duration=duration)

        elif command == "transcribe":
            if not args:
                print("Error: Missing file path")
                return True

            result = self.transcribe_audio(args[0])
            print(f"\nTranscription: {result['text']}\n")

        elif command == "translate":
            if not args:
                print("Error: Missing text to translate")
                return True

            text = " ".join(args)
            translation = self.translate_text(text)
            print(f"\nTranslation: {translation}\n")

        elif command == "process":
            if not args:
                print("Error: Missing file path")
                return True

            result = self.process_audio_file(args[0])
            if result:
                print(f"\nTranscription: {result['source_text']}")
                print(f"Translation: {result['translated_text']}\n")

        elif command == "speak":
            result = self.record_and_process()
            if result:
                print(f"\nTranscription: {result['source_text']}")
                print(f"Translation: {result['translated_text']}\n")

        elif command == "switch":
            self.switch_languages()

        elif command == "lang":
            if len(args) < 2:
                print("Error: Missing language codes")
                return True

            self.set_languages(args[0], args[1])

        else:
            print(f"Unknown command: {command}")
            print("Type 'help' for available commands")

        return True

    def run(self) -> None:
        """Run the interactive CLI application."""
        try:
            running = True

            print("\nAudio Translation CLI")
            print("Type 'help' for available commands, 'exit' to quit")
            print(f"Current language setting: {self.source_lang} → {self.target_lang}")

            while running:
                try:
                    user_input = input("\n> ").strip()
                    if not user_input:
                        continue

                    parts = user_input.split()
                    command = parts[0].lower()
                    args = parts[1:]

                    running = self.handle_command(command, args)

                except KeyboardInterrupt:
                    print("\nOperation cancelled")

                except Exception as e:
                    print(f"Error: {str(e)}")

        except KeyboardInterrupt:
            print("\nExiting...")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Audio Translation CLI")
    parser.add_argument(
        "--source",
        "-s",
        choices=["en", "vi"],
        default="vi",
        help="Source language (default: vi)",
    )
    parser.add_argument(
        "--target",
        "-t",
        choices=["en", "vi"],
        default="en",
        help="Target language (default: en)",
    )
    parser.add_argument(
        "--record",
        "-r",
        type=int,
        metavar="DURATION",
        help="Record audio for DURATION seconds",
    )
    parser.add_argument("--process", "-p", metavar="FILE", help="Process audio file")
    parser.add_argument(
        "--interactive", "-i", action="store_true", help="Start interactive mode"
    )

    return parser.parse_args()
