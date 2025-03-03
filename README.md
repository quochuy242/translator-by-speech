# Speech-to-Text Translation System

A comprehensive audio translation system that combines speech recording, automatic speech recognition (ASR), and text translation capabilities to enable seamless translation between Vietnamese and English.

## Overview

This project integrates three core modules:

1. **Audio Recording Module**: Captures audio from microphone input with support for both fixed-duration recording and smart silence detection.
2. **ASR Module**: Transcribes speech to text using the Whisper model (`suzii/vi-whisper-large-v3-turbo-v1`).
3. **Translation Module**: Translates text between Vietnamese and English using VINAI's neural machine translation models.

The system provides both programmatic APIs for developers and a user-friendly command-line interface for end users.

## Features

- **Bidirectional Translation**: Support for Vietnamese ↔ English translation
- **Real-time Audio Processing**: Record and immediately transcribe/translate audio
- **Multiple Recording Modes**: Fixed duration or automatic silence detection
- **Batch Processing**: Process multiple audio files or text segments
- **Optimized Performance**: Efficient model loading and inference
- **User-friendly CLI**: Interactive command-line interface for all operations

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for faster processing)

### Dependencies

```bash
pip install -r requirements.txt
```

## Project Structure

```bash
├── translator_by_speech
│   ├── cli.py  # Command-line interface
│   ├── record.py   # Audio recording module
│   ├── speech_recognition.py   # ASR module and translation pipeline
│   ├── translator.py   # Text translation module
│   ├── pipeline.py   # Combine modules into the pipeline
├── recordings/               # Directory for stored audio recordings
└── transcripts/              # Directory for transcription and translation outputs
```

## Usage

### Command-Line Interface

The easiest way to use the system is through the provided CLI:

```bash
# Start interactive mode
python main.py

# Record 10 seconds of audio and translate
python main.py --record 10

# Process an existing audio file
python main.py --process recordings/sample.wav

# Change language direction (English to Vietnamese)
python main.py --source en --target vi
```

### Interactive Commands

Once in the interactive mode, you can use these commands:

- `record [duration]` - Record audio (with optional duration)
- `transcribe <file>` - Transcribe an audio file
- `translate <text>` - Translate text
- `process <file>` - Process audio file (transcribe + translate)
- `speak` - Record and process audio in one step
- `switch` - Switch source and target languages
- `lang <src> <tgt>` - Set source and target languages
- `status` - Show current status
- `help` - Show help information
- `exit` - Exit the application

### API Usage

You can also use the individual modules in your own Python code:

```python
# Recording audio
from translator_by_speech.record import AudioRecorder
recorder = AudioRecorder()
audio_path = recorder.record(duration=5)  # Record for 5 seconds

# ASR (Speech to Text)
from translator_by_speech.speech_recognition import ASRModel
asr = ASRModel()
transcription = asr.transcribe_audio_file(audio_path)

# Translation
from translator_by_speech.translator import create_vi2en_translator
translator = create_vi2en_translator()
translation = translator.translate(transcription["text"])

# Complete Pipeline
from translator_by_speech.pipeline import SpeechTranslationPipeline
pipeline = SpeechTranslationPipeline()
result = pipeline.translate_speech_from_file(audio_path)
print(f"Original: {result['source_text']}")
print(f"Translation: {result['translated_text']}")
```

## Models

This project uses the following AI models:

- **ASR**: `suzii/vi-whisper-large-v3-turbo-v1` (Vietnamese-optimized Whisper model)
- **Vietnamese to English**: `vinai/vinai-translate-vi2en-v2`
- **English to Vietnamese**: `vinai/vinai-translate-en2vi-v2`

## Performance Considerations

- The first run will download the models, which may take some time depending on your internet connection
- Using a GPU significantly improves processing speed
- ASR (speech recognition) is the most resource-intensive part of the pipeline

## Limitations

- Currently supports only Vietnamese and English
- Accuracy may vary depending on audio quality and background noise
- Large models require significant memory (especially for the ASR component)

## Future Improvements

- Add support for more languages
- Implement streaming ASR for real-time translation
- Create a graphical user interface
- Optimize models for faster inference on CPU
- Add support for batch processing of multiple files

## License

This project is released under the MIT License.
