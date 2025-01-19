from pathlib import Path
from loguru import logger

SAMPLE_RATE = 16000
TEST_OUTPUT_DIR = Path("../assets/tests/")

logger.add(
  "../assets/logs/record.log", level="DEBUG", format="{time} | {level} | {message}"
)
