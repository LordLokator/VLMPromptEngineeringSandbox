# main.py
import sys
import threading
import time
from loguru import logger
from vlm_serve import VLM
from web_engine import start_server_threaded

from streaming import get_stream_source
from video_hanlding.clip_handling import ClipRecorder

logger.remove()
logger.add(sys.stderr, level="DEBUG")


def main():
    logger.info("Main app initialization...")

    # Start server in separate thread
    start_server_threaded(prompt)

    # Instantiate VLM
    # vlm = VLM()
    stream = get_stream_source()
    recorder = ClipRecorder(buffer_seconds=2, fps=10)
    recorder.start(stream, every_seconds=10)


    # Integration logic
    # try:
    #     while True:
    #         ...

    # except KeyboardInterrupt:
    #     logger.info("Exited with KeyboardInterrupt.")

    recorder.stop()


if __name__ == "__main__":
    main()
