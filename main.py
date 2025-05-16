# main.py
import sys
import threading
import time
from loguru import logger
from vlm_serve import VLM
from web_engine import start_server

from streaming import get_stream_source
from video_hanlding.clip_handling import ClipRecorder

logger.remove()
logger.add(sys.stderr, level="DEBUG")


def run_web_server():
    logger.info("Starting web server thread...")
    start_server()


def main():
    logger.info("Main app initialization...")

    # Start server in separate thread
    server_thread = threading.Thread(target=run_web_server, daemon=True)
    server_thread.start()

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
