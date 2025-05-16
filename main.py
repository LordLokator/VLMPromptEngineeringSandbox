# main.py
import sys
import threading
from loguru import logger
from vlm_serve import VLM
from web_engine import start_server

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
    vlm = VLM()

    # Integration logic
    try:
        while True:
            ...
    except KeyboardInterrupt:
        logger.info("Exited with KeyboardInterrupt.")

if __name__ == "__main__":
    main()
