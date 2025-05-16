# main.py
import sys
import time
import timeit
from loguru import logger
from prompting.prompting import Prompt
from vlm_serve import VLM
from web_engine import start_server_threaded, broadcast

from video_hanlding.clip_handling import ClipRecorder

logger.remove()
logger.add(sys.stderr, level="DEBUG")

FPS = 30
MAX_FRAMES = 120


def main():
    logger.info("Main app initialization...")

    # Create shared Prompt object
    prompt = Prompt()

    # Start server in separate thread
    start_server_threaded(prompt)

    # Create Video stream
    recorder = ClipRecorder(
        buffer_seconds=2,
        fps=FPS
    )

    # Instantiate VLM
    vlm = VLM(prompt)

    def on_clip_flush(clip_path: str):
        logger.info(f"Clip flushed: {clip_path}")
        start = timeit.default_timer()

        try:
            output = vlm.forward(video_path=clip_path, fps=FPS, max_frames=MAX_FRAMES)
            delay = timeit.default_timer() - start

            _message = f"VLM Response (after {delay:.2f}s): {output}"
            logger.info(_message)

            # Send to frontend
            msg = {
                'output': output,
                'delay': delay,
                'status': 'OK'
            }
            broadcast(msg)

        except Exception as e:
            err = f"Failed to process: {e}"
            logger.error(err)
            msg = {
                'status': 'ERROR',
                'msg': err
            }
            broadcast(msg)

    recorder.start(every_seconds=1, flush_callback=on_clip_flush)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Exited with KeyboardInterrupt.")
        recorder.stop()


if __name__ == "__main__":
    main()
