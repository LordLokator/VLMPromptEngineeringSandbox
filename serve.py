import sys
from transformers import AutoModelForCausalLM, AutoProcessor
from VideoLLaMA3.inference.transformers_api.modeling_videollama3 import Videollama3Qwen2ForCausalLM
from VideoLLaMA3.inference.transformers_api.processing_videollama3 import Videollama3Qwen2Processor

import time
import ffmpeg
import os
import torch

from utils.file_management import makedir, full_path
from utils.vlm_wrapper import VLM, get_conv


from loguru import logger

MODEL_PATH = "DAMO-NLP-SG/VideoLLaMA3-7B"
MODEL_NAME = MODEL_PATH.split('/')[-1]  # HuggingFace path convention.
MAX_NEW_TOKENS = 1024
TEMP = 0.01
DEVICE = "cuda:0"
MAX_FRAMES: int = 60
FPS = 60

SAVE_TO_FOLDER = './demo'


def main():

    if not os.path.exists(f"{SAVE_TO_FOLDER}"):
        os.mkdir(f"{SAVE_TO_FOLDER}")
        logger.debug(f'Created folder [{SAVE_TO_FOLDER}].')

    vlm_model = VLM()
    # model: Videollama3Qwen2ForCausalLM = AutoModelForCausalLM.from_pretrained(
    #     MODEL_PATH,
    #     trust_remote_code=True,
    #     device_map={"": DEVICE},
    #     torch_dtype=torch.bfloat16,
    #     attn_implementation="flash_attention_2"
    # )

    # processor: Videollama3Qwen2Processor = AutoProcessor.from_pretrained(
    #     MODEL_PATH,
    #     trust_remote_code=True
    # )

    logger.info("Instantiated models.")

    def step(prompt: str, video_path: str):
        video_path = full_path(video_path)
        if not os.path.isfile(video_path):
            logger.error(f"Path provided is not a file. Path was [{video_path}]")
            return

        _ts = time.localtime()

        month = f"{_ts.tm_mon:02d}"
        day = f"{_ts.tm_mday:02d}"
        hour = f"{_ts.tm_hour:02d}"
        minute = f"{_ts.tm_min:02d}"

        day_folder = f"{month}_{day}"
        makedir(f"{SAVE_TO_FOLDER}/{day_folder}")

        conversation = get_conv(video_path, FPS, MAX_FRAMES, prompt)
        model_output = vlm_model.forward(conversation)
        raw_output: str = model_output[0].strip()

        report = f"At {hour}:{minute}, {MODEL_NAME} says: {raw_output}"
        logger.info(report)

        # region  output handling

        report_folder_path = f"{SAVE_TO_FOLDER}/{day_folder}/.../"
        makedir(report_folder_path)

        # endregion

        return raw_output

    step(prompt="What is on the video?", video_path='~/Videos/1.mp4')
    step(prompt="What is on the video?", video_path='~/Videos/1.mp4')
    step(prompt="What is on the video?", video_path='~/bad.path')


if __name__ == '__main__':
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add("./logs/logfile.log", mode="w", level="DEBUG")

    try:
        main()
    except Exception as e:
        logger.error(f"Error on main: [{e}]")
        raise e
