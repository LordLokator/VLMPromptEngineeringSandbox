from transformers import AutoModelForCausalLM, AutoProcessor
from VideoLLaMA3.inference.transformers_api.modeling_videollama3 import Videollama3Qwen2ForCausalLM
from VideoLLaMA3.inference.transformers_api.processing_videollama3 import Videollama3Qwen2Processor

import torch


MAX_NEW_TOKENS = 1024
TEMP = 0.01
SAVE_TO_FOLDER = './demo'
MODEL_PATH = "DAMO-NLP-SG/VideoLLaMA3-7B"
DEVICE = "cuda:0"


class VLM:
    def __init__(self):
        self.model: Videollama3Qwen2ForCausalLM = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            device_map={"": DEVICE},
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        )

        self.processor: Videollama3Qwen2Processor = AutoProcessor.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True
        )
