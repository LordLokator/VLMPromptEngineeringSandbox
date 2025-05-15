from transformers import AutoModelForCausalLM, AutoProcessor
from VideoLLaMA3.inference.transformers_api.modeling_videollama3 import Videollama3Qwen2ForCausalLM
from VideoLLaMA3.inference.transformers_api.processing_videollama3 import Videollama3Qwen2Processor

import torch
from loguru import logger

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
        logger.info("Instantiated models.")

    def forward(self, conversation):

        try:

            # VLM processor inputs. Move to VLM
            processed_inputs = self.processor(
                conversation=conversation,
                add_system_prompt=True,
                add_generation_prompt=True,
                return_tensors="pt"
            )
            inputs = {
                k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in processed_inputs.items()
            }
        except KeyboardInterrupt as e:
            logger.info(f"Keyboard Interrupt, shutting down...")
            raise e
        except Exception as e:
            logger.error(f"Error occured on VLM input preprocess: {e}")

        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        # VLM generate
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMP,
        )
        raw_output: str = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True
        )

        return raw_output


sys_prompt = "You are a helpful assistant."


def get_conv(video_path, fps, max_frames, prompt): return [
    {
        "role": "system",
        "content": sys_prompt
    },
    {
        "role": "user",
        "content": [
                {
                    "type": "video",
                    "video": {
                        "video_path": video_path,
                        "fps": fps,
                        "max_frames": max_frames
                    }
                },
            {
                    "type": "text",
                    "text": prompt
            }
        ]
    },
]
