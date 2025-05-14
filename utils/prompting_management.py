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
                },
        ]
    },
]
