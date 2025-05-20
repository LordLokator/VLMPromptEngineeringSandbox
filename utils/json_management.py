import json
import os
from loguru import logger


def append_dict_to_json(file_path: str, new_data: dict):
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            json.dump([new_data], f, indent=4)
            logger.info(f"Created new JSON file and added data: {new_data}")
        return

    with open(file_path, 'r+') as f:
        try:
            data = json.load(f)
            if not isinstance(data, list):
                logger.info("Existing JSON is not a list. Wrapping existing data in a list.")
                data = [data]
        except json.JSONDecodeError:
            logger.info("JSON file is empty or invalid. Starting fresh.")
            data = []

        data.append(new_data)
        f.seek(0)
        json.dump(data, f, indent=4)
        f.truncate()
        logger.info(f"Appended data to JSON: {new_data}")
