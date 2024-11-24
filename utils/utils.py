import os
from config.config import (OUT_FOLDER, OUT_SUBFOLDER)


def init_out_dir():
    folder_path = os.path.join(OUT_FOLDER)
    os.makedirs(folder_path, exist_ok=True)


def init_subfolder(week: int):
    folder_path = os.path.join(OUT_FOLDER, OUT_SUBFOLDER.format(week))
    os.makedirs(folder_path, exist_ok=True)


def generate_out_filename(week: int, filename: str) -> str:
    # Make sure the folder exists
    init_subfolder(week)
    folder_path = str(os.path.join(OUT_FOLDER, OUT_SUBFOLDER.format(week), filename))

    # Return the full file path
    return folder_path
