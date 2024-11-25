import os
from datetime import datetime as dt

from config.config import (OUT_FOLDER, OUT_SUBFOLDER)


def time_spent_decorator(func):
    """
    The function is a decorator which prints out when a function func is started
    and prints out the total time used to run the function once it is done.
    <br><br>

    :param func: A function to be decorated
    :returns: A wrapped function
    """

    def wrapper(*args, **kwargs):
        title = f"==> Starting function {func.__name__}"
        print(title)
        st = dt.now()
        func(*args, **kwargs)
        time_used_text = f"-- Total time used for function {func.__name__}: {(dt.now() - st).total_seconds()} seconds"
        print(time_used_text)
        print('-' * len(time_used_text))

    return wrapper


def print_title_in_output(title: str, char: str):
    """
    Prints the title section in the output console.
    Example:

    char = '*'
    title = "Example Title"
    output:
    ***************************
    ****** Example Title ******
    ***************************

    :param title: the title to be shown
    :param char: a character used as a frame
    """
    prefix = char * 6 + ' '
    suffix = ' ' + char * 6
    h_line = char * (len(prefix) + len(suffix) + len(title))

    print(h_line)
    print(prefix + title + suffix)
    print(h_line)


def init_out_dir():
    """
    Generate a base folder for storing output files
    """

    folder_path = os.path.join(OUT_FOLDER)
    os.makedirs(folder_path, exist_ok=True)


def init_subfolder(week: int):
    """
    Generate a subfolder under the base folder for storing output files in each iteration

    :param week: a week which an iteration is working
    """

    folder_path = os.path.join(OUT_FOLDER, OUT_SUBFOLDER.format(week))
    os.makedirs(folder_path, exist_ok=True)


def generate_out_filename(week: int, filename: str) -> str:
    """
    Generate an output filename under the base folder and subfolder.

    :param week: a week which an iteration is working
    :param filename: a file name
    """

    # Make sure the folder exists
    init_subfolder(week)
    folder_path = str(os.path.join(OUT_FOLDER, OUT_SUBFOLDER.format(week), filename))

    # Return the full file path
    return folder_path
