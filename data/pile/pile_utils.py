# Copyright 2022
import os

def get_pile_files(path):
    """name of the pile files"""

    file_list = [os.path.join(path, f"{i:02}.jsonl") for i in range(20)]
    return {
        "train": file_list[:-1],
        "validation": file_list[-1:],
        "test": file_list[-1:],
        }

def get_minipile_files(path, num_files):
    """name of the minipile files"""

    file_list = [f"shuffled_00_x0{i:02}.jsonl" for i in range(num_files)]
    return {
        "train": file_list[:-1],
        "validation": file_list[-1:],
        "test": file_list[-1:],
        }
