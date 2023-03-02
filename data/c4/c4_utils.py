# Copyright 2022
import os

def get_c4_files(path):
    """name of the c4 files"""

    num_files_c4=1024
    file_list = [os.path.join(path, f"c4-train.{i:05}-of-01024.json") for i in range(num_files_c4)]
    
    return {
        "train": file_list[:-1],
        "validation": file_list[-2:-1],
        "test": file_list[-1:],
        }
