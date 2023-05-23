"""
Filename: feature.py
直接读取文件化为特征
"""

import numpy as np


def pad_or_cut(value: np.ndarray, target_length: int):
    """
    填充或截断一维numpy到固定的长度
    """
    if len(value) < target_length:
        data_row = np.pad(value, [(0, target_length - len(value))],
                          mode='constant', constant_values=256)
    else:
        data_row = value[:target_length]
    return data_row


def byte_sequences_feature(sample_path: str, target_length=2000000):
    """
    把整个文件变成字节序列特征，并截断或补长到指定长度
    """
    with open(sample_path, "rb") as file_pointer:
        bytez = file_pointer.read()
        byte_data = np.frombuffer(bytez, dtype=np.uint8)
        byte_data = pad_or_cut(byte_data, target_length)
    return byte_data.astype(np.int16)
