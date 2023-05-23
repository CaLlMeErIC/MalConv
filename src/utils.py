"""
Filename: utils.py
用于存放常用读写函数
"""
import os
import json


def load_json_data(file_path: str):
    """
    读取json文件为字典
    """
    with open(file_path, 'r', encoding='utf8') as file_pointer:
        json_data = json.load(file_pointer)
    return json_data


def get_data_path(data_dir: str, file_type=""):
    """
    获取指定文件下所有文件的全路径，返回一个列表
    """
    all_files = os.walk(data_dir)
    print("正在读取" + data_dir + "下的所有文件路径名")
    print("读取的文件后缀名：" + file_type)
    result_ls = []
    for path, _, filelist in all_files:
        for filename in filelist:
            if filename.endswith(file_type):
                final_path = os.path.join(path, filename)
                final_path = final_path.replace('\\', '/')
                # 统一换成/结尾
                result_ls.append(final_path)
    return result_ls


def dict_to_json(dict_data: dict, save_path: str):
    """
    字典保存为json文件
    """
    if not dict_data:
        return
    with open(save_path, 'w', encoding='utf-8') as file_pointer:
        json.dump(dict_data, file_pointer, indent=4)
