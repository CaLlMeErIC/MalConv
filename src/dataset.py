"""
Filename: dataset.py
准备数据集
"""
import numpy as np
from src.utils import get_data_path, load_json_data
from src.feature import byte_sequences_feature


def train_data_loader(file_path: str, label_dict=None):
    """
    用于读取单标签的情况,1为恶意标签
    """
    if label_dict is None:
        label_dict = {}
    file_md5 = file_path.split('/')[-1]
    if file_md5 in label_dict:
        return byte_sequences_feature(file_path), np.int32(1)
    return byte_sequences_feature(file_path), np.int32(0)


def test_data_loader(file_path: str, _):
    """
    用于读取预测文件的情况
    """
    file_md5 = file_path.split('/')[-1]
    return byte_sequences_feature(file_path), file_md5


class MalConvDataSet:
    """
    准备数据集，分为训练和测试两种情况
    """

    def __init__(self, args_opt, data_type="train"):

        self.label_dict = load_json_data(args_opt.label_dict)

        if data_type == "train":
            self.loader = train_data_loader
            self.file_list = get_data_path(args_opt.train_dir)
        else:
            self.loader = test_data_loader
            self.file_list = get_data_path(args_opt.test_dir)

        np.random.seed(args_opt.seed)
        np.random.shuffle(self.file_list)

    def __getitem__(self, index):
        file_path = self.file_list[index]
        feature, label = self.loader(file_path, self.label_dict)
        return np.array(feature), label

    def __len__(self):
        return len(self.file_list)


if __name__ == "__main__":
    pass
