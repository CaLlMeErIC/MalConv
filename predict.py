"""
Filename: train.py
用于测试模型
"""
import mindspore.dataset as ds
import mindspore
import numpy as np
from src.utils import get_data_path, dict_to_json
from src.model import MalConv as Net
from src.dataset import MalConvDataSet
from src.args import get_args


def softmax(input_x):
    """
    使用softmax把结果化成概率
    """
    x_exp = np.exp(input_x)
    # 如果是列向量，则axis=0
    x_sum = np.sum(x_exp, axis=0, keepdims=True)
    result = x_exp / x_sum
    return result


def eval_net():
    """
    读取保存的模型
    """
    args_opt = get_args()
    dataset_ori = ds.GeneratorDataset(
        MalConvDataSet(args_opt=args_opt, data_type="test"), shuffle=True,
        column_names=["data", "file_md5"])
    dataset = dataset_ori.batch(1, drop_remainder=False)
    model = Net()
    if args_opt.model is not None:
        model_path = args_opt.model
    else:
        # 如果没有指定模型路径就选用checkpoint文件夹下的第一个文件
        model_path = get_data_path(args_opt.ckpt_dir, file_type="ckpt")[0]
    param_dict = mindspore.load_checkpoint(model_path)
    param_not_load, _ = mindspore.load_param_into_net(model, param_dict)
    if param_not_load:
        print("有参数未加载:", param_not_load)
    result_dict = {}
    for data in dataset:
        result = model(data[0])
        malware_prob =float(softmax(result[0])[1])
        md5=str(data[1])[2:-2]
        print("md5:", md5, "malware prob:", malware_prob)
        result_dict.update({md5: malware_prob})
    dict_to_json(result_dict, args_opt.output_file)


if __name__ == '__main__':
    eval_net()
