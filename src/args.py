"""
Filename: args.py
用于设置参数以及默认值
"""
import argparse


def get_args():
    """
    设置运行参数
    """
    parser = argparse.ArgumentParser(description='MalConv')
    parser.add_argument('--train_dir', default='./datas/train',
                        help='dir of train samples')
    parser.add_argument('--test_dir', default='./datas/test',
                        help='dir of test samples')
    parser.add_argument('--label_dict', default="./datas/label_dict.json", help='path to label dict')
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
    parser.add_argument('--model', default=None, help='path of model loaded')
    parser.add_argument('--epoch', type=int, default=1, help='training epoch num')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0001')
    parser.add_argument('--ckpt_dir', default="./checkpoint", help='path to pretrained ckpt model file')
    parser.add_argument('--output_file', default="result.txt", help='output result path')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--device_target', default='CPU', help='device target')
    parser.add_argument('--mode', default="graph", help="'graph' or 'pynative'")
    parser.add_argument('--device_num', type=int, default=1)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--rank_id', type=int, default=0)

    args_opt = parser.parse_args()
    return args_opt

