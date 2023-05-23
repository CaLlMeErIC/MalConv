"""
Filename: train.py
用于训练模型
"""
import mindspore.dataset as ds
from mindspore.train import Model, LossMonitor, TimeMonitor, CheckpointConfig, ModelCheckpoint
from mindspore import nn
from src.model import MalConv as Net
from src.dataset import MalConvDataSet
from src.environment import init_env
from src.args import get_args


def train_net():
    """
    训练网络
    """
    # 初始化运行时环境
    args_opt = get_args()
    print(args_opt)
    init_env(args_opt)
    # 构造数据集对象
    dataset_ori = ds.GeneratorDataset(
        MalConvDataSet(args_opt=args_opt, data_type="train"), shuffle=True,
        column_names=["data", "label"])
    dataset = dataset_ori.batch(args_opt.batch_size, drop_remainder=False)
    # 网络模型，和任务有关
    net = Net()
    # 损失函数，和任务有关
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    # 优化器实现，和任务有关
    optimizer = nn.Adam(net.trainable_params(), args_opt.lr)
    # 封装成Model
    model = Model(net, loss_fn=loss, optimizer=optimizer,
                  metrics={'top_1_accuracy', 'top_5_accuracy'})
    # checkpoint保存
    config_ck = CheckpointConfig(save_checkpoint_steps=dataset.get_dataset_size(),
                                 keep_checkpoint_max=5)
    ckpt_cb = ModelCheckpoint(prefix="MalConv", directory=args_opt.ckpt_dir, config=config_ck)
    # 模型训练
    model.train(args_opt.epoch, dataset, callbacks=[LossMonitor(), TimeMonitor(), ckpt_cb])


if __name__ == '__main__':
    train_net()
