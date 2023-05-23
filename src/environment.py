"""
Filename: environment.py
用于设置环境变量
"""
import mindspore as ms
from mindspore.communication.management import init, get_rank, get_group_size


class DefaultConfig:
    """
    设置默认的环境
    """

    def __init__(self):
        self.seed = 1
        self.device_target = "CPU"
        self.context_mode = "graph"  # should be in ['graph', 'pynative']
        self.device_num = 1
        self.device_id = 0
        self.rank_id = 0

    def print_args(self):
        """
        打印参数
        """
        print("seed", self.seed)
        print("target", self.device_target)
        print("mode", self.context_mode)
        print("device num", self.device_num)
        print("device id", self.device_id)
        print("rank id", self.rank_id)

    def set_args(self, input_args):
        """
        设置参数
        """
        if input_args.seed:
            self.seed = 1
        if input_args.device_target:
            self.device_target = input_args.device_target
        if input_args.mode:
            self.context_mode = input_args.mode
        if input_args.device_num:
            self.device_num = input_args.device_num
        if input_args.device_id:
            self.device_id = input_args.device_id
        if input_args.rank_id:
            self.rank_id = input_args.rank_id


def init_env(args_opt=None):
    """初始化运行时环境."""
    cfg = DefaultConfig()
    cfg.set_args(args_opt)
    ms.set_seed(cfg.seed)
    # 如果device_target设置是None，利用框架自动获取device_target，否则使用设置的。
    if cfg.device_target != "None":
        if cfg.device_target not in ["Ascend", "GPU", "CPU"]:
            raise ValueError(f"Invalid device_target: {cfg.device_target}, "
                             f"should be in ['None', 'Ascend', 'GPU', 'CPU']")
        ms.set_context(device_target=cfg.device_target)

    # 配置运行模式，支持图模式和PYNATIVE模式
    if cfg.context_mode not in ["graph", "pynative"]:
        raise ValueError(f"Invalid context_mode: {cfg.context_mode}, "
                         f"should be in ['graph', 'pynative']")
    context_mode = ms.GRAPH_MODE if cfg.context_mode == "graph" else ms.PYNATIVE_MODE
    ms.set_context(mode=context_mode)

    cfg.device_target = ms.get_context("device_target")
    # 如果是CPU上运行的话，不配置多卡环境
    if cfg.device_target == "CPU":
        cfg.device_id = 0
        cfg.device_num = 1
        cfg.rank_id = 0

    # 设置运行时使用的卡
    if hasattr(cfg, "device_id") and isinstance(cfg.device_id, int):
        ms.set_context(device_id=cfg.device_id)

    if cfg.device_num > 1:
        # init方法用于多卡的初始化，不区分Ascend和GPU，get_group_size和get_rank方法只能在init后使用
        init()
        print("run distribute!", flush=True)
        group_size = get_group_size()
        if cfg.device_num != group_size:
            raise ValueError(f"the setting device_num: {cfg.device_num} "
                             f"not equal to the real group_size: {group_size}")
        cfg.rank_id = get_rank()
        ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL,
                                     gradients_mean=True)
        if hasattr(cfg, "all_reduce_fusion_config"):
            ms.set_auto_parallel_context(all_reduce_fusion_config=cfg.all_reduce_fusion_config)
    else:
        cfg.device_num = 1
        cfg.rank_id = 0
        print("run standalone!", flush=True)


if __name__ == "__main__":
    init_env()
