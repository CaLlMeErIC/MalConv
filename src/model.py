"""
filename:model.py
References:
        - Edward Raff et al. 2018. Malware Detection by Eating a Whole EXE.
          https://arxiv.org/abs/1710.09435
"""

from typing import Optional
from mindspore import nn
from mindspore import Tensor
import mindspore


class MalConv(nn.Cell):
    """
    The MalConv model.
    """

    def __init__(
            self,
            num_classes: int = 2,
            *,
            num_embeddings: int = 257,
            embedding_dim: int = 8,
            channels: int = 128,
            kernel_size: int = 512,
            stride: int = 512,
            padding_idx: Optional[int] = 256,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes

        # By default, num_embeddings (257) = byte (0-255) + padding (256).
        self.embedding = nn.Embedding(num_embeddings, embedding_dim,
                                      padding_idx=padding_idx,)

        self.conv1 = nn.Conv1d(embedding_dim, channels, kernel_size=kernel_size,
                               stride=stride, has_bias=True)
        self.conv2 = nn.Conv1d(embedding_dim, channels, kernel_size=kernel_size,
                               stride=stride, has_bias=True)

        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.func = nn.SequentialCell(
            nn.Flatten(),
            nn.Dense(channels, channels),
            nn.ReLU(),
            nn.Dense(channels, num_classes),
        )

    def embed(self, input_x: Tensor) -> Tensor:
        """
        Perform embedding.
        """
        input_x = self.embedding(input_x)

        # Treat embedding dimension as channel.
        input_x = input_x.permute(0, 2, 1)
        return input_x

    def forward_embedded(self, input_x: Tensor) -> Tensor:
        """
        Perform gate convolution
        """
        input_x = self.conv1(input_x) * mindspore.ops.sigmoid(self.conv2(input_x))

        # Perform global max pooling.
        input_x = self.max_pool(input_x)

        input_x = self.func(input_x)
        return input_x

    def construct(self, input_x: Tensor) -> Tensor:
        """
        construct network
        """
        input_x = self.embed(input_x)
        input_x = self.forward_embedded(input_x)
        return input_x
