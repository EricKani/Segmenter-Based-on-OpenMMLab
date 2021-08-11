import torch.nn as nn

from ..builder import HEADS
from .decode_head import BaseDecodeHead

from einops import rearrange
from mmcv.cnn.utils.weight_init import trunc_normal_


def init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


@HEADS.register_module()
class MySegmenterHead_Linear(BaseDecodeHead):
    def __init__(self, d_encoder, **kwargs):
        super(MySegmenterHead_Linear, self).__init__(input_transform=None, **kwargs)
        self.d_encoder = d_encoder
        self.head = nn.Linear(self.d_encoder, self.num_classes)
        self.apply(init_weights)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)
        GS = x.shape[-1]

        x = rearrange(x, "b n h w -> b (h w) n")
        x = self.head(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=GS)

        return x