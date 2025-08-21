import torch
from torch import nn 
from torch.nn import functional as F
from mmcv.runner import BaseModule
from mmcv.cnn import ConvModule

from .builder import HEAD





@HEAD.register_module()
class FlowHead(BaseModule):
    _flow_channels = {'Basic':(128, 64), }
    _flow_kernel = {'Basic': (7, 3)}
    _flow_padding = {'Basic': (3, 1)}

    _h_channels = {'Basic': (256, 192)}
    _h_kernel = {'Basic':{1, 3}}
    _h_padding = {'Basic':{0, 1}}
    
    _out_channel = {'Basic':256}
    _out_kernel = {'Basic':3}
    _out_padding = {'Basic': 1}
    def __init__(self, 
                h_channel:int,
                net_type:str = 'Basic',
                **kwargs):
        super().__init__()
        flow_channels = self._flow_channels.get(net_type)
        flow_kernel = self._flow_kernel.get(net_type)
        flow_padding = self._flow_padding.get(net_type)

        h_channels = self._h_channels.get(net_type)
        h_kernel = self._h_kernel.get(net_type)
        h_padding = self._h_padding.get(net_type)
    
        out_kernel = self._out_kernel.get(net_type)
        out_channel = self._out_channel.get(net_type)
        out_padding = self._out_padding.get(net_type)

        h_net = self._make_encoder(h_channel, h_channels, h_kernel, h_padding, **kwargs)
        self.h_net = nn.Sequential(*h_net)

        flow_net = self._make_encoder(2, flow_channels, flow_kernel, flow_padding, **kwargs)
        self.flow_net =  nn.Sequential(*flow_net)


        self.out_net = ConvModule(
            in_channels=flow_channels[-1] + h_channels[-1], 
            out_channels=out_channel, 
            kernel_size=out_kernel,
            padding=out_padding,
            **kwargs)
        self.predict_layer = nn.Conv2d(
            in_channels=out_channel,
            out_channels=2,
            kernel_size=3,
            padding=1)
        


    def _make_encoder(self, in_channel: int, channels: int, kernels: int,
                      paddings: int, conv_cfg: dict, norm_cfg: dict,
                      act_cfg: dict) -> None:
        encoder = []

        for ch, k, p in zip(channels, kernels, paddings):

            encoder.append(
                ConvModule(
                    in_channels=in_channel,
                    out_channels=ch,
                    kernel_size=k,
                    padding=p,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            in_channel = ch
        return encoder 



    
    def forward(self, flow, h_feat):
        flow_feat = self.flow_net(flow)
        h_feat = self.h_net(h_feat)

        out = self.out_net(torch.cat([flow_feat, h_feat], dim=1))
        return self.predict_layer(out)




