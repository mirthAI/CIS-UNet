

# For ResNet200; layers = [3, 24, 36, 3]
# For ResNet152; layers = [3, 8, 36, 3]
# For ResNet101; layers = [3, 4, 23, 3]
# For Resnet50;  layers = [3, 4, 6, 3]
# For Resnet34;  layers = [3, 4, 6, 3]


import torch
import torch.nn as nn
from monai.networks.nets.resnet import ResNetBlock, ResNetBottleneck, resnet10, resnet18, resnet34, resnet50, resnet101, resnet152, resnet200
from monai.networks.nets.swin_unetr import SwinTransformer
from monai.networks.blocks import UnetrUpBlock, UnetOutBlock, UnetrBasicBlock
from monai.networks.layers.factories import Conv, Norm, Pool
from monai.networks.layers.utils import get_pool_layer
from monai.utils import ensure_tuple_rep
from monai.utils.module import look_up_option


def get_inplanes():
    return [64, 128, 256, 512]


def get_avgpool():
    return [0, 1, (1, 1), (1, 1, 1)]

class ResNet(nn.Module):
    """
    ResNet based on: `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`_
    and `Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet? <https://arxiv.org/pdf/1711.09577.pdf>`_.
    Adapted from `<https://github.com/kenshohara/3D-ResNets-PyTorch/tree/master/models>`_.

    Args:
        block: which ResNet block to use, either Basic or Bottleneck.
            ResNet block class or str.
            for Basic: ResNetBlock or 'basic'
            for Bottleneck: ResNetBottleneck or 'bottleneck'
        layers: how many layers to use.
        block_inplanes: determine the size of planes at each step. Also tunable with widen_factor.
        spatial_dims: number of spatial dimensions of the input image.
        n_input_channels: number of input channels for first convolutional layer.
        conv1_t_size: size of first convolution layer, determines kernel and padding.
        conv1_t_stride: stride of first convolution layer.
        no_max_pool: bool argument to determine if to use maxpool layer.
        shortcut_type: which downsample block to use. Options are 'A', 'B', default to 'B'.
            - 'A': using `self._downsample_basic_block`.
            - 'B': kernel_size 1 conv + norm.
        widen_factor: widen output for each layer.
        num_classes: number of output (classifications).
        feed_forward: whether to add the FC layer for the output, default to `True`.
        bias_downsample: whether to use bias term in the downsampling block when `shortcut_type` is 'B', default to `True`.

    """

    def __init__(
        self,
        block: type[ResNetBlock | ResNetBottleneck] | str,
        layers: list[int],
        block_inplanes: list[int],
        spatial_dims: int = 3,
        n_input_channels: int = 3,
        conv1_t_size: tuple[int] | int = 7,
        conv1_t_stride: tuple[int] | int = 1,
        no_max_pool: bool = False,
        shortcut_type: str = "B",
        widen_factor: float = 1.0,
        num_classes: int = 400,
        feed_forward: bool = True,
        bias_downsample: bool = True,  # for backwards compatibility (also see PR #5477)
    ) -> None:
        super().__init__()

        if isinstance(block, str):
            if block == "basic":
                block = ResNetBlock
            elif block == "bottleneck":
                block = ResNetBottleneck
            else:
                raise ValueError("Unknown block '%s', use basic or bottleneck" % block)

        conv_type: type[nn.Conv1d | nn.Conv2d | nn.Conv3d] = Conv[Conv.CONV, spatial_dims]
        norm_type: type[nn.BatchNorm1d | nn.BatchNorm2d | nn.BatchNorm3d] = Norm[Norm.BATCH, spatial_dims]
        pool_type: type[nn.MaxPool1d | nn.MaxPool2d | nn.MaxPool3d] = Pool[Pool.MAX, spatial_dims]
        avgp_type: type[nn.AdaptiveAvgPool1d | nn.AdaptiveAvgPool2d | nn.AdaptiveAvgPool3d] = Pool[
            Pool.ADAPTIVEAVG, spatial_dims
        ]

        block_avgpool = get_avgpool()
        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool
        self.bias_downsample = bias_downsample
        self.i = 0  # to be deleted
        conv1_kernel_size = ensure_tuple_rep(conv1_t_size, spatial_dims)
        conv1_stride = ensure_tuple_rep(conv1_t_stride, spatial_dims)

        self.conv1 = conv_type(
            n_input_channels,
            self.in_planes,
            kernel_size=conv1_kernel_size,  # type: ignore
            stride=conv1_stride,  # type: ignore
            padding=tuple(k // 2 for k in conv1_kernel_size),  # type: ignore
            bias=False,
        )
        self.bn1 = norm_type(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = pool_type(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0], spatial_dims, shortcut_type, stride=2)
        self.layer2 = self._make_layer(block, block_inplanes[1], layers[1], spatial_dims, shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, block_inplanes[2], layers[2], spatial_dims, shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, block_inplanes[3], layers[3], spatial_dims, shortcut_type, stride=2)
        self.avgpool = avgp_type(block_avgpool[spatial_dims])
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, num_classes) if feed_forward else None

        for m in self.modules():
            if isinstance(m, conv_type):
                nn.init.kaiming_normal_(torch.as_tensor(m.weight), mode="fan_out", nonlinearity="relu")
            elif isinstance(m, norm_type):
                nn.init.constant_(torch.as_tensor(m.weight), 1)
                nn.init.constant_(torch.as_tensor(m.bias), 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(torch.as_tensor(m.bias), 0)

    def _downsample_basic_block(self, x: torch.Tensor, planes: int, stride: int, spatial_dims: int = 3) -> torch.Tensor:
        out: torch.Tensor = get_pool_layer(("avg", {"kernel_size": 1, "stride": stride}), spatial_dims=spatial_dims)(x)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), *out.shape[2:], dtype=out.dtype, device=out.device)
        out = torch.cat([out.data, zero_pads], dim=1)
        return out

    def _make_layer(
        self,
        block: type[ResNetBlock | ResNetBottleneck],
        planes: int,
        blocks: int,
        spatial_dims: int,
        shortcut_type: str,
        stride: int = 1,
    ) -> nn.Sequential:
        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        norm_type: Callable = Norm[Norm.BATCH, spatial_dims]

        downsample: nn.Module | partial | None = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if look_up_option(shortcut_type, {"A", "B"}) == "A":
                downsample = partial(
                    self._downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    spatial_dims=spatial_dims,
                )
            else:
                downsample = nn.Sequential(
                    conv_type(
                        self.in_planes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=self.bias_downsample,
                    ),
                    norm_type(planes * block.expansion),
                )
       
        layers = [
            block(
                in_planes=self.in_planes, planes=planes, spatial_dims=spatial_dims, stride=stride, downsample=downsample
            )
        ]
        self.in_planes = planes * block.expansion
        for _i in range(1, blocks):
            layers.append(block(self.in_planes, planes, spatial_dims=spatial_dims))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(x.shape)
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        # if not self.no_max_pool:
        #     x1 = self.maxpool(x1)
        # print(x.shape)
        xx = self.layer1(x1)
        x2 = self.layer2(xx)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        out = [x, x1, xx, x2, x3, x4]
        return out
    
class CIS_UNet(nn.Module):
    def __init__(self, spatial_dims, in_channels, num_classes, encoder_channels, block=ResNetBlock, layers = (3, 4, 6, 3), block_inplanes = (64, 128, 256, 512),
                feature_size = 48, norm_name = 'instance', conv1_t_stride = 1):
        super().__init__()
        
        self.encoder = ResNet(block=block, layers=layers, block_inplanes=block_inplanes, n_input_channels=in_channels, conv1_t_stride=conv1_t_stride)
        self.swintransformer = SwinTransformer(in_chans=block_inplanes[-1], embed_dim=feature_size, window_size=(3, 3, 3), patch_size=1, depths=(8,0,0,0), num_heads=(24,0,0,0))
        self.decoder5 = UnetrUpBlock(spatial_dims=spatial_dims, in_channels= 2 * feature_size, out_channels = feature_size, kernel_size=3, upsample_kernel_size=2, norm_name=norm_name, res_block=True)
        self.decoder4 = UnetrUpBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=block_inplanes[-2], kernel_size=3, upsample_kernel_size=2, norm_name=norm_name, res_block=True)
        self.decoder3 = UnetrUpBlock(spatial_dims=spatial_dims,in_channels=block_inplanes[-2],out_channels=block_inplanes[-3],kernel_size=3,upsample_kernel_size=2,norm_name=norm_name,res_block=True)
        self.decoder2 = UnetrUpBlock(spatial_dims=spatial_dims,in_channels=block_inplanes[-3],out_channels=block_inplanes[-4],kernel_size=3,upsample_kernel_size=2,norm_name=norm_name,res_block=True)
        self.decoder1 = UnetrUpBlock(spatial_dims=spatial_dims,in_channels=block_inplanes[-4],out_channels=block_inplanes[-4],kernel_size=3,upsample_kernel_size=2,norm_name=norm_name,res_block=True)
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=block_inplanes[-4], out_channels=num_classes)
        # self.out = UnetrBasicBlock(spatial_dims=spatial_dims,in_channels=in_channels,out_channels=num_classes,kernel_size=3,stride=1,norm_name=norm_name, res_block=True)
    def forward(self, x):
        out = self.encoder(x)
        x0, x1, x2, x3, x4, x5 = out
        # print(f"x0: {x0.shape} | x1: {x1.shape} | x2: {x2.shape} | x3: {x3.shape} | x4: {x4.shape} | x5: {x5.shape} ")
        bn = self.swintransformer(x5)
        # print(bn[0].shape, bn[1].shape)
        bn0, bn1 = bn[:2]
        dec = self.decoder5(bn1, bn0)
        # print(dec.shape, x4.shape)
        dec = self.decoder4(dec, x4)
        # print(dec.shape, x3.shape)
        dec = self.decoder3(dec, x3)
        # print(dec.shape, x2.shape)
        dec = self.decoder2(dec, x2)
        # print(dec.shape, x1.shape)
        dec = self.decoder1(dec, x1)
        return self.out(dec)
        # return out
