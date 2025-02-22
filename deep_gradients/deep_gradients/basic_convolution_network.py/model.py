"""
Full defination of Resnet-50, all of it in this single file. 
References:

"""

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F

from typing import Optional, Callable, Type, Union, List



def conv3x3(stride: int = 1, padding: int = 1) -> nn.Conv2d:
    "3x3 convolution with padding"
    return nn.Conv2d(
        kernel_size = 3, #cause 3x3 convolution matrix
        stride = stride,
        padding = padding,
        bias = False,
    )

def conv1x1(stride:int=1)-> nn.Conv2d:
    "1x1 convolution"
    return nn.Conv2d(kernel_size=1. stride=stride, bias=False)


class BasicBlock(nn.Module):

    def __init__(self, stride: int = 1, base_width: int = 64, dilation: int = 1, norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if groups !=1 or base_width !=64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation >1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        
        #Both self.conv1 and self.downsample layers downsample the input when stride !=1 
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    
    def forward(self, x:Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

    
    
    
class Bottleneck(nn.Module):

    def __init__(
            self, 
            inplanes: int,
            planes: int,
            stride: int =1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer = Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width/64.0)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv1 = conv1x1(inplanes, width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes*self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace = True)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x:Tensor) -> Tensor:
        identify = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    

    class ResNet(nn.Module):

        def __init__(
                self, 
                block: Type[Union[BasicBlock, Bottleneck]],
                layers: List[int], 
                num_classes: int = 1000,
                zero_init_residual: bool = False,
                groups : int = 1, 
                width_per_group: int = 64,
                replace_stride_with_dilation: Optional[List[bool]] = None,
                norm_layer: Optional[Callable[..., nn.Module]] = None,
        ) -> None:
            super().__init__()

            if norm_layer is None:
                norm_layer = nn.BatchNorm2d
            self._norm_layer = norm_layer

            self.inplanes = 64
            self.dilation = 1
            if replace_stride_with_dilation is None:

                replace_stride_with_dilation = [False, False, False]
            
            if len(replace_stride_with_dilation) !=3:
                raise ValueError("replace_stride_with_dilation should be None")
            
            self.groups = groups
            self.base_width = width_per_group
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size = 7, stride = 2, padding = 3, bias=False)
            self.bn1 = norm_layer(self.inplanes)
            self.relu = nn.ReLU(inplace = True)
            self.maxpool = nn.MaxPool2d(kernel_size = 3, stride=2, padding=1)
            self.layer1 = self._make_layer(block, 64, layers[0])
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=(replace_stride_with_dilation[0]))
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate = (replace_stride_with_dilation[1]))
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate = (replace_stride_with_dilation[1]))
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)


        def _make_layer(
                self, 
                block: Type[Union[BasicBlock, Bottleneck]],
                planes:int,
                blocks: int, 
                stride:int=1,
                dilate:bool = False
        ) -> nn.Sequential:
             
            norm_layer = self._norm_layer
            downsample = None
            previous_dilation = self.dilation
            if dilate:
                self.dilation *= stride
                stride = 1
            
            if stride !=1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes*block.expansion, stride),
                    norm_layer(planes * block.expansion)
                    )
                
            layers = []
            layers.append(
                block(
                    self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
                )
            )
            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(
                    block(
                        self.inplanes,
                        planes,
                        groups=self.groups,
                        base_width=self.base_width,
                        dilation=self.dilation,
                        norm_layer=norm_layer,
                    )
                )

            return nn.Sequential(*layers)
        
        def _forward_impl(self, x: Tensor) -> Tensor:
            # See note [TorchScript super()]
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

            return x

        def forward(self, x: Tensor) -> Tensor:
            return self._forward_impl(x) 
        
        def _forward_impl(self, x: Tensor) -> Tensor:
            # See note [TorchScript super()]
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

            return x

        def forward(self, x: Tensor) -> Tensor:
            return self._forward_impl(x)



