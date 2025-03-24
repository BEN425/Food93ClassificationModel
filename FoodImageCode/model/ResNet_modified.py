import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torchvision.models as models

import math

# fix the 'same' padding option in the Conv2d layer when stride is above 1
class Conv2dSame(torch.nn.Conv2d):

    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.size()[-2:]

        pad_h = self.calc_same_pad(i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])
        pad_w = self.calc_same_pad(i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1])

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

def conv3x3(in_channels, out_channels, kernel, stride, dilated=False):
    if dilated:
        return Conv2dSame(in_channels, out_channels, kernel, stride, dilation=2, bias=False)      
    else:
        return Conv2dSame(in_channels, out_channels, kernel, stride, bias=False)      

def batch_norm_2d(channels, momentum=1e-3, eps=1e-5):
    return nn.BatchNorm2d(channels, momentum=momentum, eps=eps)

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilate=False):
        super().__init__() 

        self.conv1 = conv3x3(in_channels, out_channels, 1, 1, dilate)
        self.bn1 = batch_norm_2d(out_channels)
        
        self.bn2 = batch_norm_2d(out_channels)
        self.conv3 = conv3x3(out_channels, out_channels*4, 1, 1, dilate)
        self.bn3 = batch_norm_2d(out_channels*4)
        self.relu = nn.ReLU()

        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != 4*out_channels:
            self.conv2 = conv3x3(out_channels, out_channels, 3, stride, dilate)
            self.downsample = nn.Sequential(
                conv3x3(in_channels, out_channels*4, 1, stride, dilate),
                batch_norm_2d(out_channels*4)
            )
        else:
            self.conv2 = conv3x3(out_channels, out_channels, 3, 1, dilate)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        return self.relu(out)

class ModifiedResNet(nn.Module):
    def __init__(self, in_channels, out_channels, category_num=24):
        super().__init__()  
        block_nums = [3, 4, 6, 3]
        self.in_channels = 64

        self.conv1 = conv3x3(in_channels, out_channels, 7, 2)
        self.bn1 = batch_norm_2d(out_channels)
        self.relu = nn.ReLU()

        self.layer1 = self._make_layer(Bottleneck, 64, block_nums[0], 1)
        #print("layer1", self.layer1)
        self.layer2 = self._make_layer(Bottleneck, 128, block_nums[1], 2)
        #print("layer2", self.layer2)
        self.layer3 = self._make_layer(Bottleneck, 256, block_nums[2], 2, dilate=True)
        #print("layer3", self.layer3)
        self.layer4 = self._make_layer(Bottleneck, 512, block_nums[3], 2, dilate=True)
        #print("layer4", self.layer4)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*4, category_num)

        self.apply(self.initialize_weights)
    
    def initialize_weights(self, module):
        if isinstance(module, nn.Conv2d):
            init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Linear):
            init.xavier_uniform_(module.weight)
            module.bias.data.zero_()

    def _make_layer(self, block, out_channels, num_blocks, stride=1, dilate=False):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for idx, stride in enumerate(strides):
            if idx == 0 and stride != 1:
                layers.append(block(self.in_channels, out_channels, stride, dilate=False))
            else:
                layers.append(block(self.in_channels, out_channels, 1, dilate))
            self.in_channels = out_channels*4
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # bs, 64, 112, 112
        out1 = self.layer1(out)
        #print("layer1 output:", out1.shape)
        out2 = self.layer2(out1)
        #print("layer2 output:", out2.shape)
        out3 = self.layer3(out2)
        #print("layer3 output:", out3.shape)
        out4 = self.layer4(out3)
        #print("layer4 output:", out4.shape)
        out = self.avgpool(out4)
        #print("avg output:", out.shape)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out 


if __name__ == '__main__':
    model = ModifiedResNet(3, 64)
    pretrained_resnet = models.resnet50(pretrained=True)

    # Load the pretrained weights into ModifiedResNet
    pretrained_dict = pretrained_resnet.state_dict()
    model_dict = model.state_dict()
    #print(model_dict.keys())

    # Filter out unnecessary keys and find out which layers match
    loaded_layers = []
    pretrained_dict_filtered = {
        k: v for k, v in pretrained_dict.items() if k in model_dict and k not in 
        ['fc.weight', 
         'fc.bias', 
         'layer3.0.conv1.weight', 
         'layer3.0.conv2.weight', 
         'layer3.0.conv3.weight', 
         'layer4.0.conv1.weight', 
         'layer4.0.conv2.weight', 
         'layer4.0.conv3.weight']}
    loaded_layers = list(pretrained_dict_filtered.keys())

    # Update the model dictionary with the pretrained weights
    model_dict.update(pretrained_dict_filtered)

    # Load the state_dict into the model
    model.load_state_dict(model_dict)

    # Print the layers that were loaded with pretrained weights
    # print("The following layers were loaded with pretrained weights:")
    # for layer in loaded_layers:
    #     print(layer)

    #Test the model with dummy input
    x = torch.randn(2, 3, 224, 224)
    out = torch.sigmoid(model(x))
    print(out)

