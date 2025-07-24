import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torchvision.models as models

import math

# CBAM Module 
#---------------------------------------------------------------#
class ChannelAttention(nn.Module):
    def __init__(self, in_planes: int, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.tensor) -> torch.Tensor :
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor :
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, planes: int, ratio=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor :
        x = x * self.ca(x)
        x = x * self.sa(x)        
        return x 
#---------------------------------------------------------------#

# SE Block
#---------------------------------------------------------------#
class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor :
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
#---------------------------------------------------------------#


# fix the 'same' padding option in the Conv2d layer when stride is above 1
class Conv2dSame(nn.Conv2d):

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
            bias     = self.bias,
            stride   = self.stride,
            padding  = 0,
            dilation = self.dilation,
            groups   = self.groups,
        )

def conv3x3(in_channels, out_channels, kernel, stride, dilation=False):
    if dilation:
        return Conv2dSame(in_channels, out_channels, kernel, stride, dilation=2, bias=False)      
    else:
        return Conv2dSame(in_channels, out_channels, kernel, stride, bias=False)

def batch_norm_2d(channels, momentum=1e-3, eps=1e-5):
    return nn.SyncBatchNorm(channels, momentum=momentum, eps=eps) \
        if dist.is_initialized() and dist.get_world_size() > 1 else \
        nn.BatchNorm2d(channels, momentum=momentum, eps=eps)

class Bottleneck(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride=1, dilate=False, use_cbam=False, use_se=False):
        super().__init__()
        self.use_cbam = use_cbam
        self.use_se = use_se

        self.conv1 = conv3x3(in_channels, out_channels, 1, 1, dilate)
        self.bn1 = batch_norm_2d(out_channels)

        if stride != 1 or in_channels != 4 * out_channels:
            self.conv2 = conv3x3(out_channels, out_channels, 3, stride, dilate)
        else:
            self.conv2 = conv3x3(out_channels, out_channels, 3, 1, dilate)
        self.bn2 = batch_norm_2d(out_channels)

        self.conv3 = conv3x3(out_channels, out_channels * 4, 1, 1, dilate)
        self.bn3 = batch_norm_2d(out_channels * 4)

        if stride != 1 or in_channels != out_channels * 4:
            self.downsample = nn.Sequential(
                conv3x3(in_channels, out_channels * 4, 1, stride, dilate),
                batch_norm_2d(out_channels * 4)
            )
        else:
            self.downsample = nn.Sequential()

        self.relu = nn.ReLU()

        if self.use_cbam:
            self.cbam = CBAM(out_channels * 4)
        if self.use_se:
            self.se = SEBlock(out_channels * 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor :
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        if self.use_cbam:
            out = self.cbam(out)
        if self.use_se:
            out = self.se(out)
        return self.relu(out)

class ModifiedResNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_classes=24,
            use_cbam_layers: "list[bool]"=[False, False, False, False],
            use_se_layers: "list[bool]"=[False, False, False, False]
        ):

        super().__init__()
        self.in_channels = out_channels
        block_nums = [3, 4, 6, 3]

        self.conv1 = conv3x3(in_channels, out_channels, 7, 2)
        self.bn1 = batch_norm_2d(out_channels)
        self.relu = nn.ReLU()

        self.layer1 = self._make_layer(Bottleneck, 64, block_nums[0], stride=1,
                                       use_cbam=use_cbam_layers[0], use_se=use_se_layers[0])
        #print("layer1", self.layer1)
        self.layer2 = self._make_layer(Bottleneck, 128, block_nums[1], stride=2,
                                        use_cbam=use_cbam_layers[1], use_se=use_se_layers[1])
        #print("layer2", self.layer2)
        self.layer3 = self._make_layer(Bottleneck, 256, block_nums[2], stride=2, dilate=True,
                                        use_cbam=use_cbam_layers[2], use_se=use_se_layers[2])
        #print("layer3", self.layer3)
        self.layer4 = self._make_layer(Bottleneck, 512, block_nums[3], stride=2, dilate=True,
                                        use_cbam=use_cbam_layers[3], use_se=use_se_layers[3])
        #print("layer4", self.layer4)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * 4, num_classes)

        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Conv2d):
            # init.xavier_uniform_(module.weight)
            init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                # module.bias.data.zero_()
                init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            init.xavier_uniform_(module.weight)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            init.ones_(module.weight)
            init.zeros_(module.bias)

    def _make_layer(self, block: nn.Module, out_channels: int, num_blocks: int, stride=1, dilate=False, use_cbam=False, use_se=False):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for idx, stride in enumerate(strides):
            # if idx == 0 and stride != 1:
            #     layers.append(block(self.in_channels, out_channels, stride, dilate=False))
            # else:
            #     layers.append(block(self.in_channels, out_channels, 1, dilate))
            layers.append(block(self.in_channels, out_channels, stride, dilate, use_cbam, use_se))
            self.in_channels = out_channels * 4
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor :

        # Conv 1
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # Conv 2~5
        # bs, 64, 112, 112
        out1 = self.layer1(out)
        # print("layer1 output:", out1.shape)
        out2 = self.layer2(out1)
        # print("layer2 output:", out2.shape)
        out3 = self.layer3(out2)
        # print("layer3 output:", out3.shape)
        out4 = self.layer4(out3)
        # print("layer4 output:", out4.shape)
        
        # CAM
        with torch.no_grad() :
            cam = self.generate_cam(out4)
        
        # Global average pooling
        out = self.avgpool(out4)
        # print("avg output:", out.shape)
        
        # Fully connected layer
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        # print("final output:", out.shape)
        
        return {
            "feat": out4,
            "pred": out,
            "cam" : cam
        }

    def generate_cam(self, feature_vec: torch.Tensor, normalize: bool = False) -> torch.Tensor :
        '''
        Generate CAM by inputting the feature map from the last convolutioon layer into the fully connected layer
        
        Arguments :
            feature_vec `Tensor` "[B, 2048, 14, 14]": Feature map from the last convolution layer
            normalize `bool`: Whether to normalize the CAM. Default is False
        '''
        
        # feature_vec: [B, 2048, 14, 14]
        B = feature_vec.size(0)
        fc_weights = self.fc.weight.data  # [num_classes, 2048]
    
        # Compute CAMs: [B, num_classes, 14, 14]
        cams = torch.einsum('oc,bcxy->boxy', fc_weights, feature_vec)
    
        # Normalize CAMs per sample and per class
        if normalize :
            cams = F.relu(cams)
            cams_reshaped = cams.view(B, cams.size(1), -1)
            min_vals = cams_reshaped.min(dim=2, keepdim=True)[0]
            max_vals = cams_reshaped.max(dim=2, keepdim=True)[0]
            norm_cams = (cams_reshaped - min_vals) / (max_vals - min_vals + 1e-5)
            norm_cams = norm_cams.view_as(cams)
            return norm_cams  # [B, num_classes, 14, 14]
            
        return cams

    # def convol_last_layer(self, x: torch.Tensor) -> torch.Tensor :
    #     out = self.conv1(x)
    #     out = self.bn1(out)
    #     out = self.relu(out)
    #     # bs, 64, 112, 112
    #     out1 = self.layer1(out)
    #     #print("layer1 output:", out1.shape)
    #     out2 = self.layer2(out1)
    #     #print("layer2 output:", out2.shape)
    #     out3 = self.layer3(out2)
    #     #print("layer3 output:", out3.shape)
    #     out4 = self.layer4(out3)
    #     return out4


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

