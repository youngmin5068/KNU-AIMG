import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import conv3x3
from tools import *



class BasicBlockV2(nn.Module):

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):

        super(BasicBlockV2, self).__init__()
        self.relu = nn.LeakyReLU(inplace=True)
        self.bn1 = nn.GroupNorm(inplanes,inplanes)
        self.conv1 = conv3x3(inplanes, planes, stride=stride)
        self.bn2 = nn.GroupNorm(planes,planes)
        self.conv2 = conv3x3(planes, planes, stride=1)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += residual
        return out

class BasicBlockV1(nn.Module):

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlockV1, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.GroupNorm(planes,planes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.GroupNorm(planes,planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNetV2(nn.Module):
    def __init__(self,
                 input_channels, num_filters,
                 first_layer_kernel_size, first_layer_conv_stride,
                 blocks_per_layer_list, block_strides_list, block_fn,
                 first_layer_padding=0,
                 first_pool_size=None, first_pool_stride=None, first_pool_padding=0,
                 growth_factor=2):
        super(ResNetV2, self).__init__()
        self.first_conv = nn.Conv2d(
            in_channels=input_channels, out_channels=num_filters,
            kernel_size=first_layer_kernel_size,
            stride=first_layer_conv_stride,
            padding=first_layer_padding,
            bias=False,
        )
        self.first_pool = nn.MaxPool2d(
            kernel_size=first_pool_size,
            stride=first_pool_stride,
            padding=first_pool_padding,
        )
        self.layer_list = nn.ModuleList()
        current_num_filters = num_filters
        self.inplanes = num_filters
        for i, (num_blocks, stride) in enumerate(zip(
                blocks_per_layer_list, block_strides_list)):
            self.layer_list.append(self._make_layer(
                block=block_fn,
                planes=current_num_filters,
                blocks=num_blocks,
                stride=stride,
            ))
            current_num_filters *= growth_factor
        
        self.final_bn = nn.GroupNorm(
            current_num_filters // growth_factor * block_fn.expansion,  current_num_filters // growth_factor * block_fn.expansion
        )
        self.relu = nn.LeakyReLU()
        self.num_filters = num_filters
        self.growth_factor = growth_factor

    def forward(self, x):

        h = self.first_conv(x)
        h = self.first_pool(h)
        for i, layer in enumerate(self.layer_list):
            h = layer(h)
        h = self.final_bn(h)
        h = self.relu(h)

        return h

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = nn.Sequential(
            nn.Conv2d(self.inplanes, planes * block.expansion,
                      kernel_size=1, stride=stride, bias=False),
        )
        layers_ = [
            block(self.inplanes, planes, stride, downsample)
        ]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers_.append(block(self.inplanes, planes))

        return nn.Sequential(*layers_)

class ResNetV1(nn.Module):

    def __init__(self, initial_filters, block, layers, input_channels=1):

        self.inplanes = initial_filters
        self.num_layers = len(layers)
        super(ResNetV1, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, initial_filters, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.GroupNorm(initial_filters,initial_filters)
        self.relu = nn.LeakyReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        for i in range(self.num_layers):
            num_filters = initial_filters * pow(2,i)
            num_stride = (1 if i == 0 else 2)
            setattr(self, 'layer{0}'.format(i+1), self._make_layer(block, num_filters, layers[i], stride=num_stride))
        self.num_filter_last_seq = initial_filters * pow(2, self.num_layers-1)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(planes * block.expansion,planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)
    
    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        for i in range(self.num_layers):
            x = getattr(self, 'layer{0}'.format(i+1))(x)
        return x

class DownsampleNetworkResNet18V1(ResNetV1):
    def __init__(self):
        super(DownsampleNetworkResNet18V1, self).__init__(
            initial_filters=64,
            block=BasicBlockV1,
            layers=[2, 2, 2, 2],
            input_channels=3)
    def forward(self, x):
        last_feature_map = super(DownsampleNetworkResNet18V1, self).forward(x)
        return last_feature_map

class AbstractMILUnit:
    def __init__(self, parameters, parent_module):
        self.parameters = parameters
        self.parent_module = parent_module

class PostProcessingStandard(nn.Module):
    def __init__(self, parameters):
        super(PostProcessingStandard, self).__init__()
        self.gn_conv_last = nn.Conv2d(parameters["post_processing_dim"],
                                      parameters["num_classes"],
                                      (1, 1), bias=False)
    def forward(self, x_out):
        out = self.gn_conv_last((x_out))

        return torch.sigmoid(out)

class GlobalNetwork(AbstractMILUnit):

    def __init__(self, parameters, parent_module):
        super(GlobalNetwork, self).__init__(parameters, parent_module)
        self.downsampling_branch = ResNetV2(input_channels=1, num_filters=16,
                    first_layer_kernel_size=(7,7), first_layer_conv_stride=2,
                    first_layer_padding=3,
                    first_pool_size=3, first_pool_stride=2, first_pool_padding=0,
                    blocks_per_layer_list=[2, 2, 2, 2, 2],
                    block_strides_list=[1, 2, 2, 2, 2],
                    block_fn=BasicBlockV2,
                    growth_factor=2)
        self.postprocess_module = PostProcessingStandard(parameters)
    def add_layers(self):
        self.parent_module.ds_net = self.downsampling_branch
        self.parent_module.left_postprocess_net = self.postprocess_module
    def forward(self, x):
        last_feature_map = self.downsampling_branch.forward(x)
        cam = self.postprocess_module.forward( (last_feature_map))

        return last_feature_map, cam

class TopTPercentAggregationFunction(AbstractMILUnit):

    def __init__(self, parameters, parent_module):
        super(TopTPercentAggregationFunction, self).__init__(parameters, parent_module)
        self.percent_t = parameters["percent_t"]
        self.parent_module = parent_module

    def forward(self, cam):
        batch_size, num_class, H, W = cam.size()
        cam_flatten = cam.view(batch_size, num_class, -1)
        top_t = int(round(W*H*self.percent_t))
        selected_area = cam_flatten.topk(top_t, dim=2)[0]

        return selected_area.mean(dim=2)

class RetrieveROIModule(AbstractMILUnit):

    def __init__(self, parameters, parent_module):

        super(RetrieveROIModule, self).__init__(parameters, parent_module)
        self.crop_method = "upper_left"
        self.num_crops_per_class = parameters["K"]
        self.crop_shape = parameters["crop_shape"]
        self.gpu_number = None if parameters["device_type"]!="gpu" else parameters["gpu_number"]

    def forward(self, x_original, cam_size, h_small):

        _, _, H, W = x_original.size()
        (h, w) = cam_size
        N, C, h_h, w_h = h_small.size()

        assert h_h == h, "h_h!=h"
        assert w_h == w, "w_h!=w"

        crop_x_adjusted = int(np.round(self.crop_shape[0] * h / H))
        crop_y_adjusted = int(np.round(self.crop_shape[1] * w / W))
        crop_shape_adjusted = (crop_x_adjusted, crop_y_adjusted)

        current_images = h_small
        all_max_position = []
        max_vals = current_images.view(N, C, -1).max(dim=2, keepdim=True)[0].unsqueeze(-1)
        min_vals = current_images.view(N, C, -1).min(dim=2, keepdim=True)[0].unsqueeze(-1)
        range_vals = max_vals - min_vals
        normalize_images = current_images - min_vals
        normalize_images = normalize_images / range_vals
        current_images = normalize_images.sum(dim=1, keepdim=True)

        for _ in range(self.num_crops_per_class):

            max_pos = get_max_window(current_images, crop_shape_adjusted, "avg")
            all_max_position.append(max_pos)
            mask = generate_mask_uplft(current_images, crop_shape_adjusted, max_pos, self.gpu_number)
            current_images = current_images * mask
        return torch.cat(all_max_position, dim=1).data.cpu().numpy()

class LocalNetwork(AbstractMILUnit):

    def add_layers(self):
        self.parent_module.dn_resnet = ResNetV1(64, BasicBlockV1, [2,2,2,2], 3)
    def forward(self, x_crop):

        res = self.parent_module.dn_resnet(x_crop.expand(-1, 3, -1 , -1))
        res = res.mean(dim=2).mean(dim=2)
        return res

class AttentionModule(AbstractMILUnit):

    def add_layers(self):

        self.parent_module.mil_attn_V = nn.Linear(512, 128, bias=False)
        self.parent_module.mil_attn_U = nn.Linear(512, 128, bias=False)
        self.parent_module.mil_attn_w = nn.Linear(128, 1, bias=False)
        self.parent_module.classifier_linear = nn.Linear(512, self.parameters["num_classes"], bias=False)

    def forward(self, h_crops):

        batch_size, num_crops, h_dim = h_crops.size()
        h_crops_reshape = h_crops.view(batch_size * num_crops, h_dim)
        attn_projection = torch.sigmoid(self.parent_module.mil_attn_U(h_crops_reshape)) * torch.tanh(self.parent_module.mil_attn_V(h_crops_reshape))

        attn_score = self.parent_module.mil_attn_w(attn_projection)
        attn_score_reshape = attn_score.view(batch_size, num_crops)
        attn = F.softmax(attn_score_reshape, dim=1)
        z_weighted_avg = torch.sum(attn.unsqueeze(-1) * h_crops, 1)
        y_crops = torch.sigmoid(self.parent_module.classifier_linear(z_weighted_avg))
        return z_weighted_avg, attn, y_crops

