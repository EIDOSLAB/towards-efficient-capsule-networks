import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.capsule import LinearCaps2d, PrimaryCaps2d
from ops.utils import conv2d_output_shape

class Block(nn.Module):
    """Depthwise conv + Pointwise conv"""

    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            in_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_planes,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(
            in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class MobileNet(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [
        64,
        (128, 2),
        128,
        (256, 2),
        256,
        (512, 2),
        512,
        512,
        512,
        512,
        512,
        (1024, 2),
        1024,
    ]

    def __init__(self, num_classes=10):
        super(MobileNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.linear = nn.Linear(1024, num_classes)
        self.relu = nn.ReLU()

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def load_checkpoint(checkpoint_file):
    """
    Function to load pruned model or normal model checkpoint.
    :param str checkpoint_file: path to checkpoint file, such as `models/ckpt/mobilenet.pth`
    """
    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    net = MobileNet()
    #### load pruned model ####
    for key, module in net.named_modules():
        # torch.nn.BatchNorm2d
        if isinstance(module, nn.BatchNorm2d):
            module.weight = torch.nn.Parameter(checkpoint[key + ".weight"])
            module.bias = torch.nn.Parameter(checkpoint[key + ".bias"])
            module.num_features = module.weight.size(0)
            module.running_mean = module.running_mean[0 : module.num_features]
            module.running_var = module.running_var[0 : module.num_features]
        # torch.nn.Conv2d
        elif isinstance(module, nn.Conv2d):
            # for conv2d layer, bias and groups should be consider
            module.weight = torch.nn.Parameter(checkpoint[key + ".weight"])
            module.out_channels = module.weight.size(0)
            module.in_channels = module.weight.size(1)
            if module.groups is not 1:
                # group convolution case
                # only support for MobileNet, pointwise conv
                module.in_channels = module.weight.size(0)
                module.groups = module.in_channels
            if key + ".bias" in checkpoint:
                module.bias = torch.nn.Parameter(checkpoint[key + ".bias"])
        # torch.nn.Linear
        elif isinstance(module, nn.Linear):
            module.weight = torch.nn.Parameter(checkpoint[key + ".weight"])
            if key + ".bias" in checkpoint:
                module.bias = torch.nn.Parameter(checkpoint[key + ".bias"])
            module.out_features = module.weight.size(0)
            module.in_features = module.weight.size(1)

    net.load_state_dict(checkpoint)
    return net

class Mobilev1CapsNet(nn.Module):

    def __init__(self, config, device):
        super(Mobilev1CapsNet, self).__init__()
        self.config = config
        self.device = device

        if config.backbone_ratio_remain_flops == 50:
            self.mobilenet = load_checkpoint("dump/models/EagleEye/mobilenetv1_50flops_latest.pth")
            if config.dataset == "cifar10":
                self.c0, self.h0, self.w0 = 619, 2,2
            else:
                self.c0, self.h0, self.w0 = 619, 7,7
        else:
            self.mobilenet = load_checkpoint("dump/models/EagleEye/imagenet_mobilenet_full_model.pth")
            if config.dataset == "cifar10":
                self.c0, self.h0, self.w0 = 1024, 2,2
            else:
                self.c0, self.h0, self.w0 = 1024, 7,7
        modules=list(self.mobilenet.children())[:-2]

        self.mobilenet = nn.Sequential(*modules)
        if self.config["freeze"]:
            for p in self.mobilenet.parameters():
                p.requires_grad = False

            self.mobilenet.eval()


        self.num_primaryCaps_types = int(self.c0/config.dim_primaryCaps[0])
        self.primaryCaps = PrimaryCaps2d(input_channels=self.c0,
                                         input_height=self.h0,
                                         input_width=self.w0,
                                         kernel_size=self.h0,
                                         stride=config.stride_primaryCaps,
                                         padding=config.padding_primaryCaps,
                                         dilation=config.dilation_primaryCaps,
                                         routing_method=config.routing,
                                         num_iterations=config.primary_num_routing_iterations,
                                         squashing=config.squashing_primaryCaps,
                                         output_caps_types=self.num_primaryCaps_types,
                                         output_caps_shape=config.dim_primaryCaps,
                                         device=device)

        self.h1, self.w1 = conv2d_output_shape((self.h0, self.w0),
                                     kernel_size=self.h0,
                                     stride=config.stride_primaryCaps,
                                     pad=config.padding_primaryCaps,
                                     dilation=config.dilation_primaryCaps)

        self.num_primary_units = self.h1 * self.w1 * self.num_primaryCaps_types
        self.classCaps = LinearCaps2d(input_height=self.h1,
                                     input_width=self.w1,
                                     routing_method=config.routing,
                                     num_iterations=config.num_routing_iterations,
                                     input_caps_types=self.num_primaryCaps_types,
                                     input_caps_shape=config.dim_primaryCaps,
                                     output_caps_types=config.num_classes,
                                     output_caps_shape=config.dim_classCaps,
                                     transform_share=config.transform_share_classCaps,
                                     device=device)

    def forward(self, x, target=None):
        """
        The dimension transformation procedure of an input tensor in each layer:
            0. input: [b, c0, h0, w0] -->
            1. resnet50 --> [b, c1, h1, w1] -->
            2. primaryCaps poses --> [b, B, h2, w2, is0, is1] -->
            3. classCaps poses --> [b, C, 1, 1, os0, os1] -->
            4. view poses --> [b, C, os0, os1]
        :param x: Image tensor, shape [b, channels, ih, iw]
        :param target: One-hot encoded target tensor, shape [b, num_classes]
        :return: (output_caps_poses, output_caps_activations)
                 The capsules poses and activations tensors of capsule class layer.
                 output_caps_poses: [b, C, os0, os1], output_caps_activations: [b, C]
        """
        batch_size = x.size(0)
        # Input: [b, c0, h0, w0]
        x = F.relu(self.mobilenet(x))
        #print(x.size())
        #x = self.padding(x)
        #print(x.size())
    
        # x: [b, c1, h1, w1]
        output_caps_poses, output_caps_activations = self.primaryCaps(x)
        self.primary_caps_activations = output_caps_activations
        # output_caps_poses: [b, B, h2, w2, is0, is1]
        # output_caps_activations: [b, B, h2, w2]
        output_caps_poses, output_caps_activations = self.classCaps(output_caps_poses, output_caps_activations)
        # output_caps_poses: [b, C, 1, 1, os0, os1]
        # output_caps_activations: [b, C, 1, 1]

        output_caps_poses = output_caps_poses.view(batch_size, output_caps_poses.size(1),
                                                       output_caps_poses.size(-2),
                                                       output_caps_poses.size(-1))
        output_caps_activations = output_caps_activations.view(batch_size, output_caps_poses.size(1))
        # output_caps_poses: [b, C, os0, os1]
        # output_caps_activations: [b, C]
        return output_caps_poses, output_caps_activations