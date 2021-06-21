'''MobileNetV2 in PyTorch.

See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .gate_function import virtual_gate

class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride, cfg=None):
        super(Block, self).__init__()
        self.stride = stride

        if cfg is None:
            planes = expansion * in_planes
        else:
            planes = cfg

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.gate = virtual_gate(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)


        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )


    def forward(self, x):

        out = self.bn1(self.conv1(x))
        out = F.relu(out)
        #out = F.leaky_relu(out)
        out = self.gate(out)

        out = self.bn2(self.conv2(out))
        out = F.relu(out)
        #out = F.leaky_relu(out)
        out = self.gate(out)

        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out

cfg = [(1,  16, 1, 1),
       (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
       (6,  32, 3, 1),
       (6,  64, 4, 2),
       (6,  96, 3, 1),
       (6, 160, 3, 2),
       (6, 320, 1, 1)]

class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)


    def __init__(self, num_classes=10, custom_cfg = None):
        super(MobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        if custom_cfg is None:
            self.cfg = cfg
            cfg_flag = False
        else:
            self.cfg = custom_cfg
            cfg_flag = True
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32, custom_cfg=cfg_flag)

        #self.gate = soft_gate(320)

        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)

    def _make_layers(self, in_planes, custom_cfg = None):
        layers = []
        if custom_cfg is False:
            for expansion, out_planes, num_blocks, stride in self.cfg:
                strides = [stride] + [1]*(num_blocks-1)
                for stride in strides:
                    layers.append(Block(in_planes, out_planes, expansion, stride))
                    in_planes = out_planes
        else:
            for expansion, out_planes, num_blocks, stride, planes_list in self.cfg:
                strides = [stride] + [1]*(num_blocks-1)
                idx = 0
                for stride in strides:
                    layers.append(Block(in_planes, out_planes, expansion, stride, cfg=planes_list[idx]))
                    in_planes = out_planes
                    idx+=1

        return nn.Sequential(*layers)

    def forward(self, x, feature_out=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)

        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        #out = F.avg_pool2d(out, 4)
        #print(out.size())
        if feature_out:
            feature = out

        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        if feature_out:
            return out, feature
        else:
            return out
    def count_structure(self):
        structure = []
        for m in self.modules():
            if isinstance(m, virtual_gate):
                structure.append(m.width)
        self.structure = structure
        return sum(structure), structure

    def set_vritual_gate(self, arch_vector):
        i = 0
        start = 0
        for m in self.modules():
            if isinstance(m, virtual_gate):
                end = start + self.structure[i]
                m.set_structure_value(arch_vector.squeeze()[start:end])
                start = end

                i+=1

    def get_gate_grads(self):
        all_grad = []
        for m in self.modules():
            if isinstance(m, virtual_gate):
                #print(m.weights.grad.data)
                all_grad.append(m.get_grads().clone())
        #print(all_grad[0])
        return all_grad


def test():
    net = MobileNetV2()
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

#test()