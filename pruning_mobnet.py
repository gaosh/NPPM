import argparse
import numpy as np
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from utils import *
from models.mobilenetv2_hyper import MobileNetV2
#from models.gate_function import soft_gate
from models.gate_function import virtual_gate
from models.hypernet import Simplified_Gate
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar10)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--depth', type=int, default=56,
                    help='depth of the vgg')
parser.add_argument('--model', default='', type=str, metavar='PATH',
                    help='path to the model (default: none)')
parser.add_argument('--save', default='.', type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')




dir = '/datasets/cifar10/'
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

depth = args.depth

if not os.path.exists(args.save):
    os.makedirs(args.save)

if not os.path.exists(args.save):
    os.makedirs(args.save)

model = MobileNetV2()
if args.cuda:
    model.cuda()
model_name = 'mobnetv2'
stat_dict = torch.load('./checkpoint/%s-pruned.pth.tar'%(model_name))


model.load_state_dict(stat_dict['net'])
model.cuda()

width, structure = model.count_structure()
hyper_net = Simplified_Gate(structure=structure, T=0.4, base=3.0,)
hyper_net.cuda()
hyper_net.load_state_dict(stat_dict['hyper_net'])

hyper_net.eval()

with torch.no_grad():
    vector = hyper_net()
    vector[vector>0.5]=1
    vector[vector<=0.5]=0
    parameters = hyper_net.transfrom_output(vector.detach())


parameters = parameters


cfg = [(1,  16, 1, 1, []),
       (6,  24, 2, 1, []),  # NOTE: change stride 2 -> 1 for CIFAR10
       (6,  32, 3, 1, []),
       (6,  64, 4, 2, []),
       (6,  96, 3, 1, []),
       (6, 160, 3, 2, []),
       (6, 320, 1, 1, [])]
cfg_idx = 0
diff_length = [1,3,6,10,13,16,17]
for i in range(len(parameters)):
    if i >= diff_length[cfg_idx]:
        cfg_idx+=1
    cfg[cfg_idx][-1].append(int(parameters[i].sum().item()))
print(cfg)
newmodel = MobileNetV2(custom_cfg=cfg)
newmodel.cuda()
#print(newmodel)
#layer_id_in_cfg = 0
old_modules = list(model.modules())
new_modules = list(newmodel.modules())
#start_mask = torch.ones(3)
start_mask = torch.ones(3)
soft_gate_count = 0
conv_count =0
end_mask = parameters[soft_gate_count]

# modules[layer_id - 2].register_forward_hook(conv_hook)
# modules[layer_id + 1].register_forward_hook(conv_hook)
# modules[layer_id + 3].register_forward_hook(conv_hook)


for layer_id in range(len(old_modules)):
    m0 = old_modules[layer_id]
    m1 = new_modules[layer_id]
    if isinstance(m0, nn.BatchNorm2d):
        # print(m0)
        # print(m1)
        idx1 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        if idx1.size == 1:
            idx1 = np.resize(idx1,(1,))
        if layer_id==2:
            #print(m0)
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()
            m1.running_mean = m0.running_mean.clone()
            m1.running_var = m0.running_var.clone()
            #print(layer_id)
            continue
        elif isinstance(old_modules[layer_id + 1], virtual_gate):
            # If the next layer is the channel selection layer, then the current batchnorm 2d layer won't be pruned.

            #print(m0)
            m1.weight.data = m0.weight.data[idx1.tolist()].clone()
            m1.bias.data = m0.bias.data[idx1.tolist()].clone()
            m1.running_mean = m0.running_mean[idx1.tolist()].clone()
            m1.running_var = m0.running_var[idx1.tolist()].clone()
            # We need to set the channel selection layer.
            # m2 = new_modules[layer_id + 2]
            # m2.indexes.data.zero_()
            # m2.indexes.data[:] = 1.0
        elif isinstance(old_modules[layer_id - 2], virtual_gate):
            # If the next layer is the channel selection layer, then the current batchnorm 2d layer won't be pruned.

            print(m0)
            m1.weight.data = m0.weight.data[idx1.tolist()].clone()
            m1.bias.data = m0.bias.data[idx1.tolist()].clone()
            m1.running_mean = m0.running_mean[idx1.tolist()].clone()
            m1.running_var = m0.running_var[idx1.tolist()].clone()


        else:
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()
            m1.running_mean = m0.running_mean.clone()
            m1.running_var = m0.running_var.clone()

    elif isinstance(m0, nn.Conv2d):
        #print(old_modules[layer_id+2])
        if conv_count == 0:
            m1.weight.data = m0.weight.data.clone()
            conv_count += 1
            continue

        if isinstance(old_modules[layer_id+2], virtual_gate):
            print(conv_count)
            conv_count += 1
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            #print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
            print('Shape:{:d}'.format(idx1.size))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))

            print(m0)
            print(m1)

            w1 = m0.weight.data[idx1.tolist(), :, :, :].clone()
            m1.weight.data = w1.clone()
            print(m1.weight.data.size())

            m0_next = old_modules[layer_id+3]
            m1_next = new_modules[layer_id+3]
            print(m0_next)
            print(m1_next)
            if isinstance(m0_next, nn.Conv2d):

                w1 = m0_next.weight.data[idx1.tolist(), :, :, :].clone()
                m1_next.weight.data = w1.clone()
                print(m1_next.weight.data.size())

            m0_next = old_modules[layer_id + 5]
            m1_next = new_modules[layer_id + 5]
            print(m0_next)
            print(m1_next)
            if isinstance(m0_next, nn.Conv2d):
                print(idx1.size)
                w1 = m0_next.weight.data[:, idx1.tolist(), :, :].clone()
                #m1_next.weight.data = w1.clone()
                m1_next.weight.data.copy_(w1)
                print(m1_next.weight.data.size())

            soft_gate_count += 1
            start_mask = end_mask.clone()
            if soft_gate_count < len(parameters):
                end_mask = parameters[soft_gate_count]
                print(end_mask.size())
            continue
        if isinstance(old_modules[layer_id -1], virtual_gate):
            continue
        if isinstance(old_modules[layer_id -3], virtual_gate):
            continue
        # We need to consider the case where there are downsampling convolutions.
        # For these convolutions, we just copy the weights.

        m1.weight.data = m0.weight.data.clone()
        #print(m1)

    elif isinstance(m0, nn.Linear):
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        if idx0.size == 1:
            idx0 = np.resize(idx0, (1,))

        #m1.weight.data = m0.weight.data[:, idx0].clone()
        m1.weight.data = m0.weight.data.clone()
        m1.bias.data = m0.bias.data.clone()



model.cpu()
newmodel.cpu()
#print(newmodel)
t_o=print_model_param_nums(model)
t_n=print_model_param_nums(newmodel)
print_model_param_flops(model, input_res=32)
print_model_param_flops(newmodel, input_res=32)

all_parameters = torch.cat(parameters)
print(all_parameters)
pruning_rate = float((all_parameters==1).sum())/float(all_parameters.size(0))
print(pruning_rate)



torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, os.path.join(args.save, './checkpoint/%s_new.pth.tar'%(model_name)))