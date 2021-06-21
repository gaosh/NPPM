from __future__ import absolute_import
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import truncnorm
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
from torch.nn.utils import weight_norm
from torch.utils.data import Dataset


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    #U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, T, offset=0):
    # if inference:
    #     y = logits + offset + 0.57721
    #     return F.sigmoid(y / T)

    gumbel_sample = sample_gumbel(logits.size())
    if logits.is_cuda:
        gumbel_sample = gumbel_sample.cuda()

    y = logits + gumbel_sample + offset
    #y = logits + offset
    return F.sigmoid(y / T)

def hard_concrete(out):
    out_hard = torch.zeros(out.size())
    out_hard[out>=0.5]=1
    out_hard[out<0.5] = 0
    if out.is_cuda:
        out_hard = out_hard.cuda()
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    out_hard = (out_hard - out).detach() + out
    return out_hard


class custom_grad_weight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, grad_w=1):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        #train = train
        ctx.grad_w = grad_w

        input_clone = input.clone()
        return input_clone.float()

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        grad_input = grad_output.clone()

        gw = ctx.grad_w
        # print(gw)
        if grad_input.is_cuda and type(gw) is not int:
            gw = gw.cuda()

        return grad_input * gw, None, None



class Simplified_Gate(nn.Module):
    def __init__(self, structure=None, T=0.4, base = 3, ):
        super(Simplified_Gate, self).__init__()
        self.structure = structure
        self.T = T
        self.base = base

        self.p_list = nn.ModuleList([simple_gate(structure[i]) for i in range(len(structure))])
    def forward(self,):

        if self.training:
            outputs = [gumbel_softmax_sample(self.p_list[i](), T=self.T, offset=self.base) for i in range(len(self.structure))]
        else:
            outputs = [hard_concrete(gumbel_softmax_sample(self.p_list[i](), T=self.T, offset=self.base)) for i in range(len(self.structure))]

        out = torch.cat(outputs, dim=0)
        # print(out.size())
        return out

    def resource_output(self):
        outputs = [gumbel_softmax_sample(self.p_list[i](), T=self.T, offset=self.base) for i in
                   range(len(self.structure))]

        outputs = [hard_concrete(outputs[i]) for i in
                   range(len(self.structure))]

        out = torch.cat(outputs, dim=0)

        return out

    def transfrom_output(self, inputs):
        arch_vector = []
        start = 0
        for i in range(len(self.structure)):
            end = start + self.structure[i]
            arch_vector.append(inputs[start:end])
            start = end

        return arch_vector

    def set_orth_grads(self, l_grads, p_grads, r_grads):
        grads = []
        for i in range(len(self.p_list)):
            proj_p = (l_grads[i]*p_grads[i]).sum()/(l_grads[i].norm().pow(2))*l_grads[i]
            orth_p = p_grads[i] - proj_p
            modifed_grad = orth_p + r_grads[i] + l_grads[i]
            grads.append(modifed_grad)
        for i in range(len(self.p_list)):
            self.p_list[i].weight.grad = grads[i]

    def get_grads(self, to_zero=True):

        grads = []
        for i in range(len(self.p_list)):
            grads.append(self.p_list[i].weight.grad.clone())
            # print(grads[-1].mean())
            if to_zero:
                self.p_list[i].weight.grad.data.zero_()
        return grads

class HyperStructure(nn.Module):
    def __init__(self, structure=None, T=0.4, sparsity=0, base=2, wn_flag=True):
        super(HyperStructure, self).__init__()
        #self.fc1 = nn.Linear(64, 256, bias=False)
        self.bn1 = nn.LayerNorm([256])

        self.T = T
        #self.inputs = 0.5*torch.ones(1,64)
        self.structure = structure

        self.Bi_GRU = nn.GRU(64, 128,bidirectional=True)

        self.h0 = torch.zeros(2,1,128)
        self.inputs = nn.Parameter(torch.Tensor(len(structure),1, 64))
        nn.init.orthogonal_(self.inputs)
        self.inputs.requires_grad=False
        print(structure)
        self.sparsity = [sparsity]*len(structure)

        print(self.sparsity)
        if wn_flag:
            self.linear_list = [weight_norm(nn.Linear(64, structure[i], bias=False)) for i in range(len(structure))]
        else:
            self.linear_list = [nn.Linear(64, structure[i], bias=False,) for i
                                in range(len(structure))]

        self.mh_fc = torch.nn.ModuleList(self.linear_list)
        #decay
        self.base = base

        self.iteration = 0

        self.gw = torch.ones(len(structure))
    def forward(self):
        self.iteration+=1
        if self.bn1.weight.is_cuda:
            self.inputs = self.inputs.cuda()
            self.h0 = self.h0.cuda()
        outputs, hn = self.Bi_GRU(self.inputs, self.h0)
        outputs = [F.relu(self.bn1(outputs[i,:])) for i in  range(len(self.structure))]

        #out = self.fc2(out)
        if self.bn1.weight.is_cuda:
            self.gw = self.gw.cuda()

        outputs = [self.mh_fc[i](outputs[i], self.gw[i]) for i in range(len(self.mh_fc))]

        out = torch.cat(outputs, dim=1)
        #print(out.size())
        out = gumbel_softmax_sample(out, T=self.T, offset=self.base)
        #out = F.sigmoid(out / self.T)
        if not self.training or self.hard_flag:
            out = hard_concrete(out)
        # if self.training:
        #     self.update_bias()

        return out.squeeze()

    def transfrom_output(self, inputs):
        arch_vector = []
        start = 0
        for i in range(len(self.structure)):
            end = start + self.structure[i]
            arch_vector.append(inputs[start:end])
            start = end

        return arch_vector

    def resource_output(self):

        if self.bn1.weight.is_cuda:
            self.inputs = self.inputs.cuda()
            self.h0 = self.h0.cuda()
        outputs, hn = self.Bi_GRU(self.inputs, self.h0)
        outputs = [F.relu(self.bn1(outputs[i, :])) for i in range(len(self.structure))]

        # out = self.fc2(out)
        if self.bn1.weight.is_cuda:
            self.gw = self.gw.cuda()

        outputs = [self.mh_fc[i](outputs[i]) for i in range(len(self.mh_fc))]

        out = torch.cat(outputs, dim=1)
        out = gumbel_softmax_sample(out, T=self.T, offset=self.base)

        out = hard_concrete(out)
        return out.squeeze()

    def collect_gw(self):

        return self.gw.clone()

    def set_gw(self, all_gw):
        self.gw.copy_(all_gw)

    def get_grads(self):
        grads = []

        for i in range(len(self.mh_fc)):
            grads.append(self.mh_fc[i].weight_v.grad.data.abs()*self.mh_fc[i].mask)

        return grads

    def get_lists(self):
        size_list = []
        for i in range(len(self.linear_list)):
            size_list.append(self.linear_list[i].weight_v.size())

        return size_list

    def get_weight_g(self):
        weight_g = self.mh_fc[-1].weight_g.detach().mean()
        print('weight_g: %.4f'%(weight_g))


class simple_gate(nn.Module):
    def __init__(self, width):
        super(simple_gate, self).__init__()
        self.weight = nn.Parameter(torch.randn(width))

    def forward(self):
        return self.weight


class PP_Net(nn.Module):
    def __init__(self, structure=None, wn_flag=True):
        super(PP_Net, self).__init__()

        self.bn1 = nn.LayerNorm([128])
        self.structure = structure

        self.Bi_GRU = nn.GRU(128, 64,bidirectional=True)

        self.h0 = torch.zeros(2,1,64)
        print(structure)

        if wn_flag:
            self.linear_list = [weight_norm(nn.Linear(structure[i], 128, bias=False,)) for i in range(len(structure))]
        else:
            self.linear_list = [nn.Linear(structure[i], 128, bias=False) for i
                                in range(len(structure))]
        self.mh_fc = torch.nn.ModuleList(self.linear_list)
        self.pp = nn.Linear(128,1, bias=False)

    def forward(self, x):
        outputs = self.transfrom_output(x)
        bn = outputs[0].size(0)
        outputs = [self.mh_fc[i](outputs[i]) for i in range(len(self.mh_fc))]
        outputs = [F.relu(self.bn1(outputs[i])) for i in range(len(self.structure))]
        # print(outputs.size())
        if x.is_cuda:
            self.h0 = self.h0.cuda()

        if len(list(x.size())) == 1:
            outputs = torch.cat(outputs, dim=0).unsqueeze(1)
            h0 = self.h0.clone()
        else:
            outputs = [outputs[i].unsqueeze(0) for i in range(len(outputs))]
            outputs = torch.cat(outputs, dim=0)
            h0 = self.h0.clone().repeat(1,x.size(0),1)


        outputs, hn = self.Bi_GRU(outputs, h0)
        hn = hn.permute(1,0,2)

        hn = self.pp(hn.contiguous().view(bn,-1))

        return torch.sigmoid(hn)

    def transfrom_output(self, inputs):
        arch_vector = []
        start = 0
        for i in range(len(self.structure)):
            end = start + self.structure[i]
            if len(list(inputs.size())) == 1:
                arch_vector.append(inputs[start:end].unsqueeze(0))
                start = end
            else:
                arch_vector.append(inputs[:,start:end])
                start = end
        # print(arch_vector[0].size())
        return arch_vector

class Simple_PN(nn.Module):
    def __init__(self, structure=None, seperate=False, sparsity=0.4):
        super(Simple_PN, self).__init__()


        self.sparsity = [sparsity] * len(structure)
        self.seperate = seperate



        if not seperate:
            input_size = sum(structure)
            self.pp = nn.Linear(input_size, 1, bias=False)
        else:
            self.linear_list = [nn.Linear(structure[i], 128, bias=False) for i in range(len(structure))]
            self.mh_fc = torch.nn.ModuleList(self.linear_list)
            self.pp = nn.Linear(128, 1, bias=False)

    def forward(self, x):

        if self.seperate:
            x = self.transfrom_output(x)
            x = [self.mh_fc[i](x[i]) for i in range(len(self.mh_fc))]
            x = [F.relu(self.bn1(x[i])) for i in range(len(self.structure))]
            x = sum(x) / len(x)

        hn = self.pp(x)

        return torch.sigmoid(hn)

    def transfrom_output(self, inputs):
        arch_vector = []
        start = 0
        for i in range(len(self.structure)):
            end = start + self.structure[i]
            if len(list(inputs.size())) == 1:
                arch_vector.append(inputs[start:end].unsqueeze(0))
                start = end
            else:
                arch_vector.append(inputs[:,start:end])
                start = end
        # print(arch_vector[0].size())
        return arch_vector



class Episodic_mem(Dataset):
    def __init__(self, K=1000, avg_len = 5 , structure=None, T=0.4):
        self.K = K
        self.structure = structure
        self.avg_len = avg_len
        self.itr = 0

        len = sum(structure)
        self.k = 0
        self.T = T

        self.sub_arch = torch.zeros(K, len)
        self.acc_list = torch.zeros(K)

        self.local_arch = torch.zeros(len)
        self.local_acc = torch.zeros(1)

    def __getitem__(self, idx):
        current_arch = self.sub_arch[idx]
        current_acc = self.acc_list[idx]

        return current_arch, current_acc

    def insert_data(self, sub_arch, local_acc):

        if self.itr >= self.avg_len:

            self.verify_insert()

            self.itr = 0

            self.local_arch = sub_arch.cpu().data
            self.local_acc = local_acc.cpu().data

            self.itr += 1
        else:
            self.local_arch += sub_arch.cpu().data.clone()
            self.local_acc += local_acc.cpu().data.clone()

            self.itr += 1

    def verify_insert(self):
        if self.k < self.K:
            # print(self.local_arch[0:100])
            current_arch = self.local_arch/self.avg_len
            # print(current_arch[0:100])

            self.sub_arch[self.k, :] = current_arch


            self.acc_list[self.k] = self.local_acc/self.avg_len
            # print(self.local_acc/self.avg_len)
            # print(self.acc_list[self.k])
            self.k+=1
        elif self.k == self.K:
            acc = self.local_acc/self.avg_len
            current_arch =  self.local_arch/self.avg_len

            diff = (acc.unsqueeze(0).expand_as(self.acc_list) - self.acc_list).abs()
            values, index = diff.topk(1, largest=False)

            current_arch = (self.sub_arch[index, :] + current_arch)/2
            self.sub_arch[index, :] = current_arch

            self.acc_list[index] = (self.acc_list[index] + acc)/2
            self.k = self.K

    def __len__(self):
        return self.k




if __name__ == '__main__':
    net = HyperStructure()
    y = net()
    print(y)
