import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from models.gate_function import custom_STE
from models.gate_function import soft_gate, virtual_gate
import torchvision
from math import sqrt, floor

class resource_constraint(nn.Module):
    def __init__(self, num_epoch, cut_off_epoch, p):
        super(resource_constraint, self).__init__()
        self.num_epoch = num_epoch
        self.cut_off_epoch = cut_off_epoch
        self.p = p

    def forward(self, input, epoch):
        overall_length = 0
        for i in range(len(input)):
            overall_length+= input[i].size(0)

        for i in range(len(input)):
            if i == 0:
                #cat_tensor = input[i]
                #cat_tensor = F.tanh(input[i].abs().pow(1 / 2))
                cat_tensor = custom_STE.apply(input[i], False)
            else:
                #current_value = F.tanh(input[i].abs().pow(1 / 2))
                current_value = custom_STE.apply(input[i], False)
                cat_tensor = torch.cat([cat_tensor, current_value])

        # if epoch<= self.cut_off_epoch:
        #     w = (epoch/self.cut_off_epoch)
        # else:
        #     w = 1
        #loss = w*torch.log(F.relu(cat_tensor.mean() - self.p) + 1)
        #loss = torch.abs(cat_tensor.sum()-int(self.p*cat_tensor.size(0)))
        loss = torch.abs(cat_tensor.mean() - (self.p))
        #return (1/cat_tensor.size(0))*loss
        return loss

class Flops_constraint(nn.Module):
    def __init__(self, p, kernel_size, out_size, group_size, size_inchannel, size_outchannel, in_channel=3):
        super(Flops_constraint, self).__init__()

        self.p = p
        self.k_size = kernel_size
        self.out_size = out_size
        self.g_size = group_size
        self.in_csize = size_inchannel
        self.out_csize = size_outchannel
        self.t_flops = self.init_total_flops()
        self.inc_1st = in_channel
    def init_total_flops(self):
        total_flops = 0
        for i in range(len(self.k_size)):
            total_flops+= self.k_size[i]*(self.in_csize[i]/self.g_size[i])*self.out_csize[i]*self.out_size[i]+3*self.out_csize[i]*self.out_size[i]

        print('+ Number of FLOPs: %.5fG'%(total_flops/1e9))
        return total_flops


    def forward(self, input):
        c_in = self.inc_1st
        sum_flops = 0
        #print(len(self.k_size))
        for i in range(len(input)):
            #print(i)

            current_tensor = custom_STE.apply(input[i], False)
            if i >0:
                c_in = custom_STE.apply(input[i-1], False).sum()
            c_out = current_tensor.sum()
            #sum_flops+=current_tensor.sum()
            sum_flops+= self.k_size[i]*(c_in/self.g_size[i])*c_out*self.out_size[i]+3*c_out*self.out_size[i]
        loss = torch.log(torch.cosh(sum_flops/self.t_flops - (self.p))+ 1)
        #loss = torch.abs(sum_flops/self.t_flops - (self.p))**2
        return 2*loss

class Flops_constraint_resnet(nn.Module):
    def __init__(self, p, kernel_size, out_size, group_size, size_inchannel, size_outchannel, in_channel=3, w=2, HN=False, structure=None, loss_type='log'):
        super(Flops_constraint_resnet, self).__init__()

        self.p = p
        self.k_size = kernel_size
        self.out_size = out_size
        self.g_size = group_size
        self.in_csize = size_inchannel
        self.out_csize = size_outchannel
        self.t_flops = self.init_total_flops()
        self.inc_1st = in_channel
        self.weight = w
        self.HN = HN
        self.structure = structure
        self.loss_type = loss_type

    def init_total_flops(self):
        total_flops = 0
        for i in range(len(self.k_size)):
            total_flops+= self.k_size[i]*(self.in_csize[i]/self.g_size[i])*self.out_csize[i]*self.out_size[i]+3*self.out_csize[i]*self.out_size[i]

        print('+ Number of FLOPs: %.5fG'%(total_flops/1e9))
        return total_flops

    def forward(self, input):
        self.input = input
        if self.HN:
            #self.h = self.input.register_hook(lambda grad: grad)
            arch_vector = []
            start = 0
            for i in range(len(self.structure)):
                end = start + self.structure[i]
                input = input.squeeze()
                #print(input[start:end].size())
                arch_vector.append(input[start:end])
                start = end
        #print(len(arch_vector))
        c_in = self.inc_1st
        sum_flops = 0
        #print(len(self.k_size))
        #start = 0
        #print(len(arch_vector))
        #print(arch_vector)
        if self.HN:
            length = len(arch_vector)
        else:
            length = len(input)
        for i in range(length):
            #print(i)
            if self.HN is False:
                current_tensor = custom_STE.apply(input[i], False)
            # if i >0:
            #     c_in = custom_STE.apply(input[i-1], False).sum()
            else:
                current_tensor = arch_vector[i]
            c_out = current_tensor.sum()

            #two layer as a group
            sum_flops+= self.k_size[2*i]*(self.in_csize[2*i]/self.g_size[2*i])*c_out*self.out_size[2*i]+3*c_out*self.out_size[2*i]
            sum_flops+= self.k_size[2*i+1]*(c_out/self.g_size[2*i+1])*self.out_csize[2*i+1]*self.out_size[2*i+1]+3*self.out_csize[2*i+1]*self.out_size[2*i+1]
        #loss = torch.log(truncate_L1((sum_flops/self.t_flops - (self.p)))+1)
        #if sum_flops/self.t_flops - (self.p)>0:
        resource_value = sum_flops / self.t_flops - (self.p)
        #abs_rv = torch.clamp(torch.abs(resource_value), min=0.0000)
        if self.loss_type == 'log':

            # abs_rv = torch.abs(resource_value)
            # loss =  torch.log(abs_rv+ 1)

            resource_ratio = (sum_flops / self.t_flops)
            abs_rv = torch.clamp(resource_ratio, min=self.p)
            loss = torch.log((abs_rv / (self.p)))

        elif self.loss_type == 'mae':
            loss = torch.abs(resource_value)
        elif self.loss_type == 'mse':
            loss = torch.abs(resource_value).pow(2)


        return self.weight*loss

    def print_current_FLOPs(self, input):
        sum_flops = 0
        #print(len(self.k_size))
        if self.HN:
            #self.h = self.input.register_hook(lambda grad: grad)
            arch_vector = []
            start = 0
            for i in range(len(self.structure)):
                end = start + self.structure[i]
                input = input.squeeze()
                # print(input[start:end].size())
                arch_vector.append(input[start:end])
                start = end

            length = len(arch_vector)
        else:
            length = len(input)
        for i in range(length):
            #print(i)
            if self.HN:
                c_out = arch_vector[i].sum()
            else:
                c_out = input[i].sum()
            #two layer as a group
            sum_flops += self.k_size[2 * i] * (self.in_csize[2 * i] / self.g_size[2 * i]) * c_out * self.out_size[
                2 * i] + 3 * c_out * self.out_size[2 * i]

            sum_flops += self.k_size[2 * i + 1] * (c_out / self.g_size[2 * i + 1]) * self.out_csize[2 * i + 1] * \
                         self.out_size[2 * i + 1] + 3 * self.out_csize[2 * i + 1] * self.out_size[2 * i + 1]
        print('+ Current FLOPs: %.5fG'%(sum_flops/1e9))
    #


class Flops_constraint_mobnet(nn.Module):
    def __init__(self, p, kernel_size, out_size, group_size, size_inchannel, size_outchannel, in_channel=3, weight=2, HN=False, structure=None):
        super(Flops_constraint_mobnet, self).__init__()

        self.p = p
        self.k_size = kernel_size
        self.out_size = out_size
        self.g_size = group_size
        self.in_csize = size_inchannel
        self.out_csize = size_outchannel

        self.weight = weight
        self.inc_1st = in_channel
        self.HN = HN
        self.structure = structure
        self.t_flops = self.init_total_flops()
    def init_total_flops(self):
        total_flops = 0
        detail_flops = []


        for i in range(len(self.k_size)):
            current_flops = self.k_size[i]*(self.in_csize[i]/self.g_size[i])*self.out_csize[i]*self.out_size[i]+3*self.out_csize[i]*self.out_size[i]
            total_flops+= current_flops
            detail_flops.append(current_flops)
        detail_flops = [f/1e9 for f in detail_flops]
        #print(detail_flops)
        print('+ Number of FLOPs: %.5fG'%(total_flops/1e9))
        self.last_flops = total_flops
        return total_flops

    def print_current_FLOPs(self, input):
        sum_flops = 0
        #print(len(self.k_size))
        if self.HN:
            #self.h = self.input.register_hook(lambda grad: grad)
            arch_vector = []
            start = 0
            for i in range(len(self.structure)):
                end = start + self.structure[i]
                input = input.squeeze()
                # print(input[start:end].size())
                arch_vector.append(input[start:end])
                start = end

            length = len(arch_vector)
            print(length)
        else:
            length = len(input)
        #print(length)
        for i in range(length):
            #print(i)
            if self.HN is False:
                current_tensor = custom_STE.apply(input[i].detach(), False)
            # if i >0:
            #     c_in = custom_STE.apply(input[i-1], False).sum()
            else:
                current_tensor = arch_vector[i].detach()
            #current_tensor = custom_STE.apply(input[i].detach().cpu(), False)
            # if i >0:
            #     c_in = custom_STE.apply(input[i-1], False).sum()
            channels = current_tensor.sum()
            #two layer as a group
            sum_flops+= self.k_size[3*i]*(self.in_csize[3*i]/self.g_size[3*i])*channels*self.out_size[3*i]+3*channels*self.out_size[3*i]

            sum_flops+= self.k_size[3*i+1]*(channels/self.g_size[3*i+1])*channels*self.out_size[3*i+1]+3*channels*self.out_size[3*i+1]

            sum_flops += self.k_size[3 * i + 2] * (channels / self.g_size[3 * i + 2]) * self.out_csize[3 * i + 2] * \
                         self.out_size[3 * i + 2] + 3 * self.out_csize[3 * i + 2] * self.out_size[3 * i + 2]
        print('+ Current FLOPs: %.5fG'%(sum_flops/1e9))

    def forward(self, input):
        #c_in = self.inc_1st
        sum_flops = 0
        if self.HN:
            #self.h = self.input.register_hook(lambda grad: grad)
            arch_vector = []
            start = 0
            for i in range(len(self.structure)):
                end = start + self.structure[i]
                input = input.squeeze()
                # print(input[start:end].size())
                arch_vector.append(input[start:end])
                start = end

            length = len(arch_vector)
            #print(length)
        else:
            length = len(input)

        #print(len(self.k_size))
        for i in range(length):
            #print(i)

            if self.HN is False:
                current_tensor = custom_STE.apply(input[i], False)
            # if i >0:
            #     c_in = custom_STE.apply(input[i-1], False).sum()
            else:
                current_tensor = arch_vector[i]
            # if i >0:
            #     c_in = custom_STE.apply(input[i-1], False).sum()
            channels = current_tensor.sum()
            #two layer as a group
            sum_flops+= self.k_size[3*i]*(self.in_csize[3*i]/self.g_size[3*i])*channels*self.out_size[3*i]+3*channels*self.out_size[3*i]

            sum_flops+= self.k_size[3*i+1]*(channels/self.g_size[3*i+1])*channels*self.out_size[3*i+1]+3*channels*self.out_size[3*i+1]

            sum_flops += self.k_size[3 * i + 2] * (channels / self.g_size[3 * i + 2]) * self.out_csize[3 * i + 2] * \
                         self.out_size[3 * i + 2] + 3 * self.out_csize[3 * i + 2] * self.out_size[3 * i + 2]
        #if self.t_flops - (self.p)>0:
        #resource_value = torch.clamp(sum_flops / self.t_flops - (self.p), min=0)
        #resource_value = sum_flops / self.t_flops - (self.p)
        # resource_value = sum_flops / self.t_flops - (self.p)
        # abs_rv = torch.clamp(torch.abs(resource_value), min=0.003)
        # loss = torch.log(abs_rv+ 1)

        resource_ratio = (sum_flops / self.t_flops)
        abs_rv = torch.clamp(resource_ratio, min=self.p)
        loss = torch.log((abs_rv / (self.p)))

        #+ torch.abs(sum_flops/self.t_flops- self.last_flops/self.t_flops)**2
        #loss = torch.abs(sum_flops/self.t_flops - (self.p))**2
        #self.last_flops = sum_flops.detach()
        return self.weight*loss


def TrainVal_split(dataset, validation_split,shuffle_dataset=True):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(0)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    return train_sampler, valid_sampler

def display_structure(all_parameters):
    num_layers = len(all_parameters)
    layer_sparsity = []
    for i in range(num_layers):

        current_parameter = all_parameters[i].cpu().data
        if i == 0:
            print(current_parameter)
        layer_sparsity.append((current_parameter>=0.5).sum().item()/current_parameter.size(0))

    print_string = ''
    for i in range(num_layers):
        print_string += 'l-%d s-%.3f '%(i+1, layer_sparsity[i])
    return_string = ''
    for i in range(num_layers):
        return_string += '%.3f '%(layer_sparsity[i])
    print(print_string)
    return return_string

def display_structure_hyper(vectors):
    num_layers = len(vectors)
    layer_sparsity = []
    for i in range(num_layers):

        current_parameter = vectors[i].cpu().data
        # if i == 0:
        #     print(current_parameter)
        layer_sparsity.append(current_parameter.sum().item()/current_parameter.size(0))
    print_string = ''
    for i in range(num_layers):
        print_string += 'l-%d s-%.3f ' % (i + 1, layer_sparsity[i])
    return_string = ''
    for i in range(num_layers):
        return_string += '%.3f ' % (layer_sparsity[i])
    print(print_string)
    return return_string


def display_factor(gw_list):
    gw_list = [x.item() for x in gw_list]
    string=''
    for gw in gw_list:
        string += '%.3f ' % (gw)
    print(string)
def loss_fn_kd(outputs, labels, teacher_outputs, T, alpha):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    # T = params[0]
    # alpha = params[1]
    #beta = params[2]

    labels.requires_grad = False
    #teacher_outputs.detach()
    labels_onehot = torch.FloatTensor(labels.size(0), outputs.size(1))

    # In your for loop
    labels_onehot.zero_()
    labels_onehot.scatter_(1, labels.unsqueeze(-1).cpu(), 1)
    if outputs.is_cuda:
        labels_onehot = labels_onehot.cuda()

    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs.detach()/T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(outputs, labels) * (1. - alpha)

    #F.mse_loss(F.softmax(outputs), labels_onehot.float()) * (1. - alpha)



    #Teacher_loss = F.cross_entropy(teacher_outputs, labels)
    #KD_loss = KD_loss + beta* Teacher_loss
    return KD_loss


def smooth_one_hot(true_labels: torch.Tensor, classes: int, smoothing=0.1):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method

    """
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    label_shape = torch.Size((true_labels.size(0), classes))
    with torch.no_grad():
        true_dist = torch.empty(size=label_shape, device=true_labels.device)
        true_dist.fill_(smoothing / (classes - 1))
        true_dist.scatter_(1, true_labels.data.unsqueeze(1), confidence)
    return true_dist



def loss_label_smoothing(outputs, labels, T, alpha):
    uniform = torch.Tensor(outputs.size())
    uniform.fill_(1/outputs.size(1))
    if outputs.is_cuda:
        uniform = uniform.cuda()
    sm_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                             F.softmax(uniform/T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(outputs, labels) * (1. - alpha)
    return sm_loss


def print_model_param_nums(model=None, multiply_adds=True):
    if model == None:
        model = torchvision.models.alexnet()
    total = sum([param.nelement() for param in model.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))
    return total


def print_model_param_flops(model=None, input_res=224, multiply_adds=False):
    prods = {}

    def save_hook(name):
        def hook_per(self, input, output):
            prods[name] = np.prod(input[0].shape)

        return hook_per

    list_1 = []

    def simple_hook(self, input, output):
        list_1.append(np.prod(input[0].shape))

    list_2 = {}

    def simple_hook2(self, input, output):
        list_2['names'] = np.prod(input[0].shape)

    list_conv = []

    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups)
        bias_ops = 1 if self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)
        flops = (kernel_ops * (
            2 if multiply_adds else 1) + bias_ops) * output_channels * output_height * output_width * batch_size

        list_conv.append(flops)

    list_linear = []

    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1

        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement()

        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)

    list_bn = []

    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement() * 2)

    list_relu = []

    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())

    list_pooling = []

    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = 0
        flops = (kernel_ops + bias_ops) * output_channels * output_height * output_width * batch_size

        list_pooling.append(flops)

    list_upsample = []

    # For bilinear upsample
    def upsample_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        flops = output_height * output_width * output_channels * batch_size * 12
        list_upsample.append(flops)

    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            if isinstance(net, torch.nn.BatchNorm2d):
                net.register_forward_hook(bn_hook)
            if isinstance(net, torch.nn.ReLU):
                net.register_forward_hook(relu_hook)
            if isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
                net.register_forward_hook(pooling_hook)
            if isinstance(net, torch.nn.Upsample):
                net.register_forward_hook(upsample_hook)
            return
        for c in childrens:
            foo(c)

    if model == None:
        model = torchvision.models.alexnet()
    foo(model)
    input = torch.rand(3, 3, input_res, input_res)
    input.require_grad = True
    print(input.size())
    out = model(input)

    total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling) + sum(
        list_upsample))

    print('  + Number of FLOPs: %.5fG' % (total_flops / 3 / 1e9))

    return total_flops

def get_middle_Fsize(model, input_res=32):
    #size_in = []
    size_out = []
    size_kernel = []
    size_group = []
    size_inchannel = []
    size_outchannel = []
    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()
        size_out.append(output_height*output_width)
        size_kernel.append(self.kernel_size[0] * self.kernel_size[1])
        size_group.append(self.groups)
        size_inchannel.append(input_channels)
        size_outchannel.append(output_channels)
    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                net.register_forward_hook(conv_hook)
            return
        for c in childrens:
            foo(c)
    foo(model)
    input = torch.rand(2, 3, input_res, input_res)
    input.require_grad = True
    out = model(input)
    print(len(size_out))
    print(len(size_kernel))

    return size_out, size_kernel, size_group, size_inchannel, size_outchannel

def get_middle_Fsize_resnet(model, input_res=32):
    size_out = []
    size_kernel = []
    size_group = []
    size_inchannel = []
    size_outchannel = []

    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()
        size_out.append(output_height * output_width)
        size_kernel.append(self.kernel_size[0] * self.kernel_size[1])
        size_group.append(self.groups)
        size_inchannel.append(input_channels)
        size_outchannel.append(output_channels)

    def foo(net):
        modules = list(net.modules())
        #print(modules)

        for layer_id in range(len(modules)):
            m = modules[layer_id]
            #print(m)
            #if layer_id + 3 <= len(modules):
            if  isinstance(m, virtual_gate):
                #print(m)

                modules[layer_id - 3].register_forward_hook(conv_hook)
                modules[layer_id + 1 ].register_forward_hook(conv_hook)
                # print(modules[layer_id - 3])
                # print(modules[layer_id + 1])
            # elif isinstance(modules[layer_id - 1], soft_gate):
            #     print(m)
            #     m.register_forward_hook(conv_hook)

    foo(model)
    input = torch.rand(2, 3, input_res, input_res)
    input.require_grad = True
    out = model(input)
    print(len(size_out))
    print(len(size_kernel))

    return size_out, size_kernel, size_group, size_inchannel, size_outchannel

def get_middle_Fsize_mobnet(model, input_res=32):
    size_out = []
    size_kernel = []
    size_group = []
    size_inchannel = []
    size_outchannel = []

    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()
        size_out.append(output_height * output_width)
        size_kernel.append(self.kernel_size[0] * self.kernel_size[1])
        size_group.append(self.groups)
        size_inchannel.append(input_channels)
        size_outchannel.append(output_channels)

    def foo(net):
        modules = list(net.modules())
        #print(modules)

        for layer_id in range(len(modules)):
            m = modules[layer_id]
            #print(m)
            #if layer_id + 3 <= len(modules):
            if  isinstance(m, virtual_gate):
                #print(m)

                modules[layer_id - 2].register_forward_hook(conv_hook)
                modules[layer_id + 1].register_forward_hook(conv_hook)
                modules[layer_id + 3].register_forward_hook(conv_hook)
                # print(modules[layer_id - 3])
                # print(modules[layer_id + 1])
            # elif isinstance(modules[layer_id - 1], soft_gate):
            #     print(m)
            #     m.register_forward_hook(conv_hook)

    foo(model)
    input = torch.rand(2, 3, input_res, input_res)
    input.require_grad = True
    out = model(input)
    print(len(size_out))
    print(len(size_kernel))

    return size_out, size_kernel, size_group, size_inchannel, size_outchannel

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))