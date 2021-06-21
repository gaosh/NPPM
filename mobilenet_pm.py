from train import *
from utils import *
#from models.mobilenetv2 import MobileNetV2
from models.mobilenetv2_hyper import MobileNetV2
from models.hypernet import Simplified_Gate, PP_Net, Episodic_mem
from torch.optim.lr_scheduler import MultiStepLR
import argparse
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
#size_out, size_kernel, size_group, size_inchannel, size_outchannel = get_middle_Fsize_mobnet(net)
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
#parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--p', default=0.55, type=float)
parser.add_argument('--depth', default=56, type=int)
parser.add_argument('--dis_flag', default=False, type=bool)
parser.add_argument('--gw_flag', default=False, type=bool)
parser.add_argument('--reg_w', default=2, type=float)
parser.add_argument('--T', default=3.0, type=float)
parser.add_argument('--nf', default=3.0, type=float)
parser.add_argument('--sampling', default=True, type=str2bool)
parser.add_argument('--orth_grad', default=False, type=str2bool)
parser.add_argument('--epm_flag', default=True, type=bool)

args = parser.parse_args()
#depth = args.depth
model_name = 'mobnetv2'
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./datasets/cifar10/', train=True, download=True, transform=transform_train)

train_sampler,val_sampler = TrainVal_split(trainset, 0.1,shuffle_dataset=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, num_workers=2)
validloader = torch.utils.data.DataLoader(trainset, batch_size=128, num_workers=4,sampler=val_sampler)

testset = torchvision.datasets.CIFAR10(root='./datasets/cifar10/', train=False, download=True, transform=transform_test)

testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)


net = MobileNetV2()
net.eval()
#net.set_training_flag(False)
size_out, size_kernel, size_group, size_inchannel, size_outchannel = get_middle_Fsize_mobnet(net)
net.cuda()


stat_dict = torch.load('./checkpoint/%s-base.pth.tar' % (model_name))
net.load_state_dict(stat_dict['net'])

Epoch = 350

criterion = nn.CrossEntropyLoss()

width, structure = net.count_structure()
hyper_net = Simplified_Gate(structure=structure, T=0.4,base=args.T)
hyper_net.cuda()
params = hyper_net.parameters()
pp_net = PP_Net(structure=structure)
pp_net.cuda()

optimizer = optim.AdamW(params, lr=5e-2, weight_decay=1e-2)
optimizer_p = optim.AdamW(pp_net.parameters(), lr=1e-3, weight_decay=1e-3)

ep_mem = Episodic_mem(K=500, avg_len=2, structure=structure, )

resource_reg = Flops_constraint_mobnet(args.p, size_kernel, size_out, size_group, size_inchannel, size_outchannel,
                                       weight=args.reg_w, HN=True, structure=structure)

print(optimizer)
best_acc = 0


print('Pruning Rate: %.2f FLOPs'%(args.p))
gate_Epoch = 200

scheduler = MultiStepLR(optimizer, milestones=[int(gate_Epoch/2)], gamma=0.1)
for epoch in range(gate_Epoch):
    scheduler.step()

    train_epm(validloader, net, optimizer, optimizer_p, epoch, args, resource_constraint=resource_reg,
              hyper_net=hyper_net,
              pp_net=pp_net, epm=ep_mem, ep_bn=64, orth_grad=args.orth_grad, use_sampler=args.sampling,)
    # print(hyper_net.decay_bias)
    best_acc = valid(epoch, net, testloader, best_acc, hyper_net=hyper_net,
                     model_string='%s-pruned' % (model_name ), stage='valid_model',)

