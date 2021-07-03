from train import *
from utils import *
#from models.resnet_gate import ResNet as ResNet_gate
from models.resnet_hyper import ResNet as ResNet_hyper
from models.hypernet import Simplified_Gate, PP_Net, Episodic_mem, Simple_PN
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

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
#parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--stage', default='train-gate', type=str)
parser.add_argument('--p', default=0.5, type=float)
parser.add_argument('--depth', default=56, type=int)

parser.add_argument('--epoch', default=200, type=int)
parser.add_argument('--reg_w', default=2, type=float)
parser.add_argument('--base', default=3.0, type=float)
parser.add_argument('--nf', default=1.0, type=float)
parser.add_argument('--epm_flag', default=False, type=bool)
parser.add_argument('--loss', default='log', type=str)
parser.add_argument('--pn_type', default='pn', type=str)
parser.add_argument('--sampling', default=True, type=str2bool)
parser.add_argument('--orth_grad',  default=True, type=str2bool)
parser.add_argument('--pn_loss', default='mae', type=str)
args = parser.parse_args()
depth = args.depth
model_name = 'resnet'
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
train_sampler,val_sampler = TrainVal_split(trainset, 0.1, shuffle_dataset=True)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, num_workers=2,shuffle=True)
validloader = torch.utils.data.DataLoader(trainset, batch_size=256, num_workers=4,sampler=val_sampler)

testset = torchvision.datasets.CIFAR10(root='./datasets/cifar10/', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

net = ResNet_hyper(depth=depth, gate_flag=True)
width, structure = net.count_structure()
hyper_net = Simplified_Gate(structure=structure, T=0.4, base=args.base)
if args.pn_type == 'pn':
    pp_net = PP_Net(structure=structure)
elif args.pn_type == 'sn':
    pp_net = Simple_PN(structure=structure)
elif args.pn_type == 'sn_sep':
    pp_net = Simple_PN(structure=structure, seperate=True)

stat_dict = torch.load('./checkpoint/%s-base.pth.tar'%(model_name+str(depth)))
net.load_state_dict(stat_dict['net'])
net.foreze_weights()

loss_type = args.loss

size_out, size_kernel, size_group, size_inchannel, size_outchannel = get_middle_Fsize_resnet(net)
resource_reg = Flops_constraint_resnet(args.p, size_kernel, size_out, size_group, size_inchannel, size_outchannel,
                                       w=args.reg_w, HN=True,structure=structure, loss_type=loss_type)

Epoch = args.epoch

hyper_net.cuda()
pp_net.cuda()
net.cuda()


optimizer_p = optim.AdamW(pp_net.parameters(), lr=1e-3, weight_decay=1e-3)
optimizer = optim.AdamW(hyper_net.parameters(), lr=5e-2, weight_decay=1e-2)
ep_mem = Episodic_mem(K=500, avg_len=2, structure=structure, )

scheduler = MultiStepLR(optimizer, milestones=[int(Epoch*0.8)], gamma=0.1)
best_acc = 0

valid(0, net, testloader, best_acc, hyper_net=None, model_string=None, stage='valid_model',)

for epoch in range(Epoch):

    train_epm(validloader, net, optimizer, optimizer_p, epoch, args, resource_constraint=resource_reg, hyper_net=hyper_net,
              pp_net=pp_net, epm=ep_mem, ep_bn=64, orth_grad=args.orth_grad,use_sampler=args.sampling, loss_type=args.pn_loss)
    scheduler.step()
    best_acc = valid(epoch, net, testloader, best_acc, hyper_net=hyper_net, model_string='%s-pruned'%(model_name+str(depth)), stage='valid_model',)

