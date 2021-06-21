from train import *
from utils import *
from models.vgg_gate import *
from models.resnet_hyper import ResNet
#from models.mobilenetv2 import MobileNetV2
from models.mobilenetv2_hyper import MobileNetV2
from models.gate_function import *
from torch.optim.lr_scheduler import MultiStepLR

import argparse
import torchvision
import torchvision.transforms as transforms

from warm_up.Warmup_Sch import GradualWarmupScheduler

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

parser.add_argument('--model_name', default='resnet', type=str)
parser.add_argument('--depth',  default=56, type=int)

parser.add_argument('--epoch', default=200, type=int)
parser.add_argument('--m', default=0.9, type=float)
parser.add_argument('--sch', default='first-more',type=str)
parser.add_argument('--smooth_flag', default=False, type=str2bool)
parser.add_argument('--width', default=1, type=float)
parser.add_argument('--wd', default=1e-4, type=float)
parser.add_argument('--alpha', default=0.5, type=float)
parser.add_argument('--warmup', default=False, type=str2bool)
parser.add_argument('--train_base', default=True, type=str2bool)
args = parser.parse_args()


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
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, num_workers=2, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./datasets/cifar10/', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

#stat_dict = torch.load('./checkpoint/vgg_new.pt')

if args.model_name == 'resnet':
    if args.train_base:
        net = ResNet(depth=args.depth, gate_flag=True)
        model_name = '%s-base' % (args.model_name + str(args.depth))
    else:
        stat_dict = torch.load('./checkpoint/%s_new.pth.tar'%(args.model_name+str(args.depth)))
        net = ResNet(depth=args.depth, cfg=stat_dict['cfg'])
        net.load_state_dict(stat_dict['state_dict'])
        model_name = '%s-ft'%(args.model_name+str(args.depth))
elif args.model_name == 'mobnetv2':
    if args.train_base:
        net = MobileNetV2(gate_flag=True)
        model_name = '%s-base' % (args.model_name)
    else:
        stat_dict = torch.load('./checkpoint/%s_new.pth.tar'%(args.model_name))
        net = MobileNetV2(custom_cfg=stat_dict['cfg'])
        net.load_state_dict(stat_dict['state_dict'])
        model_name = '%s-ft' % (args.model_name)

budget_epoch=args.epoch
net.cuda()

params = [
        {'params': net.parameters()}
     ]

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params, lr=args.lr,
                     # momentum=0.9)
                     momentum=args.m, weight_decay=args.wd)


Epoch = args.epoch
print(Epoch)
if args.sch=='first-more':
    scheduler = MultiStepLR(optimizer, milestones=[int(0.5 * Epoch), int(0.75 *Epoch)], gamma=0.1)
elif args.sch =='even':
    scheduler = MultiStepLR(optimizer, milestones=[int(1/3 * 0.9*Epoch), int(2/3 * 0.9*Epoch), int(0.9*Epoch)], gamma=0.1)
elif args.sch == 'cos':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(Epoch), eta_min=0)

if args.warmup:
    base_sch = scheduler
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=base_sch)
    Epoch = Epoch+5
best_acc = 0
best_acc = valid(0, net, testloader, best_acc, hyper_net=None, model_string = model_name)
print(scheduler)

for epoch in range(0, Epoch):
    #scheduler.step()

    scheduler.step()
    retrain(epoch, net, criterion, trainloader, optimizer, smooth=args.smooth_flag,alpha=args.alpha)
    best_acc = valid(epoch, net, testloader, best_acc, hyper_net=None, model_string =model_name)

