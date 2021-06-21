from tqdm import tqdm
import torch
import os
from utils import display_structure, loss_fn_kd, loss_label_smoothing, display_factor, display_structure_hyper, LabelSmoothingLoss
import copy
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.sampler import ImbalancedAccuracySampler


def set_grad(var):
    def hook(grad):
        var.grad = grad
    return hook

def train_epm(train_loader, model, optimizer, optimizer_p, epoch, args, resource_constraint, hyper_net=None,
              pp_net=None, epm=None, ep_bn=64, orth_grad=False,use_sampler=True, loss_type = 'mae'):
    tqdm_loader = tqdm(train_loader)
    model.eval()
    pp_net.train()
    hyper_net.train()

    train_loss = 0
    resource_loss = 0
    hyper_loss = 0
    c_loss_total = 0

    correct = 0
    total = 0
    inner_bn = ep_bn

    criterion = torch.nn.CrossEntropyLoss()

    if len(epm) > inner_bn:
        if use_sampler:
            sampler = ImbalancedAccuracySampler(epm)
            shuffle = False
        else:
            sampler = None
            shuffle = True
        epm_loader = DataLoader(epm, batch_size=inner_bn, sampler=sampler, shuffle=shuffle)
        epm_flag = (True and args.epm_flag)



    else:
        epm_flag = (False and args.epm_flag)

    for batch_idx, (inputs, targets) in enumerate(tqdm_loader):

        inputs, targets = inputs.cuda(), targets.cuda()
        vector = hyper_net()
        vector.retain_grad()
        concrete_vector = hyper_net.resource_output()


        if isinstance(model, torch.nn.DataParallel):
            model.module.set_vritual_gate(vector)
        else:
            model.set_vritual_gate(vector)

        outputs = model(inputs)

        c_loss = criterion(outputs, targets)
        res_loss = 2 * resource_constraint(concrete_vector)

        if epm_flag:
            # epm_vector = vector
            vector = vector + args.nf * torch.randn(vector.size()).cuda()
            vector = torch.clamp(vector, min=0, max=1)

            pred = pp_net(vector)
            p_loss = torch.log(1 / pred)
            if not orth_grad:
                h_loss = res_loss + c_loss + p_loss

        else:
            h_loss = res_loss + c_loss

        optimizer.zero_grad()

        if epm_flag:
            if orth_grad:
                c_loss.backward(retain_graph=True)
                l_grads = hyper_net.get_grads()
                p_loss.backward()
                p_grads = hyper_net.get_grads()
                res_loss.backward()
                r_grads = hyper_net.get_grads()

                hyper_net.set_orth_grads(l_grads, p_grads, r_grads)

                #only for recording
                with torch.no_grad():
                    h_loss = res_loss + c_loss + p_loss
            else:
                h_loss.backward()
        else:
            h_loss.backward()
        optimizer.step()

        if epm_flag:
            vectors, accs = next(iter(epm_loader))
            vectors, accs = vectors.cuda(), accs.cuda()
            pred_p = pp_net(vectors).squeeze()

            if loss_type == 'mae':
                loss = F.l1_loss(pred_p, accs.float())
            elif loss_type == 'mse':
                loss = F.mse_loss(pred_p, accs.float())

            optimizer_p.zero_grad()
            loss.backward()
            optimizer_p.step()
            with torch.no_grad():

                record_loss = F.l1_loss(pred_p, accs.float())

        else:
            record_loss = torch.Tensor([0])

        with torch.no_grad():
            _, predicted = outputs.detach().max(1)
            local_correct = predicted.eq(targets).sum()
            local_acc = local_correct.float() / float(targets.size(0))

            epm.insert_data(sub_arch=concrete_vector.detach(), local_acc=local_acc.detach())



        total += targets.size(0)
        train_loss += record_loss.item()
        resource_loss += res_loss.item()
        hyper_loss += h_loss.item()
        correct += local_correct.item()
        c_loss_total += c_loss.item()
    # a.__class__.__name__
    with torch.no_grad():
        # resource_constraint.print_current_FLOPs(hyper_net.resource_output())
        vector = hyper_net()
        display_structure_hyper(hyper_net.transfrom_output(vector))
    print(
        ' * Epoch{epoch: d} Loss {loss:.3f} Res Loss {resloss: .3f} Hyper Loss {hyperloss: .3f} Acc@1 {top1:.3f}'
        .format(epoch=epoch, loss=train_loss / len(train_loader), resloss=resource_loss / len(train_loader),
                hyperloss=hyper_loss / len(train_loader), top1=correct/total))

def retrain(epoch, net, criterion,trainloader, optimizer, smooth=True, scheduler=None, alpha=0.5):
    #net.activate_weights()
    #net.set_training_flag(False)
    tqdm_loader = tqdm(trainloader)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    alpha = alpha

    for batch_idx, (inputs, targets) in enumerate(tqdm_loader):
        if scheduler is not None:
            scheduler.step()
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()

        outputs = net(inputs)
        if smooth:
            loss_smooth = LabelSmoothingLoss(classes=10,smoothing=0.1)(outputs, targets)
            loss_c = criterion(outputs, targets)
            loss = alpha*loss_smooth + (1-alpha)*loss_c
        else:
            loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    print('Epoch: %d Loss: %.3f | Acc: %.3f%% (%d/%d)'
              % (epoch, train_loss / len(trainloader), 100. * correct / total, correct, total))

def valid(epoch, net, testloader, best_acc, hyper_net=None, model_string=None, stage='valid_model'):
    txtdir = './txt/'
    if stage == 'valid_model':

        tqdm_loader = tqdm(testloader)
    elif stage == 'valid_gate':
        #net.foreze_weights()
        if hyper_net is None:
            net.set_training_flag(True)
        tqdm_loader = testloader
    criterion = torch.nn.CrossEntropyLoss()

    net.eval()
    if hyper_net is not None:
        hyper_net.eval()
        vector = hyper_net()
        # print(vector)
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            if hyper_net is not None:
                net.set_vritual_gate(vector)
                outputs = net(inputs)
                loss = criterion(outputs, targets)
            else:
                outputs = net(inputs)
                loss = criterion(outputs, targets)
            test_loss += loss.item()

            _, predicted = outputs.max(1)


            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100.*correct/total
    is_best=False
    if hyper_net is not None:
        if epoch>100:
            if acc > best_acc:
                best_acc = acc
                is_best=True
        else:
            best_acc = 0
    else:
        if acc>best_acc:
            best_acc = acc
            is_best = True
    if model_string is not None:

        if is_best:
            print('Saving..')
            if hyper_net is not None:

                state = {
                    'net': net.state_dict(),
                    'hyper_net': hyper_net.state_dict(),
                    'acc': acc,
                    'epoch': epoch,
                    'arch_vector':vector
                    #'gpinfo':gplist,
                }
            else:
                state = {
                    'net': net.state_dict(),
                    'acc': acc,
                    'epoch': epoch,
                    # 'gpinfo':gplist,
                }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')

            torch.save(state, './checkpoint/%s.pth.tar'%(model_string))

    print( 'Loss: %.3f | Acc: %.3f%% (%d/%d) | Best Acc: %.3f%%'
                    % (test_loss/len(testloader), 100.*correct/total, correct, total, best_acc))

    return best_acc


