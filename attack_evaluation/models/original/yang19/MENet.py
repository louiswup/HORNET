from __future__ import print_function

import argparse
import numpy as np
import os
import sys
from PIL import Image
#from cvxpy import *
#from fancyimpute import SoftImpute, BiScaler

import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import foolbox

from torchvision import datasets
from torchvision import transforms as T
from torch.utils.data import DataLoader

config = {
    'epsilon': 8.0 / 255.,
    'num_steps': 10,
    'step_size': 2.0 / 255,
    'random_start': True,
    'loss_func': 'xent',
}

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Normalization param
mean = np.array([0.4914, 0.4822, 0.4465]).reshape((3, 1, 1))
std = np.array([0.2023, 0.1994, 0.2010]).reshape((3, 1, 1))

maskp = 0.5
svdprob = 0.8
me_channel = 'concat'


class usvt(torch.autograd.Function):
    """ME-Net layer with universal singular value thresholding (USVT) approach.

    The ME preprocessing is embedded into a Function subclass for adversarial training.
    ----------
    Chatterjee, S. et al. Matrix estimation by universal singular value thresholding. 2015.
    https://pytorch.org/docs/stable/notes/extending.html
    """

    @staticmethod
    def forward(ctx, input):
        batch_num, c, h, w = input.size()
        output = torch.zeros_like(input).cpu().numpy()

        for i in range(batch_num):
            img = (input[i] * 2 - 1).cpu().numpy()

            if me_channel == 'concat':
                img = np.concatenate((np.concatenate((img[0], img[1]), axis=1), img[2]), axis=1)
                mask = np.random.binomial(1, maskp, h * w * c).reshape(h, w * c)
                p_obs = len(mask[mask == 1]) / (h * w * c)

                u, sigma, v = np.linalg.svd(img * mask)
                S = np.zeros((h, w))
                for j in range(int(svdprob * h)):
                    S[j][j] = sigma[j]
                S = np.concatenate((S, np.zeros((h, w * 2))), axis=1)
                W = np.dot(np.dot(u, S), v) / p_obs
                W[W < -1] = -1
                W[W > 1] = 1
                est_matrix = (W + 1) / 2
                for channel in range(c):
                    output[i, channel] = est_matrix[:, channel * h:(channel + 1) * h]
            else:
                mask = np.random.binomial(1, maskp, h * w).reshape(h, w)
                p_obs = len(mask[mask == 1]) / (h * w)
                for channel in range(c):
                    u, sigma, v = np.linalg.svd(img[channel] * mask)
                    S = np.zeros((h, w))
                    for j in range(int(svdprob * h)):
                        S[j][j] = sigma[j]
                    W = np.dot(np.dot(u, S), v) / p_obs
                    W[W < -1] = -1
                    W[W > 1] = 1
                    output[i, channel] = (W + 1) / 2

        output = output - mean
        output /= std
        output = torch.from_numpy(output).float().to(device)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # BPDA, approximate gradients
        return grad_output



class MENet(nn.Module):
    """ME-Net layer.

    To attack a trained ME-Net model, first load the checkpoint, then wrap the loaded model with ME layer.
    Example:
        model = checkpoint['model']
        menet_model = MENet(model)
    ----------
    https://pytorch.org/docs/stable/notes/extending.html
    """
    def __init__(self, model):
        super(MENet, self).__init__()
        self.model = model

    def forward(self, input):
        x = globals()['usvt'].apply(input)
        return self.model(x)


class AttackPGD(nn.Module):
    """White-box adversarial attacks with PGD.

    Adversarial examples are constructed using PGD under the L_inf bound.
    To attack a trained ME-Net model, first load the checkpoint, wrap with ME layer, then wrap with PGD layer.
    Example:
        model = checkpoint['model']
        menet_model = MENet(model)
        net = AttackPGD(menet_model, config)
    ----------
    Madry, A. et al. Towards deep learning models resistant to adversarial attacks. 2018.
    """
    def __init__(self, model, config):
        super(AttackPGD, self).__init__()
        self.model = model
        self.rand = config['random_start']
        self.step_size = config['step_size']
        self.epsilon = config['epsilon']
        self.num_steps = config['num_steps']
        assert config['loss_func'] == 'xent', 'Use cross-entropy as loss function.'

    def forward(self, inputs, targets):

        x = inputs.detach()
        if self.rand:
            x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        for i in range(self.num_steps):
            x.requires_grad_()
            with torch.enable_grad():
                logits = self.model(x)
                loss = F.cross_entropy(logits, targets, size_average=False)
            grad = torch.autograd.grad(loss, [x])[0]
            # print(grad)
            x = x.detach() + self.step_size * torch.sign(grad.detach())
            x = torch.min(torch.max(x, inputs - self.epsilon), inputs + self.epsilon)
            x = torch.clamp(x, 0, 1)

        return self.model(x), x


def attack_foolbox():
    fmodel = foolbox.models.PyTorchModel(menet_model, bounds=(0, 1), num_classes=10, preprocessing=(0, 1))
    attack_criteria = foolbox.criteria.Misclassification()
    attack = foolbox.attacks.ProjectedGradientDescentAttack(model=fmodel, criterion=attack_criteria)

    correct = 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.cpu().numpy()[0], int(targets.cpu().numpy())

        adversarial = attack(inputs.astype(np.float32), targets, epsilon=config['epsilon'],
                             stepsize=config['step_size'], iterations=config['num_steps'])

        if adversarial is None:
            adversarial = inputs.astype(np.float32)

        if np.argmax(fmodel.predictions(adversarial)) == targets:
            correct += 1.

        sys.stdout.write("\rWhite-box BPDA attack (toolbox)... Acc: %.3f%% (%d/%d)" %
                         (100. * correct / (batch_idx + 1), correct, batch_idx + 1))
        sys.stdout.flush()

    return 100. * correct / batch_idx


def attack_bpda():
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            with torch.no_grad():
                outputs, pert_inputs = net(inputs, targets)

            _, pred_idx = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += pred_idx.eq(targets.data).cpu().sum().float()

            sys.stdout.write("\rWhite-box BPDA attack... Acc: %.3f%% (%d/%d)"
                             % (100. * correct / total, correct, total))
            sys.stdout.flush()

    return 100. * correct / total


def test_generalization():
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.no_grad():
                outputs = menet_model(inputs)

            _, pred_idx = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += pred_idx.eq(targets.data).cpu().sum().float()

            sys.stdout.write("\rGeneralization... Acc: %.3f%% (%d/%d)"
                             % (100. * correct / total, correct, total))
            sys.stdout.flush()

    return 100. * correct / total

def eval_nadv(model, device, test_loader):
    model.eval()
    correct, total = 0, 0
    criterion = nn.CrossEntropyLoss()

    for data, labels in test_loader:
        if True:
            data, labels = data.to(device), labels.to(device)
        outputs = model(data.clone().detach())
        predictions = outputs.data.max(1)[1]
        total += labels.size(0)
        correct += (predictions == labels.data).sum()
        print("a epoch pass")



    acc = correct * 100. / total
    err = 100. - acc
    print("Accuracy (%): {}\t Error rate (%): {}".format(acc, err))

def load_MENet_model(model='yang19',dataset='cifar10',threat_model='Linf'):
    model_file = './adpure/ckpt.t7_ResNet18_advtrain_concat_usvt_0.5_white'

    checkpoint = torch.load(model_file)
    model = checkpoint['model']
    if isinstance(model,torch.nn.DataParallel):
        model = model.module
    model = model.to(device)
    menet_model = MENet(model)

    return menet_model


'''if __name__ == '__main__':


    print('=====> Loading trained model from checkpoint...')
    #assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    #checkpoint = torch.load(args.ckpt_dir + args.name + '.ckpt')
    checkpoint = torch.load('../adpure/ckpt.t7_ResNet18_advtrain_concat_usvt_0.5_white')

    model = checkpoint['model']
    rng_state = checkpoint['rng_state']
    torch.set_rng_state(rng_state)
    if isinstance(model,torch.nn.DataParallel):
        model = model.module

    model = model.to(device)
    menet_model = MENet(model)
    transform_test = T.Compose([T.ToTensor(),])
    testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                                     download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=256, pin_memory=True,
                                                      shuffle=False, num_workers=4)
    eval_nadv(menet_model, device, test_loader)
    net = AttackPGD(menet_model, config)
    net.eval()
    print('=====> White-box BPDA on trained model... Acc: %.3f%%' % attack_bpda())
    print('=====> White-box BPDA on trained model... Acc: %.3f%%' % attack_foolbox())
'''
