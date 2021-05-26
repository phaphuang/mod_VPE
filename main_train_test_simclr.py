from argparse import ArgumentParser
import os
import random
from matplotlib import pyplot as plt
import numpy as np
import math
import os
from pathlib import Path

import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms

from loader import get_data_path
from augmentations import *

from models.SimCLR.gtsrbClassifierData import gtsrbClassifierData
from models.SimCLR.nx_xnet import NT_Xent
from models.simclr import SimCLR
from models.SimCLR.resnet import get_resnet
from torch.utils.tensorboard import SummaryWriter

import json

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Setup
parser = ArgumentParser(description='Variational Prototyping Encoder (VPE)')
parser.add_argument('--seed',       type=int,   default=42,             help='Random seed')
parser.add_argument('--arch',       type=str,   default='vaeIdsiaStn',  help='network type: vaeIdsia, vaeIdsiaStn')
parser.add_argument('--dataset',    type=str,   default='gtsrb2TT100K', help='dataset to use [gtsrb, gtsrb2TT100K, belga2flickr, belga2toplogo]')
parser.add_argument('--exp',        type=str,   default='exp_list',     help='training scenario')
parser.add_argument('--resume',     type=str,   default=None,           help='Resume training from previously saved model')

parser.add_argument('--epochs',     type=int,   default=2000,           help='Training epochs')
parser.add_argument('--lr',         type=float, default=1e-4,           help='Learning rate')
parser.add_argument('--batch_size', type=int,   default=64,            help='Batch size')

parser.add_argument('--img_cols',   type=int,   default=64,             help='resized image width')
parser.add_argument('--img_rows',   type=int,   default=64,             help='resized image height')
parser.add_argument('--workers',    type=int,   default=1,              help='Data loader workers')

parser.add_argument('--model_path',    type=str,   default='save',      help='Saved model path')

args = parser.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)
plt.switch_backend('agg')  # Allow plotting when running remotely

save_epoch = 2 # save log images per save_epoch

# 02 rotation + flip augmentation option
# Setup Augmentations
data_aug_tr= Compose([Scale(args.img_cols), # resize longer side of an image to the defined size
                      CenterPadding([args.img_rows, args.img_cols]), # zero pad remaining regions
                      RandomHorizontallyFlip(), # random horizontal flip
                      RandomRotate(180)])  # ramdom rotation

data_aug_te= Compose([Scale(args.img_cols), 
                     CenterPadding([args.img_rows, args.img_cols])])

result_path = 'results_' + args.dataset
if not os.path.exists(result_path):
  os.makedirs(result_path)
outimg_path =  "./img_log_" + args.dataset
if not os.path.exists(outimg_path):
  os.makedirs(outimg_path)

f_loss = open(os.path.join(result_path, "log_loss.txt"),'w')
f_loss.write('Network type: %s\n'%args.arch)
f_loss.write('Learning rate: %05f\n'%args.lr)
f_loss.write('batch-size: %s\n'%args.batch_size)
f_loss.write('img_cols: %s\n'%args.img_cols)
f_loss.write('Augmentation type: flip, centercrop\n\n')
f_loss.close()

f_iou = open(os.path.join(result_path, "log_acc.txt"),'w')
f_iou.close()

f_iou = open(os.path.join(result_path, "log_val_acc.txt"),'w')
f_iou.close()

def save_model(eps, model):
    out = os.path.join(args.model_path, "checkpoint_{}.pth".format(eps))

    # To save a DataParallel model generically, save the model.module.state_dict().
    # This way, you have the flexibility to load the model any way you want to any device you want.
    torch.save(model.state_dict(), out)

data_path = get_data_path("gtsrb2TT100K")

simclr_dataloader = gtsrbClassifierData(data_path, args.exp, split='train', img_size=(args.img_rows, args.img_cols))

train_loader = DataLoader(simclr_dataloader, batch_size=args.batch_size, shuffle=True, drop_last=True)

######### Config Parameters #########
weight_decay = 1.0e-6
temperature = 0.5
projection_dim = 64
epochs = 100
clf_batch_size = 64
nodes = 1
gpus = 1
world_size = gpus * nodes

#### Initialize writer ####
writer = SummaryWriter()

if not os.path.exists(args.model_path):
    os.makedirs(args.model_path)

encoder = get_resnet("resnet18", pretrained=False)
n_features = encoder.fc.in_features

model = SimCLR(encoder, projection_dim, n_features).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)  # TODO: LARS
criterion = NT_Xent(clf_batch_size, temperature, world_size)

#### Start training ####
global_step = 0

for e in range(epochs):
    
    loss_epoch = 0
    lr = optimizer.param_groups[0]["lr"]

    for step, (x_i, x_j) in enumerate(train_loader):
        optimizer.zero_grad()
        x_i = x_i.to(device)
        x_j = x_j.to(device)

        # positive pair, with encoding
        h_i, h_j, z_i, z_j = model(x_i, x_j)

        #print(z_i.shape, z_j.shape) # [64, 64], [64, 64]

        loss = criterion(z_i, z_j)
        loss.backward()

        optimizer.step()

        if step % 50 == 0:
            print(f"Step [{step}]/{len(train_loader)}]\t Loss: {loss.item()}")
        
        writer.add_scalar("Loss/train_epoch", loss.item(), global_step)
        global_step += 1

        loss_epoch += loss.item()
    
    if e % 10 == 0:
        save_model(e, model)
    
    writer.add_scalar("Loss/train", loss_epoch / len(train_loader), e)
    writer.add_scalar("Misc/learning_rate", lr, e)

    print(
        f"Epoch [{e}/{epochs}]\t Loss: {loss_epoch / len(train_loader)}\t lr: {round(lr, 5)}"
    )

save_model(epochs-1, model)