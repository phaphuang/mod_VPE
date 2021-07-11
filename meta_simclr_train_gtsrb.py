from argparse import ArgumentParser
from models.vqvae2_disc import GANLoss
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

from loader import get_loader, get_data_path
from augmentations import *

from models.SimCLR.gtsrbClassifierData import gtsrbClassifierData
from models.SimCLR.nx_xnet import NT_Xent
from models.simclr import SimCLR
from models.SimCLR.resnet import get_resnet
from torch.utils.tensorboard import SummaryWriter

from models.simclr_vaegan import VAEIdsia, define_D

import json
from torch import autograd
import learn2learn as l2l

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Setup
parser = ArgumentParser(description='Variational Prototyping Encoder (VPE)')
parser.add_argument('--seed',       type=int,   default=42,             help='Random seed')
parser.add_argument('--arch',       type=str,   default='vaeIdsia',  help='network type: vaeIdsia, vaeIdsiaStn')
parser.add_argument('--dataset',    type=str,   default='gtsrbMeta', help='dataset to use [gtsrbMetaLoader]')
parser.add_argument('--exp',        type=str,   default='exp_list',     help='training scenario')
parser.add_argument('--resume',     type=str,   default=None,           help='Resume training from previously saved model')

parser.add_argument('--epochs',     type=int,   default=10000,           help='Training epochs')
parser.add_argument('--inner_epochs',     type=int,   default=10,           help='Inner training epochs')
parser.add_argument('--lr',         type=float, default=1e-4,           help='Learning rate')
parser.add_argument('--batch_size', type=int,   default=64,            help='Batch size')

parser.add_argument('--img_cols',   type=int,   default=64,             help='resized image width')
parser.add_argument('--img_rows',   type=int,   default=64,             help='resized image height')
parser.add_argument('--workers',    type=int,   default=1,              help='Data loader workers')

parser.add_argument('--model_path',    type=str,   default='save/simclr',      help='Saved model path')
parser.add_argument('--gantype', type=str, default="lsgan", help="Setting GAN Loss")

args = parser.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
plt.switch_backend('agg')  # Allow plotting when running remotely

save_epoch = 1 # save log images per save_epoch

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

def load_model(eps, model):
    print("########## Loading model ############")
    out = os.path.join(args.model_path, "checkpoint_{}.pth".format(eps))

    # To save a DataParallel model generically, save the model.module.state_dict().
    # This way, you have the flexibility to load the model any way you want to any device you want.
    model.load_state_dict(torch.load(out))

    return model

#### Initialize writer ####
writer = SummaryWriter()

######### Config Parameters #########
weight_decay = 1.0e-6
temperature = 0.5
projection_dim = 64
epochs = 100
clf_batch_size = 64
nodes = 1
gpus = 1
world_size = gpus * nodes

############# Design Loss #############
gan_criterion = GANLoss().to(device)

reconstruction_function = nn.BCELoss()
reconstruction_function.reduction = 'sum'
def loss_function(recon_x, x, mu, logvar):
    BCE = reconstruction_function(recon_x, x)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    return BCE + KLD

#simclr_optim = torch.optim.Adam(simclr.parameters(), lr=3e-4)  # TODO: LARS
#simclr_criterion = NT_Xent(clf_batch_size, temperature, world_size)

#num_train = len(trainloader)
#num_test = len(testloader)

def compute_grad_gp_wgan(D, x_real, x_fake, device):
    alpha = torch.rand(x_real.size(0), 1, 1, 1).to(device)

    x_interpolate = ((1 - alpha) * x_real + alpha * x_fake).detach()
    x_interpolate.requires_grad = True
    d_inter_logit = D(x_interpolate)
    grad = torch.autograd.grad(d_inter_logit, x_interpolate,
                               grad_outputs=torch.ones_like(d_inter_logit), create_graph=True)[0]

    norm = grad.view(grad.size(0), -1).norm(p=2, dim=1)

    d_gp = ((norm - 1) ** 2).mean()
    return d_gp

def compute_grad_gp(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True)[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = grad_dout2.view(batch_size, -1).sum(1)
    return reg

#### Start training ####
class MetaSim:
    def __init__(self):
        self.initialize_models()
        self.save_epoch = 100
        # Loading data
        data_loader = get_loader(args.dataset)
        data_path = get_data_path(args.dataset)

        tr_loader = data_loader(data_path, args.exp, is_transform=True, split='train', img_size=(args.img_rows, args.img_cols), augmentations=data_aug_tr)

        tr_meta_loader = l2l.data.MetaDataset(tr_loader)
        tr_labels = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
        transforms = [  # Easy to define your own transform
            l2l.data.transforms.FilterLabels(tr_meta_loader, labels=tr_labels),
            l2l.data.transforms.NWays(tr_meta_loader, n=5),
            l2l.data.transforms.KShots(tr_meta_loader, k=1),
            l2l.data.transforms.LoadData(tr_meta_loader)
        ]
        self.trainloader = l2l.data.TaskDataset(tr_meta_loader, transforms, num_tasks=1000)

        te_loader = data_loader(data_path, args.exp, is_transform=True, split='train', img_size=(args.img_rows, args.img_cols), augmentations=data_aug_te)
        te_meta_loader = l2l.data.MetaDataset(te_loader)
        te_labels = [20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42]
        transforms = [  # Easy to define your own transform
            l2l.data.transforms.FilterLabels(te_meta_loader, labels=te_labels),
            l2l.data.transforms.NWays(te_meta_loader, n=5),
            l2l.data.transforms.KShots(te_meta_loader, k=1),
            l2l.data.transforms.LoadData(te_meta_loader)
        ]
        self.testloader = l2l.data.TaskDataset(te_meta_loader, transforms, num_tasks=1000)
    
    def initialize_models(self):
        ######### Initialize the SimCLR parameters ########
        encoder = get_resnet("resnet18", pretrained=False)
        n_features = encoder.fc.in_features

        simclr = SimCLR(encoder, projection_dim, n_features).to(device)

        #### Load model parameters ####
        self.pretrain_simclr = load_model(99, simclr)
        self.pretrain_simclr.eval()

        ######### Initialize the vae and discriminator model #########
        self.net_vae = VAEIdsia(nc=3, input_size = 64, conditioned_latent_size=64, latent_variable_size=300, cnn_chn=[100, 150, 250] ,param1=[200,300,200], param2=None, param3 = [150, 150, 150])
        #net_vae.to(device)

        self.net_d = define_D(ch=3, nf=32)
        #net_d.to(device)

        self.metanet_vae = VAEIdsia(nc=3, input_size = 64, conditioned_latent_size=64, latent_variable_size=300, cnn_chn=[100, 150, 250] ,param1=[200,300,200], param2=None, param3 = [150, 150, 150])
        self.metanet_vae.to(device)

        self.metanet_d = define_D(ch=3, nf=32)
        self.metanet_d.to(device)

        # Construct optimiser
        self.vae_optim = optim.Adam(self.net_vae.parameters(), lr=args.lr) # 1e-4
        self.d_optim = optim.Adam(self.net_d.parameters(), lr=args.lr*0.1)

        self.metavae_optim = optim.Adam(self.metanet_vae.parameters(), lr=args.lr*10) # 1e-4
        self.metad_optim = optim.Adam(self.metanet_d.parameters(), lr=args.lr*0.1)


    def inner_loop(self, e, data):

        #print('start train episode: %d/%d'% (str(e), str(ne)))

        input, target, template = data

        target = torch.squeeze(target)
        input, template = input.to(device), template.to(device)

        # positive pair, with encoding
        _, _, z_i, _ = self.pretrain_simclr(input, input)

        ######## Training VAE #########
        self.metavae_optim.zero_grad()

        recon, mu, logvar, input_stn = self.metanet_vae(input, z_i)
        vae_loss = loss_function(recon, template, mu, logvar)

        vae_loss.backward()
        self.metavae_optim.step()

        ######## Training Discriminator #############
        template.requires_grad = True

        self.metad_optim.zero_grad()

        d_fake_logit = self.metanet_d(recon.detach())
        d_real_logit = self.metanet_d(template)

        ones = torch.ones_like(d_real_logit).to(device)
        zeros = torch.zeros_like(d_fake_logit).to(device)

        if args.gantype == "wgangp":
            # wgan gp
            d_fake = torch.mean(d_fake_logit)
            d_real = -torch.mean(d_real_logit)
            d_gp = compute_grad_gp_wgan(self.metanet_d, template, recon, device)
            d_loss = d_real + d_fake + 0.1 * d_gp
        elif args.gantype == 'zerogp':
            d_fake = F.binary_cross_entropy(d_fake_logit, zeros, reduction='none').mean()
            d_real = F.binary_cross_entropy(d_real_logit, ones, reduction='none').mean()
            d_gp = compute_grad_gp(torch.mean(d_real_logit), template)
            d_loss = d_real + d_fake + 10.0 * d_gp
        elif args.gantype == 'lsgan':
            d_fake = F.mse_loss(torch.mean(d_fake_logit), zeros)
            d_real = F.mse_loss(torch.mean(d_real_logit), 0.9 * ones)
            d_loss = d_real + d_fake

        d_loss.backward()
        self.metad_optim.step()
        
        return vae_loss.item(), d_loss.item(), input, recon, template

    def test(self, e, testloader):
        data = testloader.sample()

        vae_total_loss = 0
        d_total_loss = 0

        for ie in range(args.inner_epochs):
            vae_loss, d_loss, input, recon, template = self.inner_loop(e, data)
            vae_total_loss += vae_loss
            d_total_loss += d_loss

        print(f"Episode: {e}\tTotal VAE Loss: {vae_total_loss:.2f}\t D Loss: {d_total_loss:.2f}")
        
        ######### Evaluation ########
        #self.metanet_vae.eval()

        _, _, z_i, _ = self.pretrain_simclr(input, input)

        with torch.no_grad():
            recon, mu, logvar, input_stn  = self.metanet_vae(input, z_i)

        out_folder =  "%s/Epoch_%d_test"%(outimg_path, e)
        out_root = Path(out_folder)
        if not out_root.is_dir():
            os.mkdir(out_root)

        torchvision.utils.save_image(input.data, '{}/episode_{}_data.jpg'.format(out_folder,e), nrow=8, padding=2)
        torchvision.utils.save_image(input_stn.data, '{}/episode_{}_data_stn.jpg'.format(out_folder, e), nrow=8, padding=2) 
        torchvision.utils.save_image(recon.data, '{}/episode_{}_recon.jpg'.format(out_folder,e), nrow=8, padding=2)
        torchvision.utils.save_image(template.data, '{}/episode_{}_target.jpg'.format(out_folder,e), nrow=8, padding=2)

    def meta_training_loop(self, e, trainloader):
        data = trainloader.sample()

        vae_total_loss = 0
        d_total_loss = 0

        for ie in range(args.inner_epochs):
            vae_loss, d_loss, input, recon, template = self.inner_loop(e, data)
            vae_total_loss += vae_loss
            d_total_loss += d_loss
        
        print(f"Episode: {e}\tTotal VAE Loss: {vae_total_loss:.2f}\t D Loss: {d_total_loss:.2f}")
        
        for p, meta_p in zip(self.net_vae.parameters(), self.metanet_vae.parameters()):
            diff = p - meta_p.cpu()
            p.grad = diff
        self.vae_optim.step()

        for p, meta_p in zip(self.net_d.parameters(), self.metanet_d.parameters()):
            diff = p - meta_p.cpu()
            p.grad = diff
        self.d_optim.step()

        if (e%self.save_epoch == 0):
            out_folder =  "%s/Epoch_%d_train"%(outimg_path, e)
            out_root = Path(out_folder)
            if not out_root.is_dir():
                os.mkdir(out_root)

            torchvision.utils.save_image(input.data, '{}/episode_{}_data.jpg'.format(out_folder,e), nrow=8, padding=2)
            torchvision.utils.save_image(recon.data, '{}/episode_{}_recon.jpg'.format(out_folder,e), nrow=8, padding=2)
            torchvision.utils.save_image(template.data, '{}/episode_{}_target.jpg'.format(out_folder,e), nrow=8, padding=2)

    def reset_meta_model(self):
        self.metanet_vae.train()
        self.metanet_vae.load_state_dict(self.net_vae.state_dict())

        self.metanet_d.train()
        self.metanet_d.load_state_dict(self.net_d.state_dict())
    
    def training(self):
        for e in range(1, args.epochs + 1):
            self.reset_meta_model()
            self.meta_training_loop(e, self.trainloader)

            if e%100 == 0:
                self.reset_meta_model()
                self.test(e, self.testloader)


if __name__ == "__main__":
    env = MetaSim()
    env.training()