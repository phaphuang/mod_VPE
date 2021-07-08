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

parser.add_argument('--model_path',    type=str,   default='save/simclr',      help='Saved model path')
parser.add_argument('--gantype', type=str, default="wgangp", help="Setting GAN Loss")

args = parser.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
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

def load_model(eps, model):
    print("########## Loading model ############")
    out = os.path.join(args.model_path, "checkpoint_{}.pth".format(eps))

    # To save a DataParallel model generically, save the model.module.state_dict().
    # This way, you have the flexibility to load the model any way you want to any device you want.
    model.load_state_dict(torch.load(out))

    return model

# Loading data
data_loader = get_loader(args.dataset)
data_path = get_data_path(args.dataset)

tr_loader = data_loader(data_path, args.exp, is_transform=True, split='train', img_size=(args.img_rows, args.img_cols), augmentations=data_aug_tr)
te_loader = data_loader(data_path, args.exp, is_transform=True, split='test', img_size=(args.img_rows, args.img_cols), augmentations=data_aug_te)

trainloader = DataLoader(tr_loader, batch_size=args.batch_size, shuffle=True)
testloader = DataLoader(te_loader, batch_size=args.batch_size, shuffle=True)

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

######### Initialize the vae and discriminator model #########
net_vae = VAEIdsia(nc=3, input_size = 64, conditioned_latent_size=64, latent_variable_size=300, cnn_chn=[100, 150, 250] ,param1=[200,300,200], param2=None, param3 = [150, 150, 150])
net_vae.to(device)

net_d = define_D(ch=3, nf=32)
net_d.to(device)

# Construct optimiser
vae_optimizer = optim.Adam(net_vae.parameters(), lr=args.lr) # 1e-4
d_optimizer = optim.Adam(net_d.parameters(), lr=args.lr*0.1)

gan_criterion = GANLoss().to(device)

reconstruction_function = nn.BCELoss()
reconstruction_function.reduction = 'sum'
def loss_function(recon_x, x, mu, logvar):
    BCE = reconstruction_function(recon_x, x)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    return BCE + KLD

######### Initialize the SimCLR parameters ########
encoder = get_resnet("resnet18", pretrained=False)
n_features = encoder.fc.in_features

simclr = SimCLR(encoder, projection_dim, n_features).to(device)

#### Load model parameters ####
pretrain_simclr = load_model(99, simclr)
pretrain_simclr.eval()

#simclr_optimizer = torch.optim.Adam(simclr.parameters(), lr=3e-4)  # TODO: LARS
#simclr_criterion = NT_Xent(clf_batch_size, temperature, world_size)

num_train = len(tr_loader.targets)
num_test = len(te_loader.targets)
batch_iter = math.ceil(num_train/args.batch_size)
batch_iter_test = math.ceil(num_test/args.batch_size)

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
global_step = 0

def train(e):

    n_classes = tr_loader.n_classes
    n_classes_te = te_loader.n_classes

    print('start train epoch: %d'%e)
    net_vae.train()
    net_d.train()

    for i, (input, target, template) in enumerate(trainloader):
        target = torch.squeeze(target)
        input, template = input.to(device), template.to(device)

        # positive pair, with encoding
        _, _, z_i, _ = pretrain_simclr(input, input)

        ######## Training VAE #########
        vae_optimizer.zero_grad()

        recon, mu, logvar, input_stn = net_vae(input, z_i)
        vae_loss = loss_function(recon, template, mu, logvar)

        vae_loss.backward()
        vae_optimizer.step()

        ######## Training Discriminator #############
        template.requires_grad = True

        d_optimizer.zero_grad()

        d_fake_logit = net_d(recon.detach())
        d_real_logit = net_d(template)

        ones = torch.ones_like(d_real_logit).to(device)
        zeros = torch.zeros_like(d_fake_logit).to(device)

        if args.gantype == "wgangp":
            # wgan gp
            d_fake = torch.mean(d_fake_logit)
            d_real = -torch.mean(d_real_logit)
            d_gp = compute_grad_gp_wgan(net_d, template, recon, device)
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
        d_optimizer.step()

        ############### Print out the value ###############
        print('Epoch:%d  Batch:%d/%d  VAE loss:%08f  Dis loss:%08f'%(e, i, batch_iter, vae_loss.data/input.numel(), d_loss.data/input.numel()))

        f_loss = open(os.path.join(result_path, "log_loss.txt"),'a')
        f_loss.write('Epoch:%d  Batch:%d/%d  VAE loss:%08f  Dis loss:%08f'%(e, i, batch_iter, vae_loss.data/input.numel(), d_loss.data/input.numel()))
        f_loss.close()

        if i < 1 and (e%save_epoch == 0):
            out_folder =  "%s/Epoch_%d_train"%(outimg_path, e)
            out_root = Path(out_folder)
            if not out_root.is_dir():
                os.mkdir(out_root)

            torchvision.utils.save_image(input.data, '{}/batch_{}_data.jpg'.format(out_folder,i), nrow=8, padding=2)
            torchvision.utils.save_image(input_stn.data, '{}/batch_{}_data_stn.jpg'.format(out_folder, i), nrow=8, padding=2) 
            torchvision.utils.save_image(recon.data, '{}/batch_{}_recon.jpg'.format(out_folder,i), nrow=8, padding=2)
            torchvision.utils.save_image(template.data, '{}/batch_{}_target.jpg'.format(out_folder,i), nrow=8, padding=2)
        
        # if (i > 0) and (i % 10 == 0):
        #     break
    
    if e%save_epoch == 0:
        class_target = torch.LongTensor(list(range(n_classes)))
        class_template = tr_loader.load_template(class_target)

        class_template = class_template.cuda(async=True)

        _, _, z_i, _ = pretrain_simclr(class_template, class_template)

        with torch.no_grad():
            class_recon, class_mu, class_logvar, _ = net_vae(class_template, z_i)
        
        torchvision.utils.save_image(class_template.data, '{}/templates.jpg'.format(out_folder), nrow=8, padding=2)  
        torchvision.utils.save_image(class_recon.data, '{}/templates_recon.jpg'.format(out_folder), nrow=8, padding=2) 


def score_NN(pred, class_feature, label, n_classes):

  sample_correct = torch.zeros(n_classes)
  sample_all = torch.zeros(n_classes)
  sample_rank = torch.zeros(n_classes, n_classes) # rank per class
  sample_distance = torch.ones(pred.shape[0], n_classes)*math.inf

  pred = pred.data.cpu() # batch x latent size
  class_feature = class_feature.data.cpu() # n_classes x latent size
  label = label.numpy()
  for i in range(n_classes):
    cls_feat = class_feature[i,:]
    #print("Class Feature Shape: ", pred.shape, cls_feat.shape)
    cls_mat = cls_feat.repeat(pred.shape[0],1)
    #print(cls_mat.shape)
    # euclidean distance
    sample_distance[:,i] = torch.norm(pred - cls_mat,p=2, dim=1)
  
  sample_distance = sample_distance.cpu().numpy()
  indices = np.argsort(sample_distance, axis=1) # sort ascending order

  for i in range(indices.shape[0]):
    rank = np.where(indices[i,:] == label[i])[0][0] # find rank
    sample_rank[label[i]][rank:] += 1 # update rank 
    sample_all[label[i]] += 1 # count samples per class
    if rank == 0:
      sample_correct[label[i]] += 1 # count rank 1 (correct classification)

  return sample_correct, sample_all, sample_rank


mean_scores = []
mean_rank = []

def test(e, best_acc):
    n_classes = te_loader.n_classes
    print('start test epoch: %d'%e)
    net_vae.eval()
    accum_all = torch.zeros(n_classes)
    rank_all = torch.zeros(n_classes, n_classes) # rank per class
    accum_class = torch.zeros(n_classes)

    # get template latent z
    class_target = torch.LongTensor(list(range(n_classes)))
    class_template = te_loader.load_template(class_target)
    class_template = class_template.to(device)

    _, _, z_i, _ = pretrain_simclr(class_template, class_template)

    with torch.no_grad():
        class_recon, class_mu, class_logvar, _ = net_vae(class_template, z_i)
    
    for i, (input, target, template) in enumerate(testloader):

        target = torch.squeeze(target)
        input, template = input.to(device), template.to(device)

        _, _, z_i, _ = pretrain_simclr(input, input)

        with torch.no_grad():
            recon, mu, logvar, input_stn  = net_vae(input, z_i)
        
        sample_correct, sample_all, sample_rank = score_NN(mu, class_mu, target, n_classes)
        accum_class += sample_correct
        accum_all += sample_all
        rank_all = rank_all + sample_rank # [class_id, topN]
        
        print('Epoch:%d  Batch:%d/%d  processing...'%(e, i, batch_iter_test))

        if i < 1 and (e%save_epoch == 0):
            out_folder =  "%s/Epoch_%d_test"%(outimg_path, e)
            out_root = Path(out_folder)
            if not out_root.is_dir():
                os.mkdir(out_root)

            torchvision.utils.save_image(input.data, '{}/batch_{}_data.jpg'.format(out_folder,i), nrow=8, padding=2)
            torchvision.utils.save_image(input_stn.data, '{}/batch_{}_data_stn.jpg'.format(out_folder, i), nrow=8, padding=2) 
            torchvision.utils.save_image(recon.data, '{}/batch_{}_recon.jpg'.format(out_folder,i), nrow=8, padding=2)
            torchvision.utils.save_image(template.data, '{}/batch_{}_target.jpg'.format(out_folder,i), nrow=8, padding=2)

    if e%save_epoch == 0:
        torchvision.utils.save_image(class_template.data, '{}/templates.jpg'.format(out_folder), nrow=8, padding=2)  
        torchvision.utils.save_image(class_recon.data, '{}/templates_recon.jpg'.format(out_folder), nrow=8, padding=2)  

    acc_all = accum_class.sum() / accum_all.sum() 
    acc_cls = torch.div(accum_class, accum_all)
    
    rank_sample_avg = rank_all.sum(0) / accum_all.sum() # [class_id, topN]
    rank_cls = torch.div(rank_all, torch.transpose(accum_all.expand_as(rank_all),0,1))
    rank_cls_avg = torch.mean(rank_cls,dim=0)



    # write result part
    acc_trcls = torch.gather(acc_cls, 0, te_loader.tr_class)
    acc_tecls =torch.gather(acc_cls, 0, te_loader.te_class)

    print('========epoch(%d)========='%e)
    print('Seen Classes')
    for i, class_acc in enumerate(acc_trcls):
        print('cls:%d  acc:%02f'%(te_loader.tr_class[i], class_acc))
    print('Unseen Classes')
    for i, class_acc in enumerate(acc_tecls):
        print('cls:%d  acc:%02f'%(te_loader.te_class[i], class_acc))
    print('====================================')
    print('acc_avg:%02f'%acc_all)
    print('acc_cls:%02f'%acc_cls.mean())
    print('acc_trcls:%02f'%acc_trcls.mean())
    print('acc_tecls:%02f'%acc_tecls.mean())
    print('rank sample avg: %02f'%rank_sample_avg.mean())
    print('rank cls avg: %02f'%rank_cls_avg.mean())
    print('====================================')

    f_iou = open(os.path.join(result_path, "log_acc.txt"),'a')
    f_iou.write('epoch(%d), acc_cls: %04f  acc_trcls: %04f  acc_tecls: %04f  acc_all: %04f  top3: %04f  top5: %04f\n'%(e, acc_cls.mean(), acc_trcls.mean(), acc_tecls.mean(), acc_all, rank_sample_avg[2], rank_sample_avg[4]))
    f_iou.close()


    # if best_acc < acc_all: # update best score
    #   best_acc = acc_all
    if best_acc < acc_tecls.mean(): # update best score

        f_iou_class = open(os.path.join(result_path, "best_iou.txt"),'w')
        f_rank = open(os.path.join(result_path, "best_rank.txt"),'w')
        #torch.save(net_vae.state_dict(), os.path.join('%s_testBest_net.pth'%args.dataset))

        best_acc = acc_tecls.mean()
        f_iou_class.write('Best score epoch:  %d\n'%e)
        f_iou_class.write('acc cls: %.4f  acc all: %.4f  rank mean: %.4f \n'%(acc_cls.mean(), acc_all, rank_all.mean()))
        f_iou_class.write('acc tr cls: %.4f  acc te cls: %.4f\n'%(acc_trcls.mean(), acc_tecls.mean()))
        f_iou_class.write('top3: %.4f  top5: %.4f\n'%(rank_sample_avg[2], rank_sample_avg[4]))

        f_iou_class.write('\nSeen classes\n')
        for i, class_acc in enumerate(acc_trcls):
            f_iou_class.write('cls:%d  acc:%02f\n'%(te_loader.tr_class[i], class_acc))
        f_iou_class.write('\nUnseen classes\n')
        for i, class_acc in enumerate(acc_tecls):
            f_iou_class.write('cls:%d  acc:%02f\n'%(te_loader.te_class[i], class_acc))
        f_iou_class.close()
        
        for i, rank_acc in enumerate(rank_sample_avg):
            f_rank.write('rank sample %d: %.4f\n'%(i+1, rank_acc))
        f_rank.write('\n')
        for i, rank_acc in enumerate(rank_cls_avg):
            f_rank.write('rank cls %d: %.4f\n'%(i+1, rank_acc))
        f_rank.close()
            
    # Save weights and scores
    # if e % 100 == 0:
    #   pass
        # torch.save(net.state_dict(), os.path.join('flickr2belga_latest_net.pth'))

    ############# Plot scores
    mean_scores.append(acc_tecls.mean())
    es = list(range(len(mean_scores)))
    plt.plot(es, mean_scores, 'b-')
    plt.xlabel('Epoch')
    plt.ylabel('Unseen mean IoU')
    plt.savefig(os.path.join(result_path, 'unseen_ious.png'))
    plt.close()

    ############# plot rank
    # mean_rank.append(rank_all.mean())
    # rank_es = list(range(len(mean_rank)))
    # plt.plot(rank_es, mean_rank, 'b-')
    # plt.xlabel('Epoch')
    # plt.ylabel('Mean rank')
    # plt.savefig(os.path.join(result_path, 'rank.png'))
    # plt.close()

    return best_acc

if __name__ == "__main__":
    out_root = Path(outimg_path)
    if not out_root.is_dir():
        os.mkdir(out_root)
    best_acc = 0
    for e in range(1, args.epochs + 1):
        train(e)
        best_acc = test(e, best_acc)