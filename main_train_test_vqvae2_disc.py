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

from loader import get_loader_norm, get_data_path
from models import get_model
from augmentations import *

from models.vqvae2_disc import define_D, GANLoss

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Setup
parser = ArgumentParser(description='Variational Prototyping Encoder (VPE)')
parser.add_argument('--seed',       type=int,   default=42,             help='Random seed')
parser.add_argument('--arch',       type=str,   default='vqvae2Disc',  help='network type: vaeIdsia, vaeIdsiaStn')
parser.add_argument('--dataset',    type=str,   default='gtsrb2TT100K', help='dataset to use [gtsrb, gtsrb2TT100K, belga2flickr, belga2toplogo]')
parser.add_argument('--exp',        type=str,   default='exp_list',     help='training scenario')
parser.add_argument('--resume',     type=str,   default=None,           help='Resume training from previously saved model')

parser.add_argument('--epochs',     type=int,   default=2000,           help='Training epochs')
parser.add_argument('--lr',         type=float, default=1e-4,           help='Learning rate')
parser.add_argument('--batch_size', type=int,   default=64,            help='Batch size')

parser.add_argument('--img_cols',   type=int,   default=64,             help='resized image width')
parser.add_argument('--img_rows',   type=int,   default=64,             help='resized image height')
parser.add_argument('--workers',    type=int,   default=1,              help='Data loader workers')

args = parser.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
plt.switch_backend('agg')  # Allow plotting when running remotely

save_epoch = 50 # save log images per save_epoch

# 02 rotation + flip augmentation option
# Setup Augmentations
data_aug_tr= Compose([Scale(args.img_cols), # resize longer side of an image to the defined size
                      CenterPadding([args.img_rows, args.img_cols]), # zero pad remaining regions
                      RandomHorizontallyFlip(), # random horizontal flip
                      RandomRotate(180)])  # ramdom rotation

data_aug_te= Compose([Scale(args.img_cols), 
                     CenterPadding([args.img_rows, args.img_cols]),
                     CenterCrop([args.img_rows, args.img_cols])])

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

# Data
data_loader = get_loader_norm(args.dataset)
data_path = get_data_path(args.dataset)

tr_loader = data_loader(data_path, args.exp, is_transform=True, split='train', img_size=(args.img_rows, args.img_cols), augmentations=data_aug_tr)
te_loader = data_loader(data_path, args.exp, is_transform=True, split='test', img_size=(args.img_rows, args.img_cols), augmentations=data_aug_te)

trainloader = DataLoader(tr_loader, batch_size=args.batch_size, num_workers=args.workers, shuffle=True, pin_memory=True)
testloader = DataLoader(te_loader, batch_size=args.batch_size, num_workers=args.workers, shuffle=True, pin_memory=True)

# define model or load model
net = get_model(args.arch, n_classes=None)
net.to(device)

if args.resume is not None:
  pre_params = torch.load(args.resume)
  net.init_params(pre_params)

# Construct optimiser
optimizer = optim.Adam(net.parameters(), lr=args.lr) # 1e-4

num_train = len(tr_loader.targets)
num_test = len(te_loader.targets)
batch_iter = math.ceil(num_train/args.batch_size)
batch_iter_test = math.ceil(num_test/args.batch_size)

# input_ch = 3, output_ch = 3
net_d = define_D(3+3, ndf=64, netD='basic')
net_d.to(device)

d_optimizer = optim.Adam(net_d.parameters(), lr=args.lr)

#criterion = nn.BCELoss()
#criterion.reduction = 'mean'
criterion = nn.MSELoss()
criterionGAN = GANLoss().to(device)

latent_loss_weight = 0.25

def toogle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)

def compute_loss(d_out, target):
    targets = d_out.new_full(size=d_out.size(), fill_value=target)
    loss = F.binary_cross_entropy_with_logits(d_out, targets)
    return loss
  
real_label = torch.tensor(1.0).to(device)
fake_label = torch.tensor(0.0).to(device)


def train(e):
  n_classes = tr_loader.n_classes
  n_classes_te = te_loader.n_classes

  print('start train epoch: %d'%e)
  net.train()
  net_d.train()
  
  for i, (input, target, template) in enumerate(trainloader):

    # if i > 10:
    #   break

    optimizer.zero_grad()
    target = torch.squeeze(target)
    input, template = input.to(device), template.to(device)

    recon, latent_loss, _, _ = net(input)
    recon_loss = criterion(recon, template)
    latent_loss = latent_loss.mean()
    #### Original VQ-VAE loss ####
    loss = recon_loss + latent_loss_weight * latent_loss
    loss.backward()
    optimizer.step()

    #### Discriminator Loss ####
    d_optimizer.zero_grad()

    # Train with fake
    fake_ab = torch.cat([input, recon], 1)
    pred_fake = net_d.forward(fake_ab.detach())
    loss_d_fake = criterionGAN(pred_fake, False)

    # Train with real
    real_ab = torch.cat([input, template], 1)
    pred_real = net_d.forward(real_ab.detach())
    loss_d_real = criterionGAN(pred_real, True)

    # Combined D loss
    d_loss = (loss_d_fake + loss_d_real) * 0.5

    d_loss.backward()

    d_optimizer.step()

    print('Epoch:%d  Batch:%d/%d  Total Loss:%08f Recon loss:%08f VQ loss:%08f Disc Loss: %08f'%(e, i, batch_iter, loss.data, recon_loss.data, latent_loss.data, d_loss.data))
   
    f_loss = open(os.path.join(result_path, "log_loss.txt"),'a')
    f_loss.write('Epoch:%d  Batch:%d/%d  Total Loss:%08f Recon loss:%08f VQ loss:%08f Disc Loss: %08f'%(e, i, batch_iter, loss.data, recon_loss.data, latent_loss.data, d_loss.data))
    f_loss.close()

    for param_group in optimizer.param_groups:
      print("Current learning rate is: {}".format(param_group['lr']))
    
    for param_group in d_optimizer.param_groups:
      print("Current d learning rate is: {}".format(param_group['lr']))

    if i < 1 and (e%save_epoch == 0):
    #if (e%save_epoch == 0):
      out_folder =  "%s/Epoch_%d_train"%(outimg_path, e)
      out_root = Path(out_folder)
      if not out_root.is_dir():
        os.mkdir(out_root)

      torchvision.utils.save_image(input.data, '{}/batch_{}_data.jpg'.format(out_folder,i), nrow=8, padding=2, normalize=True, range=(-1,1))
      #torchvision.utils.save_image(input_stn.data, '{}/batch_{}_data_stn.jpg'.format(out_folder, i), nrow=8, padding=2, normalize=True, range=(-1,1)) 
      torchvision.utils.save_image(recon.data, '{}/batch_{}_recon.jpg'.format(out_folder,i), nrow=8, padding=2, normalize=True, range=(-1,1))
      torchvision.utils.save_image(template.data, '{}/batch_{}_target.jpg'.format(out_folder,i), nrow=8, padding=2, normalize=True, range=(-1,1))

    # if (i > 0) and (i % 10 == 0):
    #     break

  if e%save_epoch == 0:
    class_target = torch.LongTensor(list(range(n_classes)))
    class_template = tr_loader.load_template(class_target)
    class_template = class_template.to(device)
    with torch.no_grad():
      class_recon, _, _, _ = net(class_template)
    
    torchvision.utils.save_image(class_template.data, '{}/templates.jpg'.format(out_folder), nrow=8, padding=2, normalize=True, range=(-1,1))  
    torchvision.utils.save_image(class_recon.data, '{}/templates_recon.jpg'.format(out_folder), nrow=8, padding=2, normalize=True, range=(-1,1))
  
def score_NN(pred, class_feature, label, n_classes):

  sample_correct = torch.zeros(n_classes)
  sample_all = torch.zeros(n_classes)
  sample_rank = torch.zeros(n_classes, n_classes) # rank per class
  sample_distance = torch.ones(pred.shape[0], n_classes)*math.inf

  pred = pred.data.cpu() # batch x latent size
  class_feature = class_feature.data.cpu() # n_classes x latent size
  label = label.numpy()

  pred = pred.view(pred.shape[0], -1)

  for i in range(n_classes):
    cls_feat = class_feature[i,:]

    cls_feat = cls_feat.view(-1)
    cls_mat = cls_feat.repeat(pred.shape[0],1)

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
  net.eval()
  accum_all = torch.zeros(n_classes)
  rank_all = torch.zeros(n_classes, n_classes) # rank per class
  accum_class = torch.zeros(n_classes)

  # get template latent z
  class_target = torch.LongTensor(list(range(n_classes)))
  class_template = te_loader.load_template(class_target)
  class_template = class_template.to(device)
  with torch.no_grad():
    class_recon, _, _, class_z  = net(class_template)
  
  for i, (input, target, template) in enumerate(testloader):

    target = torch.squeeze(target)
    input, template = input.to(device), template.to(device)
    with torch.no_grad():
      recon, _, _, z  = net(input)
    
    sample_correct, sample_all, sample_rank = score_NN(z, class_z, target, n_classes)
    accum_class += sample_correct
    accum_all += sample_all
    rank_all = rank_all + sample_rank # [class_id, topN]
    
    print('Epoch:%d  Batch:%d/%d  processing...'%(e, i, batch_iter_test))

    if i < 1 and (e%save_epoch == 0):
      out_folder =  "%s/Epoch_%d_test"%(outimg_path, e)
      out_root = Path(out_folder)
      if not out_root.is_dir():
        os.mkdir(out_root)

      torchvision.utils.save_image(input.data, '{}/batch_{}_data.jpg'.format(out_folder,i), nrow=8, padding=2, normalize=True, range=(-1,1))
      #torchvision.utils.save_image(input_stn.data, '{}/batch_{}_data_stn.jpg'.format(out_folder, i), nrow=8, padding=2, normalize=True, range=(-1,1)) 
      torchvision.utils.save_image(recon.data, '{}/batch_{}_recon.jpg'.format(out_folder,i), nrow=8, padding=2, normalize=True, range=(-1,1))
      torchvision.utils.save_image(template.data, '{}/batch_{}_target.jpg'.format(out_folder,i), nrow=8, padding=2, normalize=True, range=(-1,1))

  if e%save_epoch == 0:
    torchvision.utils.save_image(class_template.data, '{}/templates.jpg'.format(out_folder), nrow=8, padding=2, normalize=True, range=(-1,1))  
    torchvision.utils.save_image(class_recon.data, '{}/templates_recon.jpg'.format(out_folder), nrow=8, padding=2, normalize=True, range=(-1,1))  

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
    torch.save(net.state_dict(), os.path.join('%s_testBest_net.pth'%args.dataset))

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
    
  #### Save weights and scores
  if e % 100 == 0:
    checkpoint = {
      'vqvae': net.state_dict(),
      'discriminator': net_d.state_dict(),
      'epoch': e
    }
    torch.save(checkpoint, os.path.join('{}_latest_net_ep{}.pth'.format(args.dataset, e)))

  ############# Plot scores
  mean_scores.append(acc_tecls.mean())
  es = list(range(len(mean_scores)))
  plt.plot(es, mean_scores, 'b-')
  plt.xlabel('Epoch')
  plt.ylabel('Unseen mean IoU')
  plt.savefig(os.path.join(result_path, 'unseen_ious.png'))
  plt.close()

  return best_acc

if __name__ == "__main__":
  out_root = Path(outimg_path)
  if not out_root.is_dir():
    os.mkdir(out_root)
  best_acc = 0
  for e in range(1, args.epochs + 1):
    train(e)
    best_acc = test(e, best_acc)