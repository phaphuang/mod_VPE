from argparse import ArgumentParser
from models.vqvae2_base import Encoder
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
from models.image2image import ContentEncoder, StyleEncoder, Decoder, Discriminator, PerceptualContentLossVgg16, ContentLoss, StyleLoss, total_variation_loss
from augmentations import *

import torchvision.models as models

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
parser.add_argument('--batch_size', type=int,   default=16,            help='Batch size')

parser.add_argument('--img_cols',   type=int,   default=64,             help='resized image width')
parser.add_argument('--img_rows',   type=int,   default=64,             help='resized image height')
parser.add_argument('--workers',    type=int,   default=1,              help='Data loader workers')

args = parser.parse_args()

SEED = 1
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
plt.switch_backend('agg')  # Allow plotting when running remotely

save_epoch = 50 # save log images per save_epoch

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

# set up GPU
# we could do os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2"

# Data
data_loader = get_loader_norm(args.dataset)
data_path = get_data_path(args.dataset)

tr_loader = data_loader(data_path, args.exp, is_transform=True, split='train', img_size=(args.img_rows, args.img_cols), augmentations=data_aug_tr)
te_loader = data_loader(data_path, args.exp, is_transform=True, split='test', img_size=(args.img_rows, args.img_cols), augmentations=data_aug_te)

trainloader = DataLoader(tr_loader, batch_size=args.batch_size, num_workers=args.workers, shuffle=True, pin_memory=True)
testloader = DataLoader(te_loader, batch_size=args.batch_size, num_workers=args.workers, shuffle=True, pin_memory=True)

# define model or load model
channels = 3
nsf = ndf = 64

content_net = ContentEncoder(channels=channels, nsf=nsf).to(device)
style_net = StyleEncoder(channels=channels, nsf=nsf).to(device)
decoder = Decoder(channels=channels, ndf=ndf).to(device)
discrimininator = Discriminator(channels=channels, ndf=ndf).to(device)

# Define Vgg model
model16 = models.vgg16(pretrained=True).to(device)

# if args.resume is not None:
#   pre_params = torch.load(args.resume)
#   net.init_params(pre_params)

# Construct optimiser
content_optim = optim.Adam(content_net.parameters(), lr=args.lr, betas=(0.5, 0.999)) # 1e-4
style_optim = optim.Adam(style_net.parameters(), lr=args.lr, betas=(0.5, 0.999))
dec_optim = optim.Adam(decoder.parameters(), lr=args.lr, betas=(0.5, 0.999))
disc_optim = optim.Adam(discrimininator.parameters(), lr=args.lr, betas=(0.5, 0.999))

num_train = len(tr_loader.targets)
num_test = len(te_loader.targets)
batch_iter = math.ceil(num_train/args.batch_size)
batch_iter_test = math.ceil(num_test/args.batch_size)

mse_loss = nn.MSELoss()
perceptual_loss_content = PerceptualContentLossVgg16(model16)


def squared_difference(input, target):
    return (input - target) ** 2

def generator_loss(fake):
	# LS Gan
	n_scale = len(fake)
	loss = []
	fake_loss = 0
	for i in range(n_scale) :
		fake_loss = torch.mean(squared_difference(fake[i], 1.0))
		loss.append(fake_loss)

	return sum(loss) # sum up different scale loss

def discriminator_loss(real, fake):
	n_scale = len(real)
	loss = []

	real_loss = 0
	fake_loss = 0

	for i in range(n_scale) :
		real_loss = torch.mean(squared_difference(real[i], 1.0))
		fake_loss = torch.mean((fake[i])**2)

		loss.append(real_loss + fake_loss)

	return sum(loss)


def train(e):
    n_classes = tr_loader.n_classes

    print('start train epoch: %d'%e)
    content_net.train()
    style_net.train()
    decoder.train()
    discrimininator.train()
    
    for i, (input, target, template) in enumerate(trainloader):

        target = torch.squeeze(target)
        input, template = input.to(device), template.to(device)

        ########################### Stage 1 Training #########################

        content_optim.zero_grad()
        style_optim.zero_grad()
        decoder.zero_grad()

        content_A, content_A_layers = content_net(input)
        content_B, content_B_layers = content_net(template)
        style_A, style_A_layers = style_net(input)
        style_B, style_B_layers = style_net(template)

        sty_A_con_B = decoder(content_B, style_A, style_A_layers, stylize=True)
        sty_B_con_A = decoder(content_A, style_B, style_B_layers, stylize=True)

        # decoding
        sty_A_con_A = decoder(content_A, content_A, content_A_layers)
        sty_B_con_B = decoder(content_B, content_B, content_B_layers)

        # 1) reconstruction loss
        loss_recon_A_1 = mse_loss(sty_A_con_A, input)
        loss_recon_B_1 = mse_loss(sty_B_con_B, template)

        # 2) content loss
        perceptual_loss_A_1 = perceptual_loss_content(sty_A_con_A, input)
        perceptual_loss_B_1 = perceptual_loss_content(sty_B_con_B, template)

        # 3) total variation loss
        tv_loss_A = total_variation_loss(sty_A_con_A)
        tv_loss_B = total_variation_loss(sty_B_con_B)

        gen_loss_1 = 10 * (loss_recon_A_1 + loss_recon_B_1) + 1 * (perceptual_loss_A_1+perceptual_loss_B_1) + 10 * (tv_loss_A+tv_loss_B)

        # gen_loss_1.backward(retain_graph=True)
        # content_optim.step()
        # style_optim.step()
        # dec_optim.step()

        ########################### Stage 2 Training #########################
        # content_optim.zero_grad()
        # style_optim.zero_grad()
        # dec_optim.zero_grad()


        cycle_content_A, cycle_content_A_layers = content_net(sty_B_con_A)
        cycle_content_B, cycle_content_B_layers = content_net(sty_A_con_B)
        cycle_style_A, cycle_style_A_layers = style_net(sty_A_con_B)
        cycle_style_B, cycle_style_B_layers = style_net(sty_B_con_A)

        cycle_A = decoder(cycle_content_A, cycle_style_B, cycle_style_B_layers, stylize=True)
        cycle_B = decoder(cycle_content_B, cycle_style_A, cycle_style_A_layers, stylize=True)

        content_feature_loss_A = mse_loss(cycle_content_A, content_A)
        content_feature_loss_B = mse_loss(cycle_content_B, content_B)
        style_feature_loss_A = mse_loss(cycle_style_A, style_A)
        style_feature_loss_B = mse_loss(cycle_style_B, style_B)

        feature_loss = content_feature_loss_A + content_feature_loss_B + style_feature_loss_A + style_feature_loss_B

        cycle_loss_A = mse_loss(cycle_A, input)
        cycle_loss_B = mse_loss(cycle_B, template)

        perceptual_loss_fake_A_1 = perceptual_loss_content(sty_B_con_A, input)
        perceptual_loss_fake_B_1 = perceptual_loss_content(sty_A_con_B, template)

        real_A_logit, real_B_logit = discrimininator(input), discrimininator(template)
        fake_A_logit, fake_B_logit = discrimininator(sty_A_con_B), discrimininator(sty_B_con_A)

        g_loss_A = generator_loss(fake_A_logit)
        g_loss_B = generator_loss(fake_B_logit)

        gen_loss_2 = (g_loss_A + g_loss_B) * 15 + gen_loss_1 * 0.01 + (cycle_loss_A + cycle_loss_B) * 100 + \
                     (perceptual_loss_fake_A_1 + perceptual_loss_fake_B_1) * 0.05 + feature_loss
        
        gen_loss = gen_loss_1 + gen_loss_2

        gen_loss.backward()
        content_optim.step()
        style_optim.step()
        dec_optim.step()
        
        disc_optim.zero_grad()

        real_A_logit, real_B_logit = discrimininator(input.detach()), discrimininator(template.detach())
        fake_A_logit, fake_B_logit = discrimininator(sty_A_con_B.detach()), discrimininator(sty_B_con_A.detach())

        d_loss_A = discriminator_loss(real_A_logit, fake_A_logit)
        d_loss_B = discriminator_loss(real_B_logit, fake_B_logit)

        disc_loss = d_loss_A + d_loss_B

        disc_loss.backward()
        disc_optim.step()

        print('Epoch:%d  Batch:%d/%d  G1 loss:%08f G2 loss:%08f D loss:%08f'%(e, i, batch_iter, gen_loss_1.data/input.numel(), gen_loss_2.data/input.numel(), disc_loss.data/input.numel()))
    
        f_loss = open(os.path.join(result_path, "log_loss.txt"),'a')
        f_loss.write('Epoch:%d  Batch:%d/%d  G1 loss:%08f G2 loss:%08f D loss:%08f'%(e, i, batch_iter, gen_loss_1.data/input.numel(), gen_loss_2.data/input.numel(), disc_loss.data/input.numel()))
        f_loss.close()

        if i < 1 and (e%save_epoch == 0):
            out_folder =  "%s/Epoch_%d_train"%(outimg_path, e)
            out_root = Path(out_folder)
            if not out_root.is_dir():
                os.mkdir(out_root)

            torchvision.utils.save_image(input.data, '{}/batch_{}_data.jpg'.format(out_folder,i), nrow=8, padding=2)
            #torchvision.utils.save_image(input_stn.data, '{}/batch_{}_data_stn.jpg'.format(out_folder, i), nrow=8, padding=2) 
            torchvision.utils.save_image(sty_B_con_A.data, '{}/batch_{}_recon.jpg'.format(out_folder,i), nrow=8, padding=2)
            torchvision.utils.save_image(template.data, '{}/batch_{}_target.jpg'.format(out_folder,i), nrow=8, padding=2)

        # if (i > 0) and (i % 10 == 0):
        #     break

    if e%save_epoch == 0:
        class_target = torch.LongTensor(list(range(n_classes)))
        class_template = tr_loader.load_template(class_target)
        class_template = class_template.to(device)
        with torch.no_grad():
            content_A, _ = content_net(class_template)
            class_recon = decoder(content_A)
        
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

        cls_mat = cls_feat.repeat(pred.shape[0], 1, 1, 1)
        #print(cls_mat.shape)
        # euclidean distance
        sample_distance[:,i] = torch.norm(pred - cls_mat,p=2, dim=[1,2,3])
    
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

    content_net.eval()
    decoder.eval()

    accum_all = torch.zeros(n_classes)
    rank_all = torch.zeros(n_classes, n_classes) # rank per class
    accum_class = torch.zeros(n_classes)

    # get template latent z
    class_target = torch.LongTensor(list(range(n_classes)))
    class_template = te_loader.load_template(class_target)
    class_template = class_template.to(device)
    with torch.no_grad():
        content_A, _ = content_net(class_template)
        class_recon = decoder(content_A)
    
    for i, (input, target, template) in enumerate(testloader):

        target = torch.squeeze(target)
        input, template = input.to(device), template.to(device)
        with torch.no_grad():
            z_input, _ = content_net(input)
            recon = decoder(z_input)
        
        sample_correct, sample_all, sample_rank = score_NN(z_input, content_A, target, n_classes)
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
            #torchvision.utils.save_image(input_stn.data, '{}/batch_{}_data_stn.jpg'.format(out_folder, i), nrow=8, padding=2) 
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
        #torch.save(net.state_dict(), os.path.join('%s_testBest_net.pth'%args.dataset))

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

    ############# Plot scores ##############
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
