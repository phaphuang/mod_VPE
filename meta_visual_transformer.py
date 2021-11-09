from argparse import ArgumentParser
#from models.vaeTranDis import Discriminators
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

from models.vaeTranDis import DiscriminatorWithClassifier

from loader import get_loader, get_data_path
from models import get_model
from augmentations import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setup
parser = ArgumentParser(description='Variational Prototyping Encoder (VPE)')
parser.add_argument('--seed',       type=int,
                    default=42,             help='Random seed')
parser.add_argument('--arch',       type=str,   default='vaeTranDis',
                    help='network type: vaeIdsia, vaeIdsiaStn, vaeTranDis')
parser.add_argument('--dataset',    type=str,   default='gtsrb',
                    help='dataset to use [gtsrb, gtsrb2TT100K, belga2flickr, belga2toplogo]')
parser.add_argument('--exp',        type=str,
                    default='exp_list',     help='training scenario')
parser.add_argument('--resume',     type=str,   default=None,
                    help='Resume training from previously saved model')

parser.add_argument('--epochs',     type=int,
                    default=2000,           help='Training epochs')
parser.add_argument('--lr',         type=float,
                    default=1e-4,           help='Learning rate')
parser.add_argument('--batch_size', type=int,
                    default=64,            help='Batch size')

parser.add_argument('--img_cols',   type=int,   default=64,
                    help='resized image width')
parser.add_argument('--img_rows',   type=int,   default=64,
                    help='resized image height')
parser.add_argument('--workers',    type=int,   default=4,
                    help='Data loader workers')

parser.add_argument("--lambda_attr", type=float, default=1, help='discriminator predict attribute loss lambda')
parser.add_argument("--lambda_GAN", type=float, default=1, help='GAN loss lambda')
parser.add_argument("--lambda_l1", type=float, default=1, help='pixel l1 loss lambda')

parser.add_argument('--style_out_channel',    type=int,   default=20,
                    help='Output of style encoder')

parser.add_argument('--n_style',    type=int,   default=8,
                    help='input style of encoder')

args = parser.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)
plt.switch_backend('agg')  # Allow plotting when running remotely

save_epoch = 10  # save log images per save_epoch

# 02 rotation + flip augmentation option
# Setup Augmentations
data_aug_tr = Compose([Scale(args.img_cols),  # resize longer side of an image to the defined size
                       # zero pad remaining regions
                       CenterPadding([args.img_rows, args.img_cols]),
                       RandomHorizontallyFlip(),  # random horizontal flip
                       RandomRotate(180)
                ])  # ramdom rotation

data_aug_te = Compose([Scale(args.img_cols),
                       CenterPadding([args.img_rows, args.img_cols])])

result_path = 'results_' + args.dataset
if not os.path.exists(result_path):
    os.makedirs(result_path)
outimg_path = "./img_log_" + args.dataset
if not os.path.exists(outimg_path):
    os.makedirs(outimg_path)

f_loss = open(os.path.join(result_path, "log_loss.txt"), 'w')
f_loss.write('Network type: %s\n' % args.arch)
f_loss.write('Learning rate: %05f\n' % args.lr)
f_loss.write('batch-size: %s\n' % args.batch_size)
f_loss.write('img_cols: %s\n' % args.img_cols)
f_loss.write('Augmentation type: flip, centercrop\n\n')
f_loss.close()

f_iou = open(os.path.join(result_path, "log_acc.txt"), 'w')
f_iou.close()

f_iou = open(os.path.join(result_path, "log_val_acc.txt"), 'w')
f_iou.close()

# Data
data_loader = get_loader(args.dataset)
data_path = get_data_path(args.dataset)

tr_loader = data_loader(data_path, args.exp, is_transform=True, split='train', img_size=(
    args.img_rows, args.img_cols), augmentations=data_aug_tr, n_style=args.n_style)
te_loader = data_loader(data_path, args.exp, is_transform=True, split='test', img_size=(
    args.img_rows, args.img_cols), augmentations=data_aug_te, n_style=args.n_style)

trainloader = DataLoader(tr_loader, batch_size=args.batch_size,
                         num_workers=args.workers, shuffle=True, pin_memory=False)
testloader = DataLoader(te_loader, batch_size=args.batch_size,
                        num_workers=args.workers, shuffle=True, pin_memory=False)

# define model or load model
net = get_model(args.arch, n_style=args.n_style, style_out_channel=args.style_out_channel)
net.to(device)

num_classes = 43
discriminator = DiscriminatorWithClassifier(in_channel=3, num_classes=num_classes).to(device)

if args.resume is not None:
    checkpoint = torch.load(args.resume)
    discriminator.load_state_dict(checkpoint['discriminator'])
    net.load_state_dict(checkpoint['generator'])
    eps = checkpoint['epoch']
else:
    eps = 1

print("Current epoch: ", eps)

reconstruction_function = nn.BCELoss()
reconstruction_function.reduction = 'sum'
def loss_function(recon_x, x, mu, logvar):
    BCE = reconstruction_function(recon_x, x)
    #KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    #KLD = torch.sum(KLD_element).mul_(-0.5)
    return BCE #+ KLD


# Construct optimiser
optimizer_G = optim.Adam(net.parameters(), lr=args.lr)  # 1e-4
optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr)

num_train = len(tr_loader.targets)
num_test = len(te_loader.targets)
batch_iter = math.ceil(num_train/args.batch_size)
batch_iter_test = math.ceil(num_test/args.batch_size)

# Discriminator output patch shape
patch = (1, args.img_cols // 2**4, args.img_cols // 2**4)

# Loss criterion
criterion_GAN = torch.nn.MSELoss().to(device)
#criterion_GAN = torch.nn.BCELoss().to(device)
criterion_pixel = torch.nn.L1Loss().to(device)
criterion_ce = torch.nn.CrossEntropyLoss().to(device)
criterion_attr = torch.nn.BCEWithLogitsLoss().to(device)
#criterion_attr = torch.nn.BCELoss().to(device)

real_label = 1.0
fake_label = 0.0

def train(e):
    n_classes = tr_loader.n_classes
    n_classes_te = te_loader.n_classes

    print('start train epoch: %d' % e)
    net.train()
    discriminator.train()

    for i, (input, target, template, style) in enumerate(trainloader):

        valid = torch.ones((input.size(0), *patch)).to(device)
        fake = torch.zeros((input.size(0), *patch)).to(device)

        # valid = torch.full((input.size(0),), real_label, dtype=torch.float, device=device)
        # fake = torch.full((input.size(0),), fake_label, dtype=torch.float, device=device)

        input, template = input.to(device), template.to(device)
        style = style.to(device)

        target = torch.squeeze(target)
        target = target.to(device)
        target = F.one_hot(target, num_classes=num_classes).float()


        optimizer_G.zero_grad()

        recon, mu, logvar, input_stn = net(input, style)

        pred_recon, recon_class = discriminator(recon)
        loss_GAN = args.lambda_GAN * criterion_GAN(pred_recon, valid)
        # Reconstruction loss
        loss_pixel = args.lambda_l1 * loss_function(recon, template, mu, logvar)

        loss_attr = args.lambda_attr * criterion_attr(recon_class, target)

        loss_G = loss_GAN + loss_pixel + loss_attr

        loss_G.backward()
        optimizer_G.step()

        #### Training discriminator ####
        optimizer_D.zero_grad()
        pred_real, real_class = discriminator(input)
        loss_real = criterion_GAN(pred_real, valid)

        loss_attr_D = torch.zeros(1).to(device)
        loss_attr_D += criterion_attr(real_class, target)

        # Criterion GAN
        recon, mu, logvar, input_stn = net(input, style)
        pred_recon, recon_class = discriminator(recon)
        loss_fake = criterion_GAN(pred_recon, fake)
        loss_D = loss_real + loss_fake + loss_attr_D

        loss_D.backward()
        optimizer_D.step()
        
        if (i > 0) and ((i % 5) == 0):
            print('Epoch:%d  Batch:%d/%d  loss G:%08f loss D:%08f' %
                (e, i, batch_iter, loss_G.data/input.numel(), loss_D.data))

            f_loss = open(os.path.join(result_path, "log_loss.txt"), 'a')
            f_loss.write('Epoch:%d  Batch:%d/%d  loss:%08f\n' %
                        (e, i, batch_iter, loss_G.data/input.numel()))
            f_loss.close()

        if i < 1 and (e % save_epoch == 0):
            out_folder = "%s/Epoch_%d_train" % (outimg_path, e)
            out_root = Path(out_folder)
            if not out_root.is_dir():
                os.mkdir(out_root)

            torchvision.utils.save_image(
                input.data, '{}/batch_{}_data.jpg'.format(out_folder, i), nrow=8, padding=2)
            torchvision.utils.save_image(
                input_stn.data, '{}/batch_{}_data_stn.jpg'.format(out_folder, i), nrow=8, padding=2) 
            torchvision.utils.save_image(
                recon.data, '{}/batch_{}_recon.jpg'.format(out_folder, i), nrow=8, padding=2)
            torchvision.utils.save_image(
                template.data, '{}/batch_{}_target.jpg'.format(out_folder, i), nrow=8, padding=2)

    if e % save_epoch == 0:
        class_target = torch.LongTensor(list(range(n_classes)))
        class_template, tr_style = tr_loader.load_template(class_target)
        class_template = class_template.to(device)
        tr_style = tr_style.to(device)
        with torch.no_grad():
            class_recon, class_mu, class_logvar, _ = net(class_template, tr_style)

        torchvision.utils.save_image(
            class_template.data, '{}/templates.jpg'.format(out_folder), nrow=8, padding=2)
        torchvision.utils.save_image(
            class_recon.data, '{}/templates_recon.jpg'.format(out_folder), nrow=8, padding=2)


def score_NN(pred, class_feature, label, n_classes):

    sample_correct = torch.zeros(n_classes)
    sample_all = torch.zeros(n_classes)
    sample_rank = torch.zeros(n_classes, n_classes)  # rank per class
    sample_distance = torch.ones(pred.shape[0], n_classes)*math.inf

    pred = pred.data.cpu()  # batch x latent size
    class_feature = class_feature.data.cpu()  # n_classes x latent size
    label = label.numpy()

    mean = class_feature.mean(0)
    pred -= mean
    class_feature -= mean
    pred = pred/torch.norm(pred, p=2, dim=1, keepdim=True)
    class_feature = class_feature/torch.norm(class_feature, p=2, dim=1, keepdim=True)

    for i in range(n_classes):
        cls_feat = class_feature[i, :]
        cls_mat = cls_feat.repeat(pred.shape[0], 1)
        # euclidean distance
        sample_distance[:, i] = torch.norm(pred - cls_mat, p=2, dim=1)

    sample_distance = sample_distance.cpu().numpy()
    indices = np.argsort(sample_distance, axis=1)  # sort ascending order

    for i in range(indices.shape[0]):
        rank = np.where(indices[i, :] == label[i])[0][0]  # find rank
        sample_rank[label[i]][rank:] += 1  # update rank
        sample_all[label[i]] += 1  # count samples per class
        if rank == 0:
            # count rank 1 (correct classification)
            sample_correct[label[i]] += 1

    return sample_correct, sample_all, sample_rank

mean_scores = []
mean_rank = []

def test(e, best_acc):
    n_classes = te_loader.n_classes
    print('start test epoch: %d' % e)
    net.eval()
    accum_all = torch.zeros(n_classes)
    rank_all = torch.zeros(n_classes, n_classes)  # rank per class
    accum_class = torch.zeros(n_classes)

    # get template latent z
    class_target = torch.LongTensor(list(range(n_classes)))
    class_template, te_style = te_loader.load_template(class_target)
    class_template = class_template.to(device)
    # te_style = te_style.to(device)

    _, tr_style = tr_loader.load_template(class_target)
    tr_style = tr_style.to(device)

    with torch.no_grad():
        class_recon, class_mu, class_logvar, _ = net(class_template, tr_style)

    for i, (input, target, template, style) in enumerate(testloader):

        target = torch.squeeze(target)
        input, template = input.to(device), template.to(device)
        #style = style.to(device)

        #### Loading original style
        _, tr_style = tr_loader.load_template(target)
        tr_style = tr_style.to(device)

        with torch.no_grad():
            recon, mu, logvar, input_stn = net(input, tr_style)

        sample_correct, sample_all, sample_rank = score_NN(
            mu, class_mu, target, n_classes)
        accum_class += sample_correct
        accum_all += sample_all
        rank_all = rank_all + sample_rank  # [class_id, topN]

        print('Epoch:%d  Batch:%d/%d  processing...' % (e, i, batch_iter_test))

        if i < 1 and (e % save_epoch == 0):
            out_folder = "%s/Epoch_%d_test" % (outimg_path, e)
            out_root = Path(out_folder)
            if not out_root.is_dir():
                os.mkdir(out_root)

            torchvision.utils.save_image(
                input.data, '{}/batch_{}_data.jpg'.format(out_folder, i), nrow=8, padding=2)
            torchvision.utils.save_image(
                input_stn.data, '{}/batch_{}_data_stn.jpg'.format(out_folder, i), nrow=8, padding=2)
            torchvision.utils.save_image(
                recon.data, '{}/batch_{}_recon.jpg'.format(out_folder, i), nrow=8, padding=2)
            torchvision.utils.save_image(
                template.data, '{}/batch_{}_target.jpg'.format(out_folder, i), nrow=8, padding=2)

    if e % save_epoch == 0:
        torchvision.utils.save_image(
            class_template.data, '{}/templates.jpg'.format(out_folder), nrow=8, padding=2)
        torchvision.utils.save_image(
            class_recon.data, '{}/templates_recon.jpg'.format(out_folder), nrow=8, padding=2)

    acc_all = accum_class.sum() / accum_all.sum()
    acc_cls = torch.div(accum_class, accum_all)

    rank_sample_avg = rank_all.sum(0) / accum_all.sum()  # [class_id, topN]
    rank_cls = torch.div(rank_all, torch.transpose(
        accum_all.expand_as(rank_all), 0, 1))
    rank_cls_avg = torch.mean(rank_cls, dim=0)

    # write result part
    acc_trcls = torch.gather(acc_cls, 0, te_loader.tr_class)
    acc_tecls = torch.gather(acc_cls, 0, te_loader.te_class)

    print('========epoch(%d)=========' % e)
    print('Seen Classes')
    for i, class_acc in enumerate(acc_trcls):
        print('cls:%d  acc:%02f' % (te_loader.tr_class[i], class_acc))
    print('Unseen Classes')
    for i, class_acc in enumerate(acc_tecls):
        print('cls:%d  acc:%02f' % (te_loader.te_class[i], class_acc))
    print('====================================')
    print('acc_avg:%02f' % acc_all)
    print('acc_cls:%02f' % acc_cls.mean())
    print('acc_trcls:%02f' % acc_trcls.mean())
    print('acc_tecls:%02f' % acc_tecls.mean())
    print('rank sample avg: %02f' % rank_sample_avg.mean())
    print('rank cls avg: %02f' % rank_cls_avg.mean())
    print('====================================')

    f_iou = open(os.path.join(result_path, "log_acc.txt"), 'a')
    f_iou.write('epoch(%d), acc_cls: %04f  acc_trcls: %04f  acc_tecls: %04f  acc_all: %04f  top3: %04f  top5: %04f\n' % (
        e, acc_cls.mean(), acc_trcls.mean(), acc_tecls.mean(), acc_all, rank_sample_avg[2], rank_sample_avg[4]))
    f_iou.close()

    # if best_acc < acc_all: # update best score
    #   best_acc = acc_all
    if best_acc < acc_tecls.mean():  # update best score

        f_iou_class = open(os.path.join(result_path, "best_iou.txt"), 'w')
        f_rank = open(os.path.join(result_path, "best_rank.txt"), 'w')

        checkpoint = {'discriminator': discriminator.state_dict(),
                      'generator': net.state_dict(),
                      'epoch': e}
        torch.save(checkpoint, os.path.join('%s_testBest_net_%d.pth' % (args.dataset, e)))

        best_acc = acc_tecls.mean()
        f_iou_class.write('Best score epoch:  %d\n' % e)
        f_iou_class.write('acc cls: %.4f  acc all: %.4f  rank mean: %.4f \n' % (
            acc_cls.mean(), acc_all, rank_all.mean()))
        f_iou_class.write('acc tr cls: %.4f  acc te cls: %.4f\n' %
                          (acc_trcls.mean(), acc_tecls.mean()))
        f_iou_class.write('top3: %.4f  top5: %.4f\n' %
                          (rank_sample_avg[2], rank_sample_avg[4]))

        f_iou_class.write('\nSeen classes\n')
        for i, class_acc in enumerate(acc_trcls):
            f_iou_class.write('cls:%d  acc:%02f\n' %
                              (te_loader.tr_class[i], class_acc))
        f_iou_class.write('\nUnseen classes\n')
        for i, class_acc in enumerate(acc_tecls):
            f_iou_class.write('cls:%d  acc:%02f\n' %
                              (te_loader.te_class[i], class_acc))
        f_iou_class.close()

        for i, rank_acc in enumerate(rank_sample_avg):
            f_rank.write('rank sample %d: %.4f\n' % (i+1, rank_acc))
        f_rank.write('\n')
        for i, rank_acc in enumerate(rank_cls_avg):
            f_rank.write('rank cls %d: %.4f\n' % (i+1, rank_acc))
        f_rank.close()

    # Save weights and scores
    # if e % 100 == 0:
    #   pass
        # torch.save(net.state_dict(), os.path.join('flickr2belga_latest_net.pth'))

    # Plot scores
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
    for e in range(eps, args.epochs + 1):
        train(e)
        best_acc = test(e, best_acc)