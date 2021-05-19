#### source: https://github.com/tamarott/SinGAN/blob/1b4906b9afbd30bf32ea3d3f29ca6c5f4a2a223c/SinGAN/training.py
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import SinGAN.imresize as imresize

class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, stride):
        super(ConvBlock,self).__init__()
        self.add_module('conv',nn.Conv2d(in_channel ,out_channel,kernel_size=ker_size,stride=stride,padding=padd)),
        self.add_module('norm',nn.BatchNorm2d(out_channel)),
        self.add_module('LeakyRelu',nn.LeakyReLU(0.2, inplace=True))

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
   
class WDiscriminator(nn.Module):
    def __init__(self, opt):
        super(WDiscriminator, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = int(opt.nfc)
        self.head = ConvBlock(opt.nc_im,N,opt.ker_size,opt.padd_size,1)
        self.body = nn.Sequential()
        for i in range(opt.num_layer-2):
            N = int(opt.nfc/pow(2,(i+1)))
            block = ConvBlock(max(2*N,opt.min_nfc),max(N,opt.min_nfc),opt.ker_size,opt.padd_size,1)
            self.body.add_module('block%d'%(i+1),block)
        self.tail = nn.Conv2d(max(N,opt.min_nfc),1,kernel_size=opt.ker_size,stride=1,padding=opt.padd_size)

    def forward(self,x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x


class GeneratorConcatSkip2CleanAdd(nn.Module):
    def __init__(self, opt):
        super(GeneratorConcatSkip2CleanAdd, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = opt.nfc
        self.head = ConvBlock(opt.nc_im,N,opt.ker_size,opt.padd_size,1) #GenConvTransBlock(opt.nc_z,N,opt.ker_size,opt.padd_size,opt.stride)
        self.body = nn.Sequential()
        for i in range(opt.num_layer-2):
            N = int(opt.nfc/pow(2,(i+1)))
            block = ConvBlock(max(2*N,opt.min_nfc),max(N,opt.min_nfc),opt.ker_size,opt.padd_size,1)
            self.body.add_module('block%d'%(i+1),block)
        self.tail = nn.Sequential(
            nn.Conv2d(max(N,opt.min_nfc),opt.nc_im,kernel_size=opt.ker_size,stride =1,padding=opt.padd_size),
            nn.Tanh()
        )
    def forward(self,x,y):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        ind = int((y.shape[2]-x.shape[2])/2)
        y = y[:,:,ind:(y.shape[2]-ind),ind:(y.shape[3]-ind)]
        return x+y

def draw_concat(Gs, Zs, reals, NoiseAmp, in_s, mode, m_noise, m_image, opt):
    G_z = in_s
    if len(Gs) > 0:
        if mode == 'rand':
            count = 0
            if opt.mode == 'animation_train':
                pad_noise = 0
            for G, Z_opt, real_curr, real_next, noise_amp in zip(Gs, Zs, reals, reals[1:], NoiseAmp):
                if count == 0:
                    z = functions.generate_noise([1, Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise], device=opt.device)
                    z = z.expand(1, 3, z.shape[2], z.shape[3])
                else:
                    z = functions.generate_noise([opt.nc_z, Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise])
                z = m_noise(z)
                G_z = G_z[:,:,0:real_curr.shape[2],0:real_curr.shape[3]]
                G_z = m_image(G_z)
                z_in = noise_amp*z + G_z
                G_z = G(z_in.detach(), G_z)
                G_z = imresize(G_z, 1/opt.scale_factor, opt)
                G_z = G_z[:, :, 0:real_next.shape[2], 0:real_next.shape[3]]
                count += 1
        
        if mode == 'rec':
            count = 0
            for G,Z_opt,real_curr,real_next,noise_amp in zip(Gs,Zs,reals,reals[1:],NoiseAmp):
                G_z = G_z[:, :, 0:real_curr.shape[2], 0:real_curr.shape[3]]
                G_z = m_image(G_z)
                z_in = noise_amp*Z_opt+G_z
                G_z = G(z_in.detach(),G_z)
                G_z = imresize(G_z,1/opt.scale_factor,opt)
                G_z = G_z[:,:,0:real_next.shape[2],0:real_next.shape[3]]
                #if count != (len(Gs)-1):
                #    G_z = m_image(G_z)
                count += 1
    return G_z



if __name__ == "__main__":
    import easydict
    import SinGAN.functions as functions

    real = torch.randn(2,3,64,64)
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    opt = easydict.EasyDict({
        "nfc": 32,
        "nc_im": 3,
        "ker_size": 3,
        "padd_size": 0,
        "num_layer": 5,
        "min_nfc": 32,
        "nc_z": 3,
        "nzx": real.shape[2],
        "nzy": real.shape[3],
        "mode": "train",
        "noise_amp": 0.1,
        "lambda_grad": 0.1,
    })

    fixed_noise = functions.generate_noise([opt.nc_z, opt.nzx, opt.nzy])
    #print(fixed_noise.shape)    # [1, 3, 64, 64]
    z_opt = torch.full(fixed_noise.shape, 0, dtype=torch.float32)
    #print(z_opt.shape)          # [1, 3, 64, 64]

    pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2)
    pad_image = int(((opt.ker_size - 1) * opt.num_layer) / 2)
    m_noise = nn.ZeroPad2d(int(pad_noise))
    m_image = nn.ZeroPad2d(int(pad_image))
    z_opt = m_noise(z_opt)
    #print(z_opt.shape)          # [1, 3, 74, 74]

    if (Gs == []) & (opt.mode != 'SR_train'):
        z_opt = functions.generate_noise([1, opt.nzx, opt.nzy])
        z_opt = m_noise(z_opt.expand(1, 3, opt.nzx, opt.nzy))
        noise_ = functions.generate_noise([1, opt.nzx, opt.nzy])
        noise_ = m_noise(noise_.expand(1, 3, opt.nzx, opt.nzy))
        #print(z_opt.shape)  # [1, 3, 74, 74]
        #print(noise_.shape) # [1, 3, 74, 74]

    #### View output shape from discriminator ####
    netD = WDiscriminator(opt)
    out_d = netD(z_opt)
    #print(out_d.shape)          # [1, 1, 64, 64]
    errD_real = -out_d.mean()
    #print(errD_real.item())     # 0.3413020670413971

    #### Train with fake
    if (Gs == []) & (opt.mode != 'SR_train'):
        prev = torch.full([1, opt.nc_z, opt.nzx, opt.nzy], 0, dtype=torch.float32)
        in_s = prev
        prev = m_image(prev)
        z_prev = torch.full([1, opt.nc_z, opt.nzx, opt.nzx, opt.nzy], 0, dtype=torch.float32)
        z_prev = m_noise(z_prev)
        opt.noise_amp =1
    
    prev = draw_concat(Gs, Zs, reals, NoiseAmp, in_s, 'rand', m_noise, m_image, opt)
    print(prev.shape)

    netG = GeneratorConcatSkip2CleanAdd(opt)
    #print(netG)
    
    if (Gs == []) & (opt.mode != 'SR_train'):
        noise = noise_
    else:
        noise = opt.noise_amp * noise_ + prev
    
    fake = netG(noise.detach(), prev)
    output = netD(fake.detach())
    errD_fake = output.mean()
    errD_fake.backward(retain_graph=True)
    D_G_z = output.mean().item()

    gradient_penalty = functions.calc_gradient_penalty(netD, real, fake, opt.lambda_grad, device)
    gradient_penalty.backward()

    errD = errD_real + errD_fake + gradient_penalty
    print(errD)

