from torch.nn.modules import padding
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm

from collections import namedtuple

n_cont_downsample = 2
n_cont_res = 4
n_upsample = 2

class AdaIN(nn.Module):
    def __init__(self):
        super().__init__()

    def mu(self, x):
        """ Takes a (n,c,h,w) tensor as input and returns the average across
        it's spatial dimensions as (h,w) tensor [See eq. 5 of paper]"""
        return torch.sum(x,(2,3))/(x.shape[2]*x.shape[3])

    def sigma(self, x):
        """ Takes a (n,c,h,w) tensor as input and returns the standard deviation
        across it's spatial dimensions as (h,w) tensor [See eq. 6 of paper] Note
        the permutations are required for broadcasting"""
        return torch.sqrt((torch.sum((x.permute([2,3,0,1])-self.mu(x)).permute([2,3,0,1])**2,(2,3))+0.000000023)/(x.shape[2]*x.shape[3]))

    def forward(self, x, y):
        """ Takes a content embeding x and a style embeding y and changes
        transforms the mean and standard deviation of the content embedding to
        that of the style. [See eq. 8 of paper] Note the permutations are
        required for broadcasting"""
        return (self.sigma(y)*((x.permute([2,3,0,1])-self.mu(x))/self.sigma(x)) + self.mu(y)).permute([2,3,0,1])

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()

        self.res_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(out_channels),
        )
    
    def forward(self, x):
        return x + self.res_block(x)

class ContentLoss(nn.Module):
    def __init__(self, target, ):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
    
    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
    
    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

class NonLocalBlock(nn.Module):
    def __init__(self, out_channels, sub_sample=True, is_bn=True):
        super(NonLocalBlock, self).__init__()

        self.out_channels = out_channels

        self.g = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding_mode='reflect'),
            nn.MaxPool2d(out_channels)
        )

        self.phi = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding_mode='reflect'),
            nn.MaxPool2d(out_channels)
        )

        self.theta = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding_mode='reflect')
        
        w_temp = [nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding_mode='reflect')]
        if is_bn:
            w_temp += [nn.BatchNorm2d(out_channels)]
        self.w = nn.Sequential(*w_temp)

        self.softmax = nn.Softmax(dim=-1)    
    
    def forward(self, x):
        batch, _, height, width = x.shape

        g_x = self.g(x).view(batch, self.out_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch, self.out_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(x).view(batch, self.out_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        f_softmax = self.softmax(f)

        y = torch.matmul(f_softmax, g_x)
        y = y.view(batch, height, width, self.out_channels)

        w_y = self.w(y)
        z = x + w_y
        return z       

class ContentEncoder(nn.Module):
    def __init__(self, channels=3, nsf=64):
        super(ContentEncoder, self).__init__()

        self.head = nn.Sequential(
            nn.Conv2d(channels, nsf, kernel_size=7, stride=1, padding=3, padding_mode='reflect'),
            nn.InstanceNorm2d(nsf)
        )

        self.relu = nn.ReLU(True)

        self.middle1 = nn.Sequential(
            nn.Conv2d(nsf, nsf*2, kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(nsf)
        )

        self.middle2 = nn.Sequential(
            nn.Conv2d(nsf*2, nsf*4, kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(nsf)
        )

        self.non_block1 = NonLocalBlock(nsf*4)
        self.non_block2 = NonLocalBlock(nsf*4)

        self.resblock = ResBlock(nsf*4, nsf*4)

    def forward(self, x):
        content_layers = []

        out = self.head(x)
        content_layers.append(out)
        out = self.relu(out)

        out = self.middle1(out)
        content_layers.append(out)
        out = self.relu(out)
        out = self.middle2(out)
        content_layers.append(out)
        out = self.relu(out)

        for _ in range(n_cont_res):
            out = self.resblock(out)

        return out, content_layers

class StyleEncoder(nn.Module):
    def __init__(self, channels=3, nsf=64):
        super(StyleEncoder, self).__init__()

        self.head = nn.Sequential(
            nn.Conv2d(channels, nsf, kernel_size=7, stride=1, padding=3, padding_mode='reflect'),
            nn.InstanceNorm2d(nsf)
        )

        self.relu = nn.ReLU(True)

        self.middle1 = nn.Sequential(
            nn.Conv2d(nsf, nsf*2, kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(nsf)
        )

        self.middle2 = nn.Sequential(
            nn.Conv2d(nsf*2, nsf*4, kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(nsf)
        )

        self.non_block1 = NonLocalBlock(nsf*4)
        self.non_block2 = NonLocalBlock(nsf*4)

        self.resblock = ResBlock(nsf*4, nsf*4)

    def forward(self, x):
        style_layers = []

        out = self.head(x)
        style_layers.append(out)
        out = self.relu(out)

        out = self.middle1(out)
        style_layers.append(out)
        out = self.relu(out)
        out = self.middle2(out)
        style_layers.append(out)
        out = self.relu(out)

        for _ in range(n_cont_res):
            out = self.resblock(out)
            style_layers.append(out)

        return out, style_layers

class Decoder(nn.Module):
    def __init__(self, channels, ndf=64):
        super(Decoder, self).__init__()

        self.ada = AdaIN()

        self.up_sample = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.conv1 = nn.Conv2d(ndf*4, ndf*2, kernel_size=5, stride=1, padding=2, padding_mode='reflect')
        self.norm1 = nn.InstanceNorm2d(ndf*2)
        
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(ndf*2, ndf, kernel_size=5, stride=1, padding=2, padding_mode='reflect')
        self.norm2 = nn.InstanceNorm2d(ndf)

        self.conv3 = nn.Conv2d(ndf, channels, kernel_size=7, stride=1, padding=3, padding_mode='reflect')

        self.tail = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=7, stride=1, padding=3, padding_mode='reflect'),
            nn.Conv2d(channels, channels, kernel_size=7, stride=1, padding=3, padding_mode='reflect')
        )

        self.tanh = nn.Tanh()
    
    def forward(self, content, style=None, style_layers=None, stylize=False):
        if stylize:
            x = self.ada(content, style)
        else:
            x = content
        
        if stylize:
            x = self.ada(x, style_layers[2])
        x = self.up_sample(x)
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.up_sample(x)
        x = self.relu(self.norm2(self.conv2(x)))

        x = self.conv3(x)
        if stylize:
            x = self.tail(x)
        
        return self.tanh(x)

class Discriminator(nn.Module):
    def __init__(self, channels, n_scale=3, n_dis=3, ndf=64):
        super(Discriminator, self).__init__()

        self.n_scale = n_scale
        self.n_dis = n_dis

        self.lrelu = nn.LeakyReLU(0.2)

        #### source: https://discuss.pytorch.org/t/how-to-access-to-a-layer-by-module-name/83797/5
        for scale in range(self.n_scale):
            self.ndf = ndf
            self.add_module(f"scale_{str(scale)}_conv0", spectral_norm(nn.Conv2d(channels, self.ndf, kernel_size=4, stride=2, padding=1, padding_mode='reflect')))
            for i in range(1, self.n_dis):
                self.add_module(f"scale_{str(scale)}_conv{str(i)}", spectral_norm(nn.Conv2d(self.ndf, self.ndf*2, kernel_size=4, stride=2, padding=1, padding_mode='reflect')))
                self.ndf *= 2
            self.add_module(f"scale_{str(scale)}_D_logit", spectral_norm(nn.Conv2d(self.ndf, 1, kernel_size=1, stride=1)))
        
    def forward(self, input):
        D_logit = []

        for scale in range(self.n_scale):
            conv = self._modules[f"scale_{str(scale)}_conv0"]
            x = self.lrelu(conv(input))
            #print(scale, conv, x.shape)
            for i in range(1, self.n_dis):
                conv = self._modules[f"scale_{str(scale)}_conv{str(i)}"]
                x = conv(x)
                #print(i, conv, x.shape)
            conv = self._modules[f"scale_{str(scale)}_D_logit"]
            x = conv(x)
            #print(scale, conv, x.shape)

            D_logit.append(x)

            input = self.down_sample(input)

        return D_logit
            
    
    def down_sample(self, x):
        pool = nn.AvgPool2d(kernel_size=3, stride=2)
        return pool(x)

LossOutput = namedtuple("LossOutput", ["relu1_2", "relu2_2", "relu3_3", "relu4_3"])

#### source: https://towardsdatascience.com/pytorch-implementation-of-perceptual-losses-for-real-time-style-transfer-8d608e2e9902
class PerceptualContentLossVgg16(nn.Module):
    def __init__(self, vgg_model):
        super(PerceptualContentLossVgg16, self).__init__()
        self.vgg_layers = vgg_model.features
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3",
            '22': "relu4_3"
        }

    def forward(self, x, y):
        loss_f = 0
        for name, module in self.vgg_layers._modules.items():
            x, y = module(x), module(y)
            if name in self.layer_name_mapping:
                loss_f += (torch.mean((x - y) ** 2))
        return loss_f

#### source: https://discuss.pytorch.org/t/implement-total-variation-loss-in-pytorch/55574
def total_variation_loss(img, weight=1):
     bs_img, c_img, h_img, w_img = img.size()
     tv_h = torch.pow(img[:,:,1:,:]-img[:,:,:-1,:], 2).sum()
     tv_w = torch.pow(img[:,:,:,1:]-img[:,:,:,:-1], 2).sum()
     return weight*(tv_h+tv_w)/(bs_img*c_img*h_img*w_img)


if __name__ == '__main__':
    # model = models.vgg16(pretrained=True)

    # vgg = models.vgg19(pretrained=True)
    # vgg = nn.Sequential(*list(vgg.features.children())[:21])

    # x = torch.randn(1, 3, 64, 64)

    # output = model(x)
    # print(output.shape) # [1, 1000]
    # output = vgg(x)
    # print(output.shape) # [1, 512, 8, 8]

    # res = ResBlock(64, 64)
    # x = torch.randn(1, 64, 64, 64)
    # out = res(x)
    # print(out.shape)  # [1, 64, 64, 64]


    # non_lo = NonLocalBlock(out_channels=64, is_bn=False)
    # x = torch.randn(1, 64, 64, 64)
    # out = non_lo(x)
    # print(out.shape)  # [1, 64, 64, 64]

    ################ Testing Encoder ##################
    x = torch.randn(1, 3, 64, 64)
    content_enc = ContentEncoder(channels=3, nsf=64)
    content, content_layers = content_enc(x)
    print(content.shape)    # [1, 256, 16, 16]

    style_enc = StyleEncoder(channels=3, nsf=64)
    style, style_layers = style_enc(x)

    dec = Decoder(channels=3, ndf=64)
    out = dec(style, content, style_layers)
    print(out.shape)    # [1, 3, 64, 64]

    disc = Discriminator(channels=3)
    logit = disc(out)
    print("Discriminator output length: ", len(logit))   # 3 x multi-scale output

    perceptual_content_loss = PerceptualContentLossVgg16(model)
    perceptual_loss_A_1 = perceptual_content_loss(x, out)
    print("Perceptual Loss: ", perceptual_loss_A_1)
