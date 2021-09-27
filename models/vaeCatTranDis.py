# This code is modified from the repository
# https://github.com/bhpfelix/Variational-Autoencoder-PyTorch

from torch.autograd import Variable
import torch.nn as nn
import torch
import torch.nn.functional as F

from models.nn import LinearBlock, Conv2dBlock, ConvTranspose2dBlock

MAX_DIM = 64 * 16  # 1024

class View(nn.Module):
    def __init__(self):
        super(View, self).__init__()
   
    def forward(self, x):
        numel = x.numel() / x.shape[0]
        return x.view(-1, int(numel)) 

def convNoutput(convs, input_size): # predict output size after conv layers
    input_size = int(input_size)
    input_channels = convs[0][0].weight.shape[1] # input channel
    output = torch.Tensor(1, input_channels, input_size, input_size)
    with torch.no_grad():
        for conv in convs:
            output = conv(output)
    return output.numel(), output.shape

class stn(nn.Module):
    def __init__(self, input_channels, input_size, params):
        super(stn, self).__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
                    nn.Conv2d(input_channels, params[0], kernel_size=5, stride=1, padding=2),
                    nn.ReLU(True),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                    )
        self.conv2 = nn.Sequential(
                    nn.Conv2d(params[0], params[1], kernel_size=5, stride=1, padding=2),
                    nn.ReLU(True),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                    )

        out_numel, out_size = convNoutput([self.conv1, self.conv2], input_size/2)
        # set fc layer based on predicted size
        self.fc = nn.Sequential(
                View(),
                nn.Linear(out_numel, params[2]),
                nn.ReLU()
                )
        self.classifier = classifier = nn.Sequential(
                View(),
                nn.Linear(params[2], 6) # affine transform has 6 parameters
                )
        # initialize stn parameters (affine transform)
        self.classifier[1].weight.data.fill_(0)
        self.classifier[1].bias.data = torch.FloatTensor([1, 0, 0, 0, 1, 0])

    def localization_network(self, x):
        x = self.maxpool(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc(x)
        x = self.classifier(x)
        return x


    def forward(self, x):
        theta = self.localization_network(x)
        theta = theta.view(-1,2,3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x


class VAECatTranDis(nn.Module):
    def __init__(self, nc, input_size, latent_variable_size=300, cnn_chn=[100, 150, 250], 
                param1=None, param2=None, param3=None,
                n_style=4):
        super(VAECatTranDis, self).__init__()

        self.cnn_chn = cnn_chn
        self.param1 = param1
        self.param2 = param2
        self.param3 = param3

        self.input_size = input_size
        self.nc = nc
        #self.latent_variable_size = latent_variable_size

        # encoder
        self.e1 = nn.Conv2d(nc, self.cnn_chn[0], 7, 2, 3) # inchn, outchn, kernel, stride, padding, dilation, groups
        self.bn1 = nn.BatchNorm2d(self.cnn_chn[0])

        self.e2 = nn.Conv2d(self.cnn_chn[0], self.cnn_chn[1], 4, 2, 1) # 1/4
        self.bn2 = nn.BatchNorm2d(self.cnn_chn[1])

        self.e3 = nn.Conv2d(self.cnn_chn[1], self.cnn_chn[2], 4, 2, 1) # 1/8
        self.bn3 = nn.BatchNorm2d(self.cnn_chn[2])

        self.fc1 = nn.Linear(int(input_size/8*input_size/8*self.cnn_chn[2]), latent_variable_size)
        self.fc2 = nn.Linear(int(input_size/8*input_size/8*self.cnn_chn[2]), latent_variable_size)

        # decoder
        self.d1 = nn.Linear(latent_variable_size, int(input_size/8*input_size/8*self.cnn_chn[2]))

        self.up1 = nn.UpsamplingNearest2d(scale_factor=2) # 8 -> 16
        self.pd1 = nn.ReplicationPad2d(1)
        self.d2 = nn.Conv2d(self.cnn_chn[2]*2, self.cnn_chn[1], 3, 1)
        self.bn6 = nn.BatchNorm2d(self.cnn_chn[1], 1.e-3)

        self.up2 = nn.UpsamplingNearest2d(scale_factor=2) # 16 -> 32
        self.pd2 = nn.ReplicationPad2d(1)
        self.d3 = nn.Conv2d(self.cnn_chn[1]*2, self.cnn_chn[0], 3, 1)
        self.bn7 = nn.BatchNorm2d(self.cnn_chn[0], 1.e-3)

        self.up3 = nn.UpsamplingNearest2d(scale_factor=2) # 32 -> 64
        self.pd3 = nn.ReplicationPad2d(1)
        self.d4 = nn.Conv2d(self.cnn_chn[0]*2, 3, 3, 1)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        if param1 is not None:
            self.stn1 = stn(3, self.input_size, param1)
        if param2 is not None:
            self.stn2 = stn(self.cnn_chn[0], self.input_size/2, param2)
        if param3 is not None:
            self.stn3 = stn(self.cnn_chn[1], self.input_size/4, param3)

        self.style_enc = StyleEncoder(in_channel=nc, n_style=n_style, cnn_chn=cnn_chn)


    def encode(self, x):
        if self.param1 is not None:
            x = self.stn1(x)

        h1 = self.leakyrelu(self.bn1(self.e1(x)))
        if self.param2 is not None:
            h1 = self.stn2(h1)

        h2 = self.leakyrelu(self.bn2(self.e2(h1)))
        if self.param3 is not None:
            h2 = self.stn3(h2)

        h3 = self.leakyrelu(self.bn3(self.e3(h2)))
        h4 = h3.view(-1, int(self.input_size/8*self.input_size/8*self.cnn_chn[2]))

        return self.fc1(h4), self.fc2(h4), x

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z, encoded_style):

        h3_style, h2_style, h1_style = encoded_style

        h1 = self.relu(self.d1(z))
        # h1 = h1.view(-1, self.ngf*8*2, 4, 4)
        h1 = h1.view(-1, self.cnn_chn[2], int(self.input_size/8), int(self.input_size/8))
        h1 = torch.cat([h1, h3_style], dim=1)
        h2 = self.leakyrelu(self.bn6(self.d2(self.pd1(self.up1(h1)))))
        h2 = torch.cat([h2, h2_style], dim=1)
        h3 = self.leakyrelu(self.bn7(self.d3(self.pd2(self.up2(h2)))))
        h3 = torch.cat([h3, h1_style], dim=1)
        return self.sigmoid(self.d4(self.pd3(self.up3(h3))))

    def get_latent_var(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return z

    def forward(self, x, style):

        encoded_style = self.style_enc(style)

        mu, logvar, xstn = self.encode(x)
        z = self.reparametrize(mu, logvar)
        res = self.decode(z, encoded_style)
        return res, mu, logvar, xstn

    def init_params(self, net):
        print('Loading the model from the file...')
        net_dict = self.state_dict()
        if isinstance(net, dict):
            pre_dict = net
        else:
            pre_dict = net.state_dict()
        # 1. filter out unnecessary keys
        pre_dict = {k: v for k, v in pre_dict.items() if (k in net_dict)} # for fs net
        net_dict.update(pre_dict)
        # 3. load the new state dict
        self.load_state_dict(net_dict)

class StyleEncoder(nn.Module):
    def __init__(self, in_channel, n_style, cnn_chn=[100, 150, 250]):
        super(StyleEncoder, self).__init__()

        self.cnn_chn = cnn_chn

        # encoder
        self.e1 = nn.Conv2d(in_channel*n_style, self.cnn_chn[0], 7, 2, 3) # inchn, outchn, kernel, stride, padding, dilation, groups
        self.bn1 = nn.BatchNorm2d(self.cnn_chn[0])

        self.e2 = nn.Conv2d(self.cnn_chn[0], self.cnn_chn[1], 4, 2, 1) # 1/4
        self.bn2 = nn.BatchNorm2d(self.cnn_chn[1])

        self.e3 = nn.Conv2d(self.cnn_chn[1], self.cnn_chn[2], 4, 2, 1) # 1/8
        self.bn3 = nn.BatchNorm2d(self.cnn_chn[2])

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):

        h1 = self.leakyrelu(self.bn1(self.e1(x)))
        h2 = self.leakyrelu(self.bn2(self.e2(h1)))
        h3 = self.leakyrelu(self.bn3(self.e3(h2)))

        return h3, h2, h1


class AttrClassifier(nn.Module):
    def __init__(self, in_channel=3, num_classes=43):
        super(AttrClassifier, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, 2, 1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.01))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channel, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 256),
            nn.ZeroPad2d((1, 0, 1, 0)),
        )
        self.head_0 = nn.Sequential(
            nn.Conv2d(256, 256, 4, padding=1, bias=True),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.01),
        )
        self.head_1 = nn.Linear(256*4*4, num_classes, True)

    def forward(self, img):
        out = self.model(img)
        out = self.head_0(out)
        out = out.view(out.size(0), out.size(1) * out.size(2) * out.size(3))
        out = self.head_1(out)
        return out


class DiscriminatorWithClassifier(nn.Module):
    def __init__(self, in_channel=3, num_classes=43, pred_attr=True, dim=64, norm_fn='instancenorm', acti_fn='lrelu',
                fc_dim=1024, fc_norm_fn='none', fc_acti_fn='lrelu', n_layers=5, img_size=64):
        super(DiscriminatorWithClassifier, self).__init__()
        self.pred_attr = pred_attr

        self.f_size = img_size // 2**n_layers

        layers = []
        n_in = in_channel
        for i in range(n_layers):
            n_out = min(dim * 2**i, MAX_DIM)
            layers += [Conv2dBlock(
                n_in, n_out, (4, 4), stride=2, padding=1, norm_fn=norm_fn, acti_fn=acti_fn
            )]
            n_in = n_out
        self.conv = nn.Sequential(*layers)
        self.fc_adv = nn.Sequential(
            LinearBlock(1024 * self.f_size * self.f_size, fc_dim, fc_norm_fn, fc_acti_fn),
            LinearBlock(fc_dim, 1, 'none', 'none'),
            nn.Sigmoid()
        )
        self.fc_cls = nn.Sequential(
            LinearBlock(1024 * self.f_size * self.f_size, fc_dim, fc_norm_fn, fc_acti_fn),
            LinearBlock(fc_dim, num_classes, 'none', 'none'),
            nn.Sigmoid()
        )
        
        #self.auxiliary_cls = AttrClassifier(in_channel=in_channel, num_classes=num_classes)

    def forward(self, img):
        h = self.conv(img)
        h = h.view(h.size(0), -1)
        out_rf = self.fc_adv(h)

        #out_attr = self.auxiliary_cls(img)
        out_attr = self.fc_cls(h)
        
        return out_rf.view(-1), out_attr