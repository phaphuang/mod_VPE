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


class VAETranDis(nn.Module):
    def __init__(self, nc, input_size, latent_variable_size=300, cnn_chn=[100, 150, 250], 
                param1=None, param2=None, param3=None,
                n_style=4, style_out_channel=256,attention=True, n_res_blocks=8):
        super(VAETranDis, self).__init__()

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
        self.d1 = nn.Linear(latent_variable_size+style_out_channel, int(input_size/8*input_size/8*self.cnn_chn[2]))

        self.up1 = nn.UpsamplingNearest2d(scale_factor=2) # 8 -> 16
        self.pd1 = nn.ReplicationPad2d(1)
        self.d2 = nn.Conv2d(self.cnn_chn[2], self.cnn_chn[1], 3, 1)
        self.bn6 = nn.BatchNorm2d(self.cnn_chn[1], 1.e-3)

        self.up2 = nn.UpsamplingNearest2d(scale_factor=2) # 16 -> 32
        self.pd2 = nn.ReplicationPad2d(1)
        self.d3 = nn.Conv2d(self.cnn_chn[1], self.cnn_chn[0], 3, 1)
        self.bn7 = nn.BatchNorm2d(self.cnn_chn[0], 1.e-3)

        self.up3 = nn.UpsamplingNearest2d(scale_factor=2) # 32 -> 64
        self.pd3 = nn.ReplicationPad2d(1)
        self.d4 = nn.Conv2d(self.cnn_chn[0], 3, 3, 1)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        if param1 is not None:
            self.stn1 = stn(3, self.input_size, param1)
        if param2 is not None:
            self.stn2 = stn(self.cnn_chn[0], self.input_size/2, param2)
        if param3 is not None:
            self.stn3 = stn(self.cnn_chn[1], self.input_size/4, param3)

        self.style_enc = StyleEncoder(in_channel=nc, n_style=n_style,
                                      style_out_channel=style_out_channel, attention=attention, 
                                      n_res_blocks=n_res_blocks)


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

    def decode(self, z, style):
        #print(z.shape, style.shape)

        z_ = torch.cat([z, style.squeeze()], dim=1)
        h1 = self.relu(self.d1(z_))
        # h1 = h1.view(-1, self.ngf*8*2, 4, 4)
        h1 = h1.view(-1, self.cnn_chn[2], int(self.input_size/8), int(self.input_size/8))
        h2 = self.leakyrelu(self.bn6(self.d2(self.pd1(self.up1(h1)))))
        h3 = self.leakyrelu(self.bn7(self.d3(self.pd2(self.up2(h2)))))
        return self.sigmoid(self.d4(self.pd3(self.up3(h3))))

    def get_latent_var(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return z

    def forward(self, x, style):

        target_style = self.style_enc(style)

        mu, logvar, xstn = self.encode(x)
        z = self.reparametrize(mu, logvar)
        res = self.decode(z, target_style)
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

#### Adding Style Encoder ####

class ResidualBlock(nn.Module):
    def __init__(self, in_channel):
        super(ResidualBlock, self).__init__()

        conv_block = [
            nn.Conv2d(in_channel, in_channel, 3, stride=1, padding=1, bias=False),
            #nn.InstanceNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, in_channel, 3, stride=1, padding=1, bias=False),
            #nn.InstanceNorm2d(in_channel),
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

# Self Attention module from self-attention gan
class SelfAttention(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation=None):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        # print('attention size', x.size())
        m_batchsize, C, width, height = x.size()
        # print('query_conv size', self.query_conv(x).size())
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X C X (N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C X (W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # B X (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma*out + x
        return out

class Down(nn.Module):
    def __init__(self, in_channel, out_channel, normalize=True, attention=False,
                 lrelu=False, dropout=0.0, bias=False, kernel_size=4, stride=2, padding=1):
        super(Down, self).__init__()
        layers = [nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size,
                  stride=stride, padding=padding, bias=bias)]
        if attention:
            layers.append(SelfAttention(out_channel))
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channel))
        if lrelu:
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class StyleEncoder(nn.Module):
    def __init__(self, in_channel=3, n_style=4, style_out_channel=256,
                 n_res_blocks=8, attention=True):
        super(StyleEncoder, self).__init__()
        layers = []
        # Initial Conv
        layers += [nn.Conv2d(in_channel*n_style, 64, 7, stride=1, padding=3, bias=False),
                   nn.InstanceNorm2d(64),
                   nn.ReLU(inplace=True)]

        # Down scale
        layers += [Down(64, 128)]
        layers += [Down(128, 256)]
        layers += [Down(256, 512)]
        layers += [Down(512, 512, dropout=0.5)]
        layers += [Down(512, 512, dropout=0.5)]
        layers += [Down(512, style_out_channel, normalize=False, dropout=0.5)]
        self.down = nn.Sequential(*layers)

        # Style transform
        res_blks = []
        res_channel = style_out_channel
        for _ in range(n_res_blocks):
            res_blks.append(ResidualBlock(res_channel))
        self.res_layer = nn.Sequential(*res_blks)

    def forward(self, style):
        source_style = self.down(style)
        source_style = source_style.view(source_style.size(0), source_style.size(1), 1, 1)
        target_style = self.res_layer(source_style)
        return target_style

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
    def __init__(self, in_channel=3, num_classes=43, pred_attr=True):
        super(DiscriminatorWithClassifier, self).__init__()
        self.pred_attr = pred_attr

        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, 2, 1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.1))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channel, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 256),
            nn.ZeroPad2d((1, 0, 1, 0)),
        )

        self.model_one_input = nn.Sequential(
            *discriminator_block(in_channel, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 256),
            nn.ZeroPad2d((1, 0, 1, 0)),
        )

        self.f = nn.Conv2d(256, 1, 4, padding=1, bias=False)
        
        self.auxiliary_cls = AttrClassifier(in_channel=in_channel, num_classes=num_classes)

    def forward(self, img):
        out = self.model(img)
        out_rf = self.f(out)  # real or fake for image
        out_attr = self.auxiliary_cls(img)
        
        return out_rf, out_attr