import torch
import torch.nn as nn
import torch.nn.functional as F

from math import log2
from typing import Tuple

from models.helper import HelperModule

class ReZero(HelperModule):
    def build(self, in_channels: int, res_channels: int):
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, res_channels, 3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(res_channels),
            nn.ReLU(),

            nn.Conv2d(res_channels, in_channels, 3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(in_channels),
            nn.ReLU(),
        )
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.layers(x) * self.alpha + x

class ResidualStack(HelperModule):
    def build(self, in_channels: int, res_channels: int, nb_layers: int):
        self.stack = nn.Sequential(*[ReZero(in_channels, res_channels) 
                        for _ in range(nb_layers)
                    ])

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.stack(x)

class Encoder(HelperModule):
    def build(self, 
            in_channels: int, hidden_channels: int, 
            res_channels: int, nb_res_layers: int,
            downscale_factor: int,
        ):
        assert log2(downscale_factor) % 1 == 0, "Downscale must be a power of 2"
        downscale_steps = int(log2(downscale_factor))
        layers = []
        c_channel, n_channel = in_channels, hidden_channels // 2
        for _ in range(downscale_steps):
            layers.append(nn.Sequential(
                nn.Conv2d(c_channel, n_channel, 4, stride=2, padding=1),
                nn.ReLU(),
            ))
            c_channel, n_channel = n_channel, hidden_channels
        layers.append(nn.Conv2d(c_channel, n_channel, 3, stride=1, padding=1))
        layers.append(ResidualStack(n_channel, res_channels, nb_res_layers))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.layers(x)

class Decoder(HelperModule):
    def build(self, 
            in_channels: int, hidden_channels: int, out_channels: int,
            res_channels: int, nb_res_layers: int,
            upscale_factor: int,
        ):
        assert log2(upscale_factor) % 1 == 0, "Downscale must be a power of 2"
        upscale_steps = int(log2(upscale_factor))
        layers = [nn.Conv2d(in_channels, hidden_channels, 3, stride=1, padding=1)]
        layers.append(ResidualStack(hidden_channels, res_channels, nb_res_layers))
        c_channel, n_channel = hidden_channels, hidden_channels // 2
        for _ in range(upscale_steps):
            layers.append(nn.Sequential(
                nn.ConvTranspose2d(c_channel, n_channel, 4, stride=2, padding=1),
                nn.ReLU(),
            ))
            c_channel, n_channel = n_channel, out_channels
        layers.append(nn.Conv2d(c_channel, n_channel, 3, stride=1, padding=1))
        layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.layers(x)

"""
    Almost directly taken from https://github.com/rosinality/vq-vae-2-pytorch/blob/master/vqvae.py
    No reason to reinvent this rather complex mechanism.
    Essentially handles the "discrete" part of the network, and training through EMA rather than 
    third term in loss function.
"""
class CodeLayer(HelperModule):
    def build(self, in_channels: int, embed_dim: int, nb_entries: int):
        self.conv_in = nn.Conv2d(in_channels, embed_dim, 1)

        self.dim = embed_dim
        self.n_embed = nb_entries
        self.decay = 0.99
        self.eps = 1e-5

        embed = torch.randn(embed_dim, nb_entries)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(nb_entries))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, x: torch.FloatTensor) -> Tuple[torch.FloatTensor, float, torch.LongTensor]:
        x = self.conv_in(x).permute(0,2,3,1)
        flatten = x.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*x.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            # TODO: Replace this? Or can we simply comment out?
            # dist_fn.all_reduce(embed_onehot_sum)
            # dist_fn.all_reduce(embed_sum)

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - x).pow(2).mean()
        quantize = x + (quantize - x).detach()

        return quantize.permute(0, 3, 1, 2), diff, embed_ind

    def embed_code(self, embed_id: torch.LongTensor) -> torch.FloatTensor:
        return F.embedding(embed_id, self.embed.transpose(0, 1))

class Upscaler(HelperModule):
    def build(self,
            embed_dim,
            scaling_rates,
        ):

        self.stages = nn.ModuleList()
        for sr in scaling_rates:
            upscale_steps = int(log2(sr))
            layers = []
            for _ in range(upscale_steps):
                layers.append(nn.ConvTranspose2d(embed_dim, embed_dim, 4, stride=2, padding=1))
                layers.append(nn.ReLU())
            self.stages.append(nn.Sequential(*layers))

    def forward(self, x: torch.FloatTensor, stage: int) -> torch.FloatTensor:
        return self.stages[stage](x)

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

"""
    Main VQ-VAE-2 Module, capable of support arbitrary number of levels
"""
class VQVAE2(HelperModule):
    def build(self,
            in_channels               = 3,
            hidden_channels            = 128,
            res_channels              = 32,
            nb_res_layers             = 2,
            nb_levels                  = 3,
            embed_dim                 = 64,
            nb_entries                 = 512,
            scaling_rates        = [8, 4, 2],
            param1                  = None,
            input_size              = 64,
        ):
        self.nb_levels = nb_levels
        assert len(scaling_rates) == nb_levels, "Number of scaling rates not equal to number of levels!"

        self.encoders = nn.ModuleList([Encoder(in_channels, hidden_channels, res_channels, nb_res_layers, scaling_rates[0])])
        for i, sr in enumerate(scaling_rates[1:]):
            self.encoders.append(Encoder(hidden_channels, hidden_channels, res_channels, nb_res_layers, sr))

        # self.codebooks = nn.ModuleList([CodeLayer(hidden_channels+embed_dim*(nb_levels-1), embed_dim, nb_entries)])
        # for i in range(nb_levels - 1):
            # self.codebooks.append(CodeLayer(hidden_channels+embed_dim*(nb_levels-2-i), embed_dim, nb_entries))
        self.codebooks = nn.ModuleList()
        for i in range(nb_levels - 1):
            self.codebooks.append(CodeLayer(hidden_channels+embed_dim, embed_dim, nb_entries))
        self.codebooks.append(CodeLayer(hidden_channels, embed_dim, nb_entries))

        self.decoders = nn.ModuleList([Decoder(embed_dim*nb_levels, hidden_channels, in_channels, res_channels, nb_res_layers, scaling_rates[0])])
        for i, sr in enumerate(scaling_rates[1:]):
            self.decoders.append(Decoder(embed_dim*(nb_levels-1-i), hidden_channels, embed_dim, res_channels, nb_res_layers, sr))

        self.upscalers = nn.ModuleList()
        for i in range(nb_levels - 1):
            self.upscalers.append(Upscaler(embed_dim, scaling_rates[1:len(scaling_rates) - i][::-1]))

        self.param1 = param1
        if self.param1 is not None:
            self.stn1 = stn(3, input_size, self.param1)

    def forward(self, x, encode_only=False):
        if encode_only:
            encoder_outputs = []
            for enc in self.encoders:
                if len(encoder_outputs):
                    encoder_outputs.append(enc(encoder_outputs[-1]))
                else:
                    encoder_outputs.append(enc(x))
            
            return encoder_outputs
        else:
            # TODO: Might be easier to replace these with dictionaries?
            encoder_outputs = []
            code_outputs = []
            decoder_outputs = []
            upscale_counts = []
            diffs = []

            if self.param1 is not None:
                x = self.stn1(x)

            for enc in self.encoders:
                if len(encoder_outputs):
                    encoder_outputs.append(enc(encoder_outputs[-1]))
                else:
                    encoder_outputs.append(enc(x))

            for l in range(self.nb_levels-1, -1, -1):
                codebook, decoder = self.codebooks[l], self.decoders[l]

                if len(decoder_outputs): # if we have previous levels to condition on
                    code_q, code_d, code_idx = codebook(torch.cat([encoder_outputs[l], decoder_outputs[-1]], axis=1))
                else:
                    code_q, code_d, code_idx = codebook(encoder_outputs[l])
                diffs.append(code_d)

                code_outputs = [self.upscalers[i](c, upscale_counts[i]) for i, c in enumerate(code_outputs)]
                upscale_counts = [u+1 for u in upscale_counts]
                decoder_outputs.append(decoder(torch.cat([code_q, *code_outputs], axis=1)))

                code_outputs.append(code_q)
                upscale_counts.append(0)

            return decoder_outputs[-1], diffs, encoder_outputs, decoder_outputs, x

if __name__ == '__main__':
    from models.helper import get_parameter_count
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    nb_levels = 6 
    net = VQVAE2(nb_levels=nb_levels, scaling_rates=[2]*nb_levels).to(device)
    print(f"Number of trainable parameters: {get_parameter_count(net)}")

    x = torch.randn(1, 3, 64, 64).to(device)
    _, diffs, enc_out, dec_out = net(x)
    print('\n'.join(str(y.shape) for y in enc_out))
    print()
    print('\n'.join(str(y.shape) for y in dec_out))
    print()
    print('\n'.join(str(y) for y in diffs))