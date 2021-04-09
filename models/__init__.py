from models.vaeIdsiaStn import *
from models.vqvae import VQVAEModel
from models.vqcvae import VQ_CVAE
from models.vqvae2 import VQVAE2

def get_model(name, n_classes=None):
    model = _get_model_instance(name)

    if name is 'vaeIdsiaStn':
        model = model(nc=3, input_size = 64, latent_variable_size=300, cnn_chn=[100, 150, 250] ,param1=[200,300,200], param2=None, param3 = [150, 150, 150]) # idsianet cnn_chn=[100,150,250] latent = 300
        print('Use vae+Idsianet (stn1 + stn3) with random initialization!')

    if name is 'vaeIdsia':
        model = model(nc=3, input_size = 64, latent_variable_size=300, cnn_chn=[100, 150, 250] ,param1=None, param2=None, param3 = None) # idsianet cnn_chn=[100,150,250] latent = 300
        print('Use vae+Idsianet (without stns) with random initialization!')
    
    if name is 'vqvae':
        model = model(num_hiddens=128, num_residual_layers=2, num_residual_hiddens=32, num_embeddings=512, embedding_dim=64, commitment_cost=0.25, decay=0.99)
    
    if name is 'vqcvae':
        model = model(d=128, k=512, num_channels=3)
    
    if name is 'vqvae2':
        model = model(hidden_channels=64, embed_dim=32, nb_entries=256, nb_levels=2, scaling_rates=[4,2], param1=[200,300,200], input_size=64)

    return model

def _get_model_instance(name):
    try:
        return {
            'vaeIdsiaStn' : VAEIdsia,
            'vaeIdsia' : VAEIdsia,
            'vqvae' : VQVAEModel,
            'vqcvae' : VQ_CVAE,
            'vqvae2' : VQVAE2,
        }[name]
    except:
        print('Model {} not available'.format(name))
