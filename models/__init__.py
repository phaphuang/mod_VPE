from models.vaeTranDis import VAETranDis
from models.vaeIdsiaStn import *
from models.vaeCatTranDis import VAECatTranDis

def get_model(name, n_classes=None):
    model = _get_model_instance(name)

    if name is 'vaeIdsiaStn':
        model = model(nc=3, input_size = 64, latent_variable_size=300, cnn_chn=[100, 150, 250] ,param1=[200,300,200], param2=None, param3 = [150, 150, 150]) # idsianet cnn_chn=[100,150,250] latent = 300
        print('Use vae+Idsianet (stn1 + stn3) with random initialization!')

    if name is 'vaeIdsia':
        model = model(nc=3, input_size = 64, latent_variable_size=300, cnn_chn=[100, 150, 250] ,param1=None, param2=None, param3 = None) # idsianet cnn_chn=[100,150,250] latent = 300
        print('Use vae+Idsianet (without stns) with random initialization!')
    
    if name is 'vaeTranDis':
        #model = model(nc=3, input_size = 64, latent_variable_size=300, cnn_chn=[100, 150, 250] ,param1=[200,300,200], param2=None, param3 = [150, 150, 150], n_style=4, style_out_channel=256,attention=True, n_res_blocks=8) # idsianet cnn_chn=[100,150,250] latent = 300
        model = model(nc=3, input_size = 64, latent_variable_size=300, cnn_chn=[100, 150, 250] ,param1=None, param2=None, param3 =None, n_style=4, style_out_channel=20,attention=True, n_res_blocks=8) # idsianet cnn_chn=[100,150,250] latent = 300
        print('Use vae+transformer+discriminator (with stns) with random initialization!')
    
    if name is 'vaeCatTranDis':
        #model = model(nc=3, input_size = 64, latent_variable_size=300, cnn_chn=[100, 150, 250] ,param1=[200,300,200], param2=None, param3 = [150, 150, 150], n_style=4, style_out_channel=256,attention=True, n_res_blocks=8) # idsianet cnn_chn=[100,150,250] latent = 300
        model = model(nc=3, input_size = 64, latent_variable_size=300, cnn_chn=[100, 150, 250] ,param1=None, param2=None, param3 =None, n_style=4) # idsianet cnn_chn=[100,150,250] latent = 300
        print('Use vae+transformer+discriminator (with stns) with random initialization!')

    return model

def _get_model_instance(name):
    try:
        return {
            'vaeIdsiaStn' : VAEIdsia,
            'vaeIdsia' : VAEIdsia,
            'vaeTranDis': VAETranDis,
            'vaeCatTranDis': VAECatTranDis,
        }[name]
    except:
        print('Model {} not available'.format(name))
