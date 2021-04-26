from models.vqvae2_base import VQVAE2Base
from models.vqvae2_disc import VQVAE2Disc

def get_model(name, n_classes=None):
    model = _get_model_instance(name)
    
    if name is 'vqvae2Base':
        model = model()

    if name is 'vqvae2Disc':
        model = model()
    
    return model

def _get_model_instance(name):
    try:
        return {
            'vqvae2Base': VQVAE2Base,
            'vqvae2Disc': VQVAE2Disc,
        }[name]
    except:
        print('Model {} not available'.format(name))
