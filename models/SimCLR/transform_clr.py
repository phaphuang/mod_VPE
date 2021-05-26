import torchvision.transforms as transforms
from torchvision.transforms.transforms import ToPILImage

class TransformsSimCLR:

    def __init__(self, size):
        s = 1
        color_jitter = transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )

        self.train_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(size=size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
            ]
        )

        self.test_transform = transforms.Compose(
            [
                transforms.Resize(size=size),
                transforms.ToTensor()
            ]
        )
    
    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)