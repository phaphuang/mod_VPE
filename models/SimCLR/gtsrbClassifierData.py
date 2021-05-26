import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.misc as m
import random

from .transform_clr import TransformsSimCLR

class gtsrbClassifierData(Dataset):
    def __init__(self, root, exp, split='train', img_size=None, prototype_sampling_rate=0.001):
        super().__init__()

        if split == 'train':
            self.proto_rate = prototype_sampling_rate
        else:
            self.proto_rate = 0.0
        
        self.inputs = []
        self.targets = []
        self.class_names = []

        if split == 'train':
            self.split = 'GTSRB'
            self.n_classes = 43 # test on TT100K (36 classes)
            self.tr_class = torch.LongTensor([16, 18, 34, 39]) - 1
            self.te_class = torch.LongTensor([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,  17,  19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,   35,36,37,38,   40,41,42,43]) - 1

        self.img_size = img_size
        self.mean = np.array([125.00, 125.00, 125.00]) # average intensity

        self.root = root
        self.dataPath = root + exp + '/' + self.split + '_impaths_all.txt'
        #self.labelPath = root + exp + '/' + self.split + '_imclasses_all.txt'

        f_data = open(self.dataPath,'r')
        data_lines = f_data.readlines()
        #label_lines = f_label.readlines()

        for i in range(len(data_lines)):
            self.inputs.append(root+data_lines[i][0:-1])
            #self.targets.append(int(label_lines[i].split()[0])) # label: [road class, wet/dry, video index]
        
        classnamesPath = root + exp + '/' + self.split + '_classnames.txt'
        f_classnames = open(classnamesPath, 'r')
        data_lines = f_classnames.readlines()
        for i in range(len(data_lines)):
            self.class_names.append(data_lines[i][0:-1])

        assert(self.n_classes == len(self.class_names))

        print('%s %d classes'%(self.split, len(self.class_names)))
        print('Load %s: %d samples'%(self.split,  len(self.inputs)))
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        img_path = self.inputs[index]
        #print(img_path)
        # Load images
        img = m.imread(img_path)
        img = np.array(img, dtype=np.uint8)

        #img = torch.from_numpy(img)

        transform = TransformsSimCLR(size=self.img_size)
        x_i, x_j = transform(img)

        return x_i, x_j
        

