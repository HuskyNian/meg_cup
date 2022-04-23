from megengine.data import DataLoader, RandomSampler,SequentialSampler
from megengine.data.dataset import Dataset
import numpy as np

class DataFolder(Dataset):
    def __init__(self, x,y):
        self.x = x.reshape((-1,1,256,256))* np.float32(1 / 65536)
        self.y = y.reshape((-1,1,256,256))* np.float32(1 / 65536)
        n, c, h, w = self.x.shape

        self.length = len(x)
 
    def __getitem__(self, index):
        return self.x[index],self.y[index]
 
    def __len__(self):
        return self.length