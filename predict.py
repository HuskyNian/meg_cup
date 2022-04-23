from megengine.data import DataLoader, RandomSampler,SequentialSampler
from megengine.data.dataset import Dataset
import numpy as np

import megengine.functional as torch
import megengine as meg
import megengine.functional.nn as F
import megengine.module as nn

import numpy as np
#import torch.nn as nn
#import torch
#import torch.nn.functional as F
#from torch.utils.data import Dataset
from collections import OrderedDict
import matplotlib.pyplot as plt
import random
import os
from tqdm import tqdm
import os.path as osp
import gc
import pickle
from collections import OrderedDict
from model import Restormer_lite
def predict():
    net = Restormer_lite()

    with open('weights.pkl','rb') as f:
        weights = pickle.load(f)
    state_dicts = net.state_dict()
    for i in weights.keys():
        weights[i] = weights[i].reshape( state_dicts[i].shape)
    net.load_state_dict(weights)
    print('model loaded')
    print('prediction')
    content = open('dataset/burst_raw/competition_test_input.0.2.bin', 'rb').read()
    samples_ref = np.frombuffer(content, dtype = 'uint16').reshape((-1,256,256))
    fout = open('workspace/result.bin', 'wb')
    batchsz = 4
    import tqdm

    for i in tqdm.tqdm(range(0, len(samples_ref), batchsz)):
        i_end = min(i + batchsz, len(samples_ref))
        batch_inp = meg.tensor(np.float32(samples_ref[i:i_end, None, :, :]) * np.float32(1 / 65536))
        pred = net(batch_inp)
        pred = (pred.numpy()[:, 0, :, :] * 65536).clip(0, 65535).astype('uint16')
        fout.write(pred.tobytes())

    fout.close()

predict()