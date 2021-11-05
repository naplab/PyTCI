import os
import numpy as np

# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.multiprocessing
import torchaudio
# torch.multiprocessing.set_sharing_strategy('file_system')
from torch.cuda.amp import autocast
import pytorch_lightning as pl

import PyTCI

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model = PyTCI.deepspeech.DeepSpeech().to(device).eval()
model.load_state_dict(torch.load('resources/deepspeech2-pretrained.ckpt')['state_dict'])
nodes = [2592, 1312, 1024, 1024, 1024, 1024, 1024, 29]

analyzer = PyTCI.Analyzer(out_sr=100)
analyzer.load_segments()
analyzer.write_sequences()

model()

analyzer.read_responses()
analyzer.run()
