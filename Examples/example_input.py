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


def wav2spec():
    torch.nn.Sequential(
        torchaudio.transforms.MelSpectrogram(in_sr, n_fft=1024, hop_length=int(in_sr/out_sr), f_min=20, f_max=8_000, n_mels=freqbins, power=2.0),
        torchaudio.transforms.AmplitudeToDB('power', top_db=top_db),
        type("Normalize", (torch.nn.Module,), dict(forward=lambda self, x: (x - x.max()).squeeze(0).T.float() / top_db + 1))()
    )
    
    return None


def stretch_audio(scale_by=1/1.2):
    # Stretch samples
    tfm = [['tempo', '-s', str(scale_by)], ['rate', '16000']]
    x = np.stack([
        torchaudio.sox_effects.apply_effects_tensor(
            torch.from_numpy(_).T, in_sr, tfm
        )[0].squeeze(0).numpy() for _ in x])


model = PyTCI.deepspeech.DeepSpeech().to(device).eval()
model.load_state_dict(torch.load('resources/deepspeech2-pretrained.ckpt')['state_dict'])
nodes = [2592, 1312, 1024, 1024, 1024, 1024, 1024, 29]

analyzer = PyTCI.Analyzer(out_sr=100, model=model)
analyzer.load_segments()
analyzer.run()
