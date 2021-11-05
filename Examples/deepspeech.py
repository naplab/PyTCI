import math

import torch
import torch.nn as nn
import torchaudio
from omegaconf import OmegaConf
from torch.cuda.amp import autocast
from torch.nn import CTCLoss
torchaudio.set_audio_backend("sox_io")

from deepspeech_pytorch.configs.train_config import SpectConfig, BiDirectionalConfig, OptimConfig, AdamConfig, SGDConfig, UniDirectionalConfig
from deepspeech_pytorch.configs.inference_config import TranscribeConfig, ModelConfig
from deepspeech_pytorch.decoder import Decoder, GreedyDecoder
from deepspeech_pytorch.utils import load_decoder, load_model
from deepspeech_pytorch.validation import CharErrorRate, WordErrorRate
from deepspeech_pytorch.model import SequenceWise, MaskConv, InferenceBatchSoftmax, Lookahead
from deepspeech_pytorch.enums import RNNType, SpectrogramWindow

import pytorch_lightning as pl


LABELS = list("_'ABCDEFGHIJKLMNOPQRSTUVWXYZ ")
MODEL_CFG = BiDirectionalConfig(rnn_type=RNNType.lstm, hidden_size=1024, hidden_layers=7)
OPTIM_CFG = AdamConfig(learning_rate=0.00015, learning_anneal=0.99, weight_decay=1e-05, eps=1e-08, betas=[0.9, 0.999])
SPECT_CFG = SpectConfig(sample_rate=16000, window_size=0.02, window_stride=0.01, window=SpectrogramWindow.hamming)
PRECISION = 16


class SpectrogramParser(nn.Module):
    def __init__(self, audio_conf: SpectConfig, normalize: bool = False):
        """
       	Parses audio file into spectrogram with optional normalization
       	:param audio_conf: Dictionary containing the sample rate, window and the window length/stride in seconds
       	:param normalize(default False):  Apply standard mean and deviation normalization to audio tensor
       	"""
        super().__init__()
        self.window_stride = audio_conf.window_stride
        self.window_size = audio_conf.window_size
        self.sample_rate = audio_conf.sample_rate
        self.window = audio_conf.window.value
        self.normalize = normalize
        
        n_fft = int(self.sample_rate * self.window_size)
        win_length = n_fft
        hop_length = int(self.sample_rate * self.window_stride)
        if self.window == 'hamming':
            window = torch.hamming_window
        else:
            raise NotImplementedError()
        
        self.transform = torchaudio.transforms.Spectrogram(
            n_fft, win_length, hop_length, window_fn=window, power=1, normalized=False)

    @torch.no_grad()
    def forward(self, audio):
        if audio.shape[0] == 1:
            audio = audio.squeeze() # mono
        else:
            audio = audio.mean(axis=0) # multiple channels, average
        
        spect = self.transform(audio)
        spect = torch.log1p(spect)
        
        if self.normalize:
            mean = spect.mean()
            std = spect.std()
            spect.add_(-mean)
            spect.div_(std)
        
        return spect.T.contiguous()


class BatchRNN(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM, bidirectional=False, batch_norm=True):
        super(BatchRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_norm = SequenceWise(nn.BatchNorm1d(input_size)) if batch_norm else None
        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size,
                            bidirectional=bidirectional, bias=True)
        self.num_directions = 2 if bidirectional else 1

    def flatten_parameters(self):
        self.rnn.flatten_parameters()

    def forward(self, x, output_lengths):
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x = nn.utils.rnn.pack_padded_sequence(x, output_lengths, enforce_sorted=False)
        x, h = self.rnn(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x)
        if self.bidirectional:
            x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)  # (TxNxH*2) -> (TxNxH) by sum
        return x


class DeepSpeech(pl.LightningModule):
    def __init__(self, labels=LABELS, model_cfg=MODEL_CFG, precision=PRECISION, optim_cfg=OPTIM_CFG, spect_cfg=SPECT_CFG):
        super().__init__()
        self.save_hyperparameters()
        self.model_cfg = model_cfg
        self.precision = precision
        self.optim_cfg = optim_cfg
        self.spect_cfg = spect_cfg
        self.bidirectional = True if OmegaConf.get_type(model_cfg) is BiDirectionalConfig else False

        self.labels = labels
        num_classes = len(self.labels)

        self.conv = MaskConv(nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True)
        ))
        # Based on above convolutions and spectrogram size using conv formula (W - F + 2P)/ S+1
        rnn_input_size = int(math.floor((self.spect_cfg.sample_rate * self.spect_cfg.window_size) / 2) + 1)
        rnn_input_size = int(math.floor(rnn_input_size + 2 * 20 - 41) / 2 + 1)
        rnn_input_size = int(math.floor(rnn_input_size + 2 * 10 - 21) / 2 + 1)
        rnn_input_size *= 32

        self.rnns = nn.Sequential(
            BatchRNN(
                input_size=rnn_input_size,
                hidden_size=self.model_cfg.hidden_size,
                rnn_type=self.model_cfg.rnn_type.value,
                bidirectional=self.bidirectional,
                batch_norm=False
            ),
            *(
                BatchRNN(
                    input_size=self.model_cfg.hidden_size,
                    hidden_size=self.model_cfg.hidden_size,
                    rnn_type=self.model_cfg.rnn_type.value,
                    bidirectional=self.bidirectional
                ) for x in range(self.model_cfg.hidden_layers - 3)
            )
        )

        self.lookahead = nn.Sequential(
            # consider adding batch norm?
            Lookahead(self.model_cfg.hidden_size, context=self.model_cfg.lookahead_context),
            nn.Hardtanh(0, 20, inplace=True)
        ) if not self.bidirectional else None

        fully_connected = nn.Sequential(
            nn.BatchNorm1d(self.model_cfg.hidden_size),
            nn.Linear(self.model_cfg.hidden_size, num_classes, bias=False)
        )
        self.fc = nn.Sequential(
            SequenceWise(fully_connected),
        )
        self.inference_softmax = InferenceBatchSoftmax()
        self.criterion = CTCLoss(blank=self.labels.index('_'), reduction='sum', zero_infinity=True)
        self.evaluation_decoder = GreedyDecoder(self.labels)  # Decoder used for validation
        self.wer = WordErrorRate(
            decoder=self.evaluation_decoder,
            target_decoder=self.evaluation_decoder
        )
        self.cer = CharErrorRate(
            decoder=self.evaluation_decoder,
            target_decoder=self.evaluation_decoder
        )

    def forward(self, x, lengths):
        lengths = lengths.cpu().int()
        output_lengths = self.get_seq_lens(lengths)
        x, _ = self.conv(x.transpose(1,2).unsqueeze(1).contiguous(), output_lengths)

        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # Collapse feature dimension
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # TxNxH

        for rnn in self.rnns:
            x = rnn(x, output_lengths)

        if not self.bidirectional:  # no need for lookahead layer in bidirectional
            x = self.lookahead(x)

        x = self.fc(x)
        x = x.transpose(0, 1)
        # identity in training mode, softmax in eval mode
        x = self.inference_softmax(x)
        return x, output_lengths
    
    def unpack_batch(self, batch):
        inputs = batch.get('inputs', None)
        input_lengths = batch.get('input_lengths', None)
        labels = batch.get('labels', None)
        label_lengths = batch.get('label_lengths', None)
        
        return inputs, labels, input_lengths, label_lengths

    def training_step(self, batch, batch_idx):
        inputs, targets, input_sizes, target_sizes = self.unpack_batch(batch)
        if inputs is None: # skip step
            return None
        
        out, output_sizes = self(inputs, input_sizes)
        out = out.transpose(0, 1)  # TxNxH
        out = out.log_softmax(-1)

        loss = self.criterion(out, targets, output_sizes, target_sizes)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets, input_sizes, target_sizes = self.unpack_batch(batch)
        if inputs is None: # skip step
            return
        
        inputs = inputs.to(self.device)
        with autocast(enabled=self.precision == 16):
            out, output_sizes = self(inputs, input_sizes)
        decoded_output, _ = self.evaluation_decoder.decode(out, output_sizes)
        
        self.wer(preds=out, preds_sizes=output_sizes, targets=targets, target_sizes=target_sizes)
        self.cer(preds=out, preds_sizes=output_sizes, targets=targets, target_sizes=target_sizes)
        self.log('wer', self.wer.compute(), prog_bar=True, on_epoch=True)
        self.log('cer', self.cer.compute(), prog_bar=True, on_epoch=True)
    
    def test_step(self, *args):
        return self.validation_step(*args)

    def configure_optimizers(self):
        if OmegaConf.get_type(self.optim_cfg) is SGDConfig:
            optimizer = torch.optim.SGD(
                params=self.parameters(),
                lr=self.optim_cfg.learning_rate,
                momentum=self.optim_cfg.momentum,
                nesterov=True,
                weight_decay=self.optim_cfg.weight_decay
            )
        elif OmegaConf.get_type(self.optim_cfg) is AdamConfig:
            optimizer = torch.optim.AdamW(
                params=self.parameters(),
                lr=self.optim_cfg.learning_rate,
                betas=self.optim_cfg.betas,
                eps=self.optim_cfg.eps,
                weight_decay=self.optim_cfg.weight_decay
            )
        else:
            raise ValueError("Optimizer has not been specified correctly.")

        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer,
            gamma=self.optim_cfg.learning_anneal
        )
        return [optimizer], [scheduler]

    def get_seq_lens(self, input_length):
        """
        Given a 1D Tensor or Variable containing integer sequence lengths, return a 1D tensor or variable
        containing the size sequences that will be output by the network.
        :param input_length: 1D Tensor
        :return: 1D Tensor scaled by model
        """
        seq_len = input_length
        for m in self.conv.modules():
            if type(m) == nn.modules.conv.Conv2d:
                seq_len = ((seq_len + 2 * m.padding[1] - m.dilation[1] * (m.kernel_size[1] - 1) - 1) // m.stride[1] + 1)
        return seq_len.int()
    
    def activation_fx(self, layer, log=True):
        def activation(x):
            lengths = [x.shape[1]] * x.shape[0]
            output_lengths = self.get_seq_lens(lengths)
            
            #x, _ = model.conv(x, output_lengths)
            for module in self.conv.seq_module:
                x = module(x)
                mask = torch.BoolTensor(x.size()).fill_(0)
                if x.is_cuda:
                    mask = mask.cuda()
                for i, length in enumerate(output_lengths):
                    length = length.item()
                    if (mask[i].size(2) - length) > 0:
                        mask[i].narrow(2, length, mask[i].size(2) - length).fill_(1)
                x = x.masked_fill(mask, 0)
                
                if isinstance(module, torch.nn.Hardtanh):
                    layer -= 1
                    if layer < 0:
                        break
            
            sizes = x.size()
            x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # Collapse feature dimension
            x = x.transpose(1, 2).transpose(0, 1).contiguous()  # TxNxH
            if layer < 0:
                return x.detach().cpu().numpy()
            
            for rnn in self.rnns:
                x = rnn(x, output_lengths)
                layer -= 1
                if layer < 0:
                    return x.detach().cpu().numpy()
            
            if not self.bidirectional:  # no need for lookahead layer in bidirectional
                x = self.lookahead(x)
            
            x = self.fc(x)
            
            # identity in training mode, softmax in eval mode
            if log:
                x = torch.nn.functional.log_softmax(x, dim=-1)
            else:
                x = torch.nn.functional.softmax(x, dim=-1)
            layer -= 1
            if layer < 0:
                return x.detach().cpu().numpy()
            
            return None
        
        return activation


spect_parser = SpectrogramParser(audio_conf=SPECT_CFG, normalize=True)
