import os
import glob
import math
import pickle
import numpy as np
import scipy as sp
from typing import List

import torch
import torchaudio


class Analyzer:
    def __init__(
        self,
        out_sr,
        model=None,
        segment_durs=None,
        segment_maxdur=4,
        segment_overlap=0.020,
        segment_alignment='center',
        threshold=0.75,
        size_block=48,
        size_context=8,
        size_margin=1,
        size_batch_corr=None,
        channel_mask=slice(None),
        comparison_method='random-random',
        device=torch.device('cpu'),
        output_path=None,
    ):
        """
        Base class for performing Temporal Context Invariance (TCI) analysis.

        out_sr: sampling rate of (model) responses.
        model: a callable object, like a function or pytorch Module with method `forward` that take input `x` and return
                response `y`. If set, used for inferring responses used in the analysis; if None, you have to read the
                responses from file, using `read_responses`.
        segment_durs: list of segment durations to consider in seconds, use default (20--2480 ms) if None.
        segment_maxdur: maximum duration of segment, used in case of 'natural' segment.
        segment_overlap: amount of overlap between adjacent elements of `x`, in milliseconds
        segment_alignment: where to extract subsegments from; must be one of 'start', 'center', 'end', or 'max'.
        threshold: the threshold value used on cross-context correlation, to measure integration window.
        size_block: duration of blocks in a batch, in seconds.
        size_context: duration of symmetric context at start and end of a block, in seconds.
        size_margin: margin around the shared segment when calculating cross-context correlation, in seconds.
        size_batch_corr: the number of nodes per model response output to calculate CCC on simultaneously.
        channel_mask: a mask for output channels of `model` to limit analysis to a subset of units. Recommended usage
                is by a python `slice` to batch processing of layers, to avoid memory issues when layer is large.
        comparison_method: segment-aligned-respnoses comparison method, one of 'random-random' or 'natural-random'.
        device: pytorch device used for doing model inference and correlation calculation.
        output_path: if not None, location to save analysis results, i.e., cross-context correlations and integration windows.
        """
        if comparison_method not in ['random-random', 'natural-random']:
            raise ValueError('Parameter `comparison_method` has to either be "random-random" or "natural-random".')
        
        self.model = model
        self.in_sr = None
        self.out_sr = out_sr
        self.segments = None
        self.threshold = threshold

        self.segment_durs = np.array([
            0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.20, 0.24, 0.28, 0.34, 0.40, 0.48,
            0.58, 0.70, 0.84, 1.00, 1.20, 1.44, 1.72, 2.06, 2.48] if segment_durs is None else segment_durs)

        self.segment_maxdur = segment_maxdur
        self.segment_overlap = segment_overlap
        self.segment_alignment = segment_alignment

        self.size_block = size_block
        self.size_context = size_context
        self.size_margin = size_margin
        self.size_batch_corr = size_batch_corr
        self.mask_channels = channel_mask

        self.device = device
        self.output_path = output_path

        self.seq_A = None
        self.seq_A = None
        self.seq_B = None
        self.resp_A = None
        self.resp_B = None

    def load_segments(self, path, fmt='audio', preprocess=None):
        """
        Load input segments from disk and preprocess them individually (optional).

        path: if a string, `path` is a directory containing audio files; if a list, `path` is a list of audio files.
        fmt: format of segment files. Currently accepted formats are 'audio' and 'numpy'.
        preprocess: a preprocessing function or list of functions to be applied to all segments individually in-memory.
        """
        if not isinstance(path, (str, List)):
            raise ValueError('Parameter `path` should either be a string pointing to a directory, or a list of audio files.')

        if isinstance(path, str):
            path = glob.glob(os.path.join(dir, '*'))

        if fmt == 'audio':
            segments, sr = zip([torchaudio.load(file, channels_first=False) for file in path])
        elif fmt == 'numpy':
            segments, sr = zip([np.load(file) for file in path])
        else:
            raise NotImplementedError()

        self.set_segments(segments, sr, preprocess)

    def set_segments(self, segments, sr, preprocess=None):
        """
        segments: an array of segment time-series.
        sr: sampling rate of segment data in `array`.
        preprocess: a preprocessing function or list of functions to be applied to all segments individually in-memory.
        """
        if not isinstance(segments, List):
            raise ValueError('Parameter `array` should be an iterable of segment time-series.')
        if not isinstance(sr, int):
            raise ValueError('Parameter `sr` should be a positive integer indicating the sampling rate to use.')
        
        if not isinstance(sr, List):
            sr = [sr] * len(segments)
        
        if preprocess:
            if not isinstance(preprocess, List):
                preprocess = [preprocess]
            
            for fx in preprocess:
                segments, sr = zip([fx(x, s) for x, s in zip(segments, sr)])
        
        if len(set(sr)) != 1:
            raise RuntimeError('All segments need to have the same sampling rate after preprocessing.')

        self.segments = segments
        self.in_sr = sr

    def set_channel_mask(self, mask):
        self.mask_channels = mask

    def crossfade(self, x):
        """
        x: an array of tensors. Concatenation is performed on the first dimension (axis=0). `x` should be padded by
        `segment_overlap`/2 on both sides along the concatenation dimension.
        """
        overlap = round(self.segment_overlap * self.in_sr)
        assert overlap % 2 == 0
        
        if overlap == 0:
            return np.concatenate(x, axis=0)
        
        window = np.hanning(2*overlap + 3)[overlap+2:-1].reshape([-1]+[1]*(x[0].ndim-1))
        
        pieces = []
        for i in range(len(x)):
            if i > 0:
                pieces.append(x[i-1][len(x[i-1])-overlap:]*window + x[i][:overlap]*window[::-1])
            if i == 0:
                pieces.append(x[i][overlap//2:len(x[i])-overlap])
            elif i == len(x)-1:
                pieces.append(x[i][overlap:len(x[i])-overlap//2])
            else:
                pieces.append(x[i][overlap:len(x[i])-overlap])
        
        return np.concatenate(pieces, axis=0)

    def pick_seglen(self, xs, segdur):
        """
        Pick subsegments of fixed duration `segdur` from the list of source segments `xs`.

        xs: list of source segments.
        segdur: duration of subsegment to extract from source segments, in seconds.
        """
        seglen = round(segdur * self.in_sr)
        overlap = round(self.segment_overlap * self.in_sr)

        if seglen < 1:
            raise ValueError('Segment duration should be at least one sample.')
        if self.segment_alignment not in ['center', 'start', 'end', 'max']:
            raise ValueError('`segment_alignment` should be either "start", "end", "center", or "max".')
        
        segments = []
        for x in xs:
            x = np.pad(x, ((math.floor(overlap/2), math.ceil(overlap/2)), (0, 0)))
            discard = len(x) - seglen - overlap
            if discard > 0:
                if self.segment_alignment == 'center':
                    segments.append(x[math.floor(discard/2):-math.ceil(discard/2)])
                elif self.segment_alignment == 'start':
                    segments.append(x[:-discard])
                elif self.segment_alignment == 'end':
                    segments.append(x[discard:])
                else:
                    raise NotImplementedError()
                    # segments.append(...)
            elif discard == 0:
                segments.append(x)
            else:
                raise RuntimeError('All segment time-series must be at least as long as the target segment duration.')

        return np.stack(segments, axis=0)

    def sequence(self, segdur, mode):
        """
        Create a sequence by shuffling subsegments of duration `segdur`.

        segdur: duration of subsegments to use, in seconds.
        mode: shuffling mode. -1 indicates natural subsegments of length `segment_maxdur`, 0 indicates subsegments of
                length `segdur` in their original order, positive integers indicate subsegments of length `segdur` in
                a randomly shuffled form, using the integer as the random seed.
        """
        if not isinstance(mode, int):
            raise ValueError('Parameter `mode` has to be an integer.')

        if mode < 0:
            segdur = self.segment_maxdur
        
        # original segment ordering
        x = self.pick_seglen(self.segments, segdur)
        
        if mode > 0:
            np.random.seed(mode)
            x = x[np.random.permutation(len(x))]
        
        x = self.crossfade(x)

        return x

    def desequence(self, x, segdur, mode):
        """
        Convert a sequence of response into individual subresponses corresponding to input segments of length `segdur`,
        with a margin of `size_margin` around the shared segment.

        x: sequence of response.
        segdur: duration of stimulus segments used to create the stimulus sequence, in seconds.
        mode: shuffling mode used to create the stimulus sequence. -1 indicates natural subsegments of length
                `segment_maxdur`, 0 indicates subsegments of length `segdur` in their original order, positive
                integers indicate subsegments of length `segdur` in a randomly shuffled form, using the integer
                as the random seed.
        """
        if not isinstance(mode, int) or mode < -1:
            raise ValueError('Parameter `mode` has to be an integer >= -1.')

        # reshape sequence into separate segments
        if mode < 0:
            seglen = round(self.segment_maxdur * self.out_sr)
        else:
            seglen = round(segdur * self.out_sr)
        x = x.reshape((len(x)//seglen, seglen, x.shape[-1]))

        # extract extra margins around shared segments in case of noncausality, etc.
        nmargn = math.ceil(self.size_margin / segdur)
        x = np.concatenate([np.roll(x, k, axis=0) for k in range(nmargn, -nmargn-1, -1)], axis=1)
        x = x[:, round(nmargn*segdur*self.out_sr-self.size_margin*self.out_sr):round(x.shape[1]-nmargn*segdur*self.out_sr+self.size_margin*self.out_sr)]

        # if natural-random comparison, extract relevant part of natural segment response
        if mode < 0:
            seglen = round(segdur * self.out_sr)
            discard = x.shape[1] - seglen
            if discard > 0:
                if self.segment_alignment == 'center':
                    x = x[:, math.floor(discard/2):-math.ceil(discard/2)]
                elif self.segment_alignment == 'start':
                    x = x[:, :-discard]
                elif self.segment_alignment == 'end':
                    x = x[:, discard:]
                else:
                    raise NotImplementedError()
                    # x = x[:, ...]
            elif discard < 0:
                raise RuntimeError('All segment time-series must be at least as long as the target segment duration.')

        if mode > 0:
            np.random.seed(mode)
            x = x[np.argsort(np.random.permutation(len(x)))]

        return x
    
    def write_sequences(self, path, fmt='audio'):
        """
        Write two stimulus sequences for all segment durations to disk.

        path: location of directory to write the stimulus sequences.
        fmt: format of sequence files. Currently supported formats are 'audio' and 'numpy'.
        """
        self.seq_A = dict([
            (segdur, self.sequence(segdur, 1))
            for segdur in self.segment_durs
        ])
        self.seq_B = dict([
            (segdur, self.sequence(segdur, 2 if self.comparison_method=='random-random' else -1))
            for segdur in self.segment_durs
        ])
        
        for segdur in self.seq_A:
            if fmt == 'audio':
                torchaudio.save(f'{path}/seq_A_{round(segdur*1000)}ms.wav', self.seq_A[segdur], self.in_sr)
                torchaudio.save(f'{path}/seq_B_{round(segdur*1000)}ms.wav', self.seq_B[segdur], self.in_sr)
            elif fmt == 'numpy':
                np.save(f'{path}/seq_A_{round(segdur*1000)}ms.npy', (self.seq_A[segdur], self.in_sr))
                np.save(f'{path}/seq_B_{round(segdur*1000)}ms.npy', (self.seq_B[segdur], self.in_sr))
            else:
                raise NotImplementedError()

    def read_responses(self, path, fmt='npy'):
        """
        Read two response sequences for all segment durations from disk.

        path: location of directory from which to read the response sequences.
        fmt: format of sequence files. Currently supported formats are 'numpy'.
        """
        resp_A, resp_B = dict(), dict()
        for segdur in self.segment_durs:
            if fmt == 'numpy':
                resp_A[segdur], _ = np.load(f'{path}/resp_A_{round(segdur*1000)}ms.npy')
                resp_B[segdur], _ = np.load(f'{path}/resp_B_{round(segdur*1000)}ms.npy')
            else:
                raise NotImplementedError()
        
        self.resp_A = resp_A
        self.resp_B = resp_B

    def batch_infer(self, x, segdur):
        """
        Infer model responses to input sequence `x` composed of segments of length `segdur`, performed in batches.

        x: input sequence to the model.
        segdur: duration of segments composing the sequence `x`, in seconds.
        """
        seglen = round(segdur * self.in_sr)
        nbatch = math.floor(self.size_block / segdur)
        ncontx = math.ceil(self.size_context / segdur)
        ntargt = nbatch - ncontx*2
        num_batch = math.ceil(x.shape[0] / ntargt)
        
        z = []
        x = np.concatenate((x[-ncontx*seglen:], x, x[:ncontx*seglen]), axis=0)
        for k in range(num_batch):
            # prepare batch input
            xb = x[(k*ntargt)*seglen:(k*ntargt+nbatch)*seglen]
            xb = torch.as_tensor(xb, dtype='float', device=self.device)
            
            # run model inference
            zb = self.model(xb)[ncontx*seglen:-ncontx*seglen, ..., self.mask_channels]
            z.append(zb)
        
        return np.concatenate(z, axis=0)

    def crossing(self, corrs):
        """
        Find when correlations (`corrs`) cross the specified `threshold`.

        corrs: correlation matrix for all segment durations, with shape [segments x channels]
        """
        corrs = np.nanmax(corrs, axis=0)

        seglens = np.log(self.segment_durs)
        x_intrp = np.linspace(seglens.min(), seglens.max(), 1000)
        
        corrs[~np.isfinite(corrs)] = 0.01
        corrs[corrs < 0.01] = 0.01
        #corrs = np.log(corrs)
        #thresh = np.log(thresh)
        
        xings = np.zeros(corrs.shape[1])
        for j in range(corrs.shape[1]):
            y_intrp = sp.interpolate.interp1d(
                seglens,
                np.convolve(np.pad(corrs[:, j], [(1, 1)], 'edge'), [0.15, 0.7, 0.15], 'valid')
            )(x_intrp)
            
            passthresh = np.where(y_intrp >= self.thresh_ccc)[0]
            xings[j] = round(np.exp(x_intrp[passthresh[0]]), 1) if len(passthresh) > 0 else np.nan
        
        return xings
    
    def batch_corr(self, seq_A, seq_B):
        """
        Compute correlation of `seq_A` and `seq_B` along the first axis, performed in batches if `size_batch_corr` is
        set. Use batch processing if the large number of segments is causing memory issues.

        seq_A: input sequence A.
        seq_B: input sequence B.
        """
        def corr(a, b, axis=None):
            """Compute Pearson's correlation along specified axis."""
            a_mean = a.mean(axis=axis, keepdims=True)
            b_mean = b.mean(axis=axis, keepdims=True)
            a, b = (a - a_mean), (b - b_mean)
            
            a_sum2 = (a ** 2).sum(axis=axis, keepdims=True)
            b_sum2 = (b ** 2).sum(axis=axis, keepdims=True)
            if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
                a, b = (a / np.sqrt(a_sum2)), (b / np.sqrt(b_sum2))
            elif isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                a, b = (a / torch.sqrt(a_sum2)), (b / torch.sqrt(b_sum2))
            else:
                raise TypeError(f'Incompatible types: {type(a)} and {type(b)}')
            
            return (a * b).sum(axis=axis)
        
        if self.size_batch_corr:
            return np.concatenate([
                corr(
                    torch.from_numpy(seq_A[:, k*self.size_batch_corr:(k+1)*self.size_batch_corr]).to(self.device),
                    torch.from_numpy(seq_B[:, k*self.size_batch_corr:(k+1)*self.size_batch_corr]).to(self.device),
                    axis=0
                ).cpu().numpy() for k in range(math.ceil(seq_A.shape[1] / self.size_batch_corr))
            ], axis=0)
        else:
            return corr(
                torch.from_numpy(seq_A).to(self.device),
                torch.from_numpy(seq_B).to(self.device),
                axis=0
            ).cpu().numpy()

    def cross_context_corr(self, segdur):
        """
        Compute cross-context correlation for specified segment duration of `segdur`. If model is provided, uses
        the model to infer responses to input sequences. If model is set to None, uses loaded response sequences.

        segdur: duration of segment to use for calculating cross-context correlation, in seconds.
        """
        print(f'>> {segdur*1000}ms', flush=True)
        
        # first segment ordering
        if self.model:
            seq_A = self.sequence(segdur, 1)
            seq_A = self.batch_infer(seq_A, segdur)
            seq_A = self.desequence(seq_A, segdur, 1)
        elif self.resp_A:
            seq_A = self.desequence(self.resp_A[segdur], segdur, 1)
        else:
            raise RuntimeError('You should specify either a model, or load the responses before analysis.')
        
        # second segment ordering
        if self.model:
            seq_B = self.sequence(segdur, 2 if self.comparison_method=='random-random' else -1)
            seq_B = self.batch_infer(seq_B, segdur)
            seq_B = self.desequence(seq_B, segdur, 2 if self.comparison_method=='random-random' else -1)
        elif self.resp_B:
            seq_B = self.desequence(self.resp_B[segdur], segdur, 2 if self.comparison_method=='random-random' else -1)
        
        # cross-context correlation
        corr = self.batch_corr(seq_A, seq_B)
        # convert nan correlations to zero
        corr[~np.isfinite(corr)] = 0

        return corr

    def run(self):
        """
        Run TCI analysis and store returned results.
        """
        corrs = [self.cross_context_corr(segdur) for segdur in self.segment_durs]
        self.integration_windows = self.crossing(corrs)

        if self.output_path:
            with open(self.output_path, 'wb') as f:
                pickle.dump({
                    'cross_context_corrs': corrs,
                    'integration_windows': self.integration_windows
                }, f)
        
        return self.integration_windows
