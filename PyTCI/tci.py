import os
import glob
import math
import pickle
from typing import Iterable

import numpy as np
from scipy.interpolate import interp1d

import torch
import torchaudio


SEGMENT_DURS = np.array([
    0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.20, 0.24, 0.28, 0.34,
    0.40, 0.48, 0.58, 0.70, 0.84, 1.00, 1.20, 1.44, 1.72, 2.06, 2.48
])


@torch.no_grad()
def load_stimuli(path, fmt='audio', process=[]):
    """
    Load source stimuli from disk and preprocess them individually (optional).

    path: if a string, `path` is a directory containing audio files; if a list, `path` is a list of audio files.
    fmt: format of stimulus files. Currently accepted formats are 'audio' and 'numpy'.
    preprocess: a preprocessing function or list of functions to be applied to all stimuli individually in-memory.
    """
    if not isinstance(path, (str, Iterable)):
        raise ValueError('Parameter `path` should either be a string pointing to a directory, or a list of audio files.')

    if isinstance(path, str):
        path = glob.glob(os.path.join(path, '*'))
    
    if fmt == 'audio':
        stimuli, sr = zip(*[torchaudio.load(file, channels_first=False) for file in path])
    elif fmt == 'numpy':
        stimuli, sr = zip(*[np.load(file, allow_pickle=True) for file in path])
    else:
        raise NotImplementedError()

    return process_stimuli(stimuli, sr, process)


@torch.no_grad()
def process_stimuli(stimuli, sr, process=[]):
    """
    stimuli: an array of source stimulus time-series.
    sr: sampling rate of stimulus data.
    preprocess: a preprocessing function or list of functions to be applied to all stimuli individually in-memory.
    """
    if not isinstance(stimuli, Iterable):
        raise ValueError('Parameter `array` should be an iterable of stimulus time-series.')
    if not (isinstance(sr, int) or isinstance(sr, Iterable) and np.all([isinstance(s, int) for s in sr])):
        raise ValueError('Parameter `sr` should be a positive integer or a list of positive integers.')
    
    if not isinstance(sr, Iterable):
        sr = [sr] * len(stimuli)
    
    stimuli = [torch.as_tensor(x, dtype=float) for x in stimuli]
    
    if process:
        if not isinstance(process, Iterable):
            process = [process]
        
        for fx in process:
            stimuli, sr = zip(*[fx(x, s) for x, s in zip(stimuli, sr)])
    
    if len(set(sr)) == 1:
        sr = sr[0]
    else:
        raise RuntimeError('All stimuli need to have the same sampling rate after preprocessing.')
    
    return stimuli, sr


@torch.no_grad()
def masked_model(model, mask):
    return lambda x: model(x)[..., mask]


@torch.no_grad()
def crossfade(x, sr, segment_overlap=0.020):
    """
    x: an array of tensors. Concatenation is performed on the first dimension (axis=0). `x` should be padded by
    `segment_overlap`/2 on both sides along the concatenation dimension.
    """
    overlap = round(segment_overlap * sr)
    assert overlap % 2 == 0
    
    if overlap == 0:
        return torch.cat(x, dim=0)
    
    window = torch.hann_window(2*overlap + 3)[overlap+2:-1].reshape([-1]+[1]*(x[0].ndim-1))
    
    pieces = []
    for i in range(len(x)):
        if i > 0:
            pieces.append(x[i-1][len(x[i-1])-overlap:]*window + x[i][:overlap]*window.flip(0))
        if i == 0:
            pieces.append(x[i][overlap//2:len(x[i])-overlap])
        elif i == len(x)-1:
            pieces.append(x[i][overlap:len(x[i])-overlap//2])
        else:
            pieces.append(x[i][overlap:len(x[i])-overlap])
    
    return torch.cat(pieces, dim=0)


@torch.no_grad()
def extract_segments(stimuli, sr, segment_dur, segment_overlap=0.020, segment_alignment='center'):
    """
    Pick segments of fixed duration `segdur` from the list of source `stimuli`.

    stimuli: list of source segments.
    segdur: duration of segment to extract from source stimuli, in seconds.
    """
    seglen = round(segment_dur * sr)
    overlap = round(segment_overlap * sr)

    if seglen < 1:
        raise ValueError('Segment duration should be at least one sample.')
    if segment_alignment not in ['center', 'start', 'end', 'max']:
        raise ValueError('`segment_alignment` should be either "start", "end", "center", or "max".')
    
    segments = []
    for x in stimuli:
        x = torch.nn.functional.pad(x, (0, 0, math.floor(overlap/2), math.ceil(overlap/2)))
        discard = len(x) - seglen - overlap
        if discard > 0:
            if segment_alignment == 'center':
                segments.append(x[math.floor(discard/2):-math.ceil(discard/2)])
            elif segment_alignment == 'start':
                segments.append(x[:-discard])
            elif segment_alignment == 'end':
                segments.append(x[discard:])
            else:
                raise NotImplementedError()
                # segments.append(...)
        elif discard == 0:
            segments.append(x)
        else:
            raise RuntimeError('All stimulus time-series must be at least as long as the target segment duration.')
    
    return torch.stack(segments, dim=0)


@torch.no_grad()
def generate_sequence(stimuli, sr, segment_dur, seed=0, segment_overlap=0.020, segment_alignment='center'):
    """
    Create a sequence by shuffling chopped segments of duration `segdur`.

    segdur: duration of segments to use, in seconds.
    mode: shuffling mode. -1 indicates natural segments of length `segment_maxdur`, 0 indicates segments of
            length `segdur` in their original order, positive integers indicate segments of length `segdur` in
            a randomly shuffled form, using the integer as the random seed.
    """
    if not isinstance(seed, int):
        raise ValueError('Parameter `seed` has to be a non-negative integer.')
    
    if isinstance(segment_dur, Iterable):
        return [
            generate_sequence(
                stimuli=stimuli,
                sr=sr,
                segment_dur=dur,
                seed=seed,
                segment_overlap=segment_overlap,
                segment_alignment=segment_alignment
            ) for dur in segment_dur
        ]
    
    # original segment ordering
    x = extract_segments(stimuli, sr, segment_dur, segment_overlap, segment_alignment)
    
    if seed > 0:
        np.random.seed(seed)
        x = x[np.random.permutation(len(x))]
    
    x = crossfade(x, sr, segment_overlap)

    return x


@torch.no_grad()
def generate_sequence_pair(stimuli, sr, segment_dur, segment_overlap=0.020, segment_alignment='center', comparison_method='random-random', natural_dur=4.0):
    """
    Create a sequence by shuffling segments of duration `segdur`.

    segdur: duration of segments to use, in seconds.
    mode: shuffling mode. -1 indicates natural segments of length `segment_maxdur`, 0 indicates segments of
            length `segdur` in their original order, positive integers indicate segments of length `segdur` in
            a randomly shuffled form, using the integer as the random seed.
    """
    if comparison_method not in ['random-random', 'natural-random']:
        raise ValueError('Parameter `comparison_method` has to either be "random-random" or "natural-random".')
    
    seq_A = generate_sequence(
        stimuli=stimuli,
        sr=sr,
        segment_dur=segment_dur,
        segment_overlap=segment_overlap,
        segment_alignment=segment_alignment,
        seed=1
    )
    
    seq_B = generate_sequence(
        stimuli=stimuli,
        sr=sr,
        segment_dur=segment_dur if comparison_method=='random-random' else natural_dur,
        segment_overlap=segment_overlap,
        segment_alignment=segment_alignment,
        seed=2 if comparison_method=='random-random' else -1
    )
    
    if isinstance(segment_dur, Iterable) and comparison_method=='natural-random':
        seq_B = [seq_B] * len(seq_A)
    
    if isinstance(segment_dur, Iterable):
        return list(zip(seq_A, seq_B))
    else:
        return (seq_A, seq_B)


@torch.no_grad()
def rearrange_sequence(sequence, sr, segment_dur, seed=0, segment_alignment='center', natural_dur=4.0, margin=1.0):
    """
    Convert a sequence of response into individual subresponses corresponding to input segments of length `segdur`,
    with a margin of `size_margin` around the shared segment.

    x: sequence of response.
    segdur: duration of stimulus segments used to create the TCI sequence, in seconds.
    mode: shuffling mode used to create the stimulus sequence. -1 indicates natural segments of length
            `segment_maxdur`, 0 indicates segments of length `segdur` in their original order, positive
            integers indicate segments of length `segdur` in a randomly shuffled form, using the integer
            as the random seed.
    """
    if not isinstance(seed, int) or seed < -1:
        raise ValueError('Parameter `seed` has to be an integer >= -1.')
    
    if isinstance(segment_dur, Iterable):
        assert len(sequence) == len(segment_dur)
        return [
            rearrange_sequence(
                sequence=sequence[i],
                sr=sr,
                segment_dur=segment_dur[i],
                seed=seed,
                segment_alignment=segment_alignment,
                natural_dur=natural_dur,
                margin=margin
            ) for i in range(len(segment_dur))
        ]
    
    target_dur = segment_dur
    if seed < 0:
        segment_dur = natural_dur
    
    # reshape sequence into separate segments
    seglen_t = round(target_dur * sr)
    seglen = round(segment_dur * sr)
    segments = sequence.reshape((len(sequence)//seglen, seglen, sequence.shape[-1]))

    # extract extra margins around shared segments in case of noncausality, etc.
    nmargn = math.ceil(margin / segment_dur)
    segments = torch.cat([torch.roll(segments, k, 0) for k in range(nmargn, -nmargn-1, -1)], dim=1)
    segments = segments[:, round(nmargn*segment_dur*sr-margin*sr):round(segments.shape[1]-nmargn*segment_dur*sr+margin*sr)]

    # if natural-random comparison, extract relevant part of natural segment response
    if seed < 0:
        discard = segments.shape[1] - seglen_t - 2*round(margin * sr)
        if discard > 0:
            if segment_alignment == 'center':
                segments = segments[:, math.floor(discard/2):-math.ceil(discard/2)]
            elif segment_alignment == 'start':
                segments = segments[:, :-discard]
            elif segment_alignment == 'end':
                segments = segments[:, discard:]
            else:
                raise NotImplementedError()
                # x = x[:, ...]
        elif discard < 0:
            raise RuntimeError('All segment time-series must be at least as long as the target segment duration.')

    if seed > 0:
        np.random.seed(seed)
        segments = segments[np.argsort(np.random.permutation(len(segments)))]

    return segments


@torch.no_grad()
def rearrange_sequence_pair(sequence_pair, sr, segment_dur, segment_alignment='center', comparison_method='random-random', natural_dur=4.0, margin=1.0):
    """
    Convert a sequence of response into individual subresponses corresponding to input segments of length `segdur`,
    with a margin of `size_margin` around the shared segment.

    x: sequence of response.
    segdur: duration of stimulus segments used to create the TCI sequence, in seconds.
    mode: shuffling mode used to create the stimulus sequence. -1 indicates natural segments of length
            `segment_maxdur`, 0 indicates segments of length `segdur` in their original order, positive
            integers indicate segments of length `segdur` in a randomly shuffled form, using the integer
            as the random seed.
    """
    if isinstance(segment_dur, Iterable):
        seq_A, seq_B = list(zip(*sequence_pair))
    else:
        seq_A, seq_B = sequence_pair
    
    SAR_A = rearrange_sequence(
        sequence=seq_A,
        sr=sr,
        segment_dur=segment_dur,
        segment_alignment=segment_alignment,
        natural_dur=natural_dur,
        margin=margin,
        seed=1
    )
    
    SAR_B = rearrange_sequence(
        sequence=seq_B,
        sr=sr,
        segment_dur=segment_dur,
        segment_alignment=segment_alignment,
        natural_dur=natural_dur,
        margin=margin,
        seed=2 if comparison_method=='random-random' else -1
    )
    
    if isinstance(segment_dur, Iterable):
        return list(zip(SAR_A, SAR_B))
    else:
        return (SAR_A, SAR_B)


@torch.no_grad()
def infer(model, sequence, segment_dur, in_sr, out_sr, block_size=48., context_size=8., device='cpu'):
    """
    Infer model responses to input sequence `x` composed of segments of length `segdur`, performed in batches.

    x: input sequence to the model.
    segdur: duration of segments composing the sequence `x`, in seconds.
    """
    seglen_in = round(segment_dur * in_sr)
    seglen_out = round(segment_dur * out_sr)
    nbatch = math.floor(block_size / segment_dur)
    ncontx = math.ceil(context_size / segment_dur)
    ntargt = nbatch - ncontx*2
    num_batch = math.ceil(len(sequence) / ntargt / seglen_in)
    
    response = []
    sequence = torch.cat((
        sequence[-ncontx*seglen_in:], sequence, sequence[:ncontx*seglen_in]), dim=0
    )
    for k in range(num_batch):
        # batch input
        seq_batch = sequence[(k*ntargt)*seglen_in:(k*ntargt+nbatch)*seglen_in].float().to(device)

        # run model inference
        res_batch = model(seq_batch)[ncontx*seglen_out:-ncontx*seglen_out]
        response.append(res_batch)
    
    return torch.cat(response, dim=0)


@torch.no_grad()
def infer_sequence_pair(model, sequence_pair, segment_dur, **kwargs):
    """
    Infer model responses to input sequence `x` composed of segments of length `segdur`, performed in batches.

    x: input sequence to the model.
    segdur: duration of segments composing the sequence `x`, in seconds.
    """
    if isinstance(segment_dur, Iterable):
        return [
            infer_sequence_pair(
                model, sequence_pair[i], segdur, **kwargs
            ) for i, segdur in enumerate(segment_dur)
        ]
    
    return (
        infer(model, sequence_pair[0], segment_dur, **kwargs),
        infer(model, sequence_pair[1], segment_dur, **kwargs)
    )


@torch.no_grad()
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
    
    rvals = (a * b).sum(axis=axis)
    rvals[~torch.isfinite(rvals)] = 0

    return rvals


@torch.no_grad()
def batch_corr(seq_A, seq_B, batch_size=None, device='cpu'):
    """
    Compute correlation of `seq_A` and `seq_B` along the first axis, performed in batches if `batch_size` is
    set. Use batch processing if the large number of segments is causing memory issues.

    seq_A: input sequence A.
    seq_B: input sequence B.
    """
    if batch_size:
        return torch.cat([
            corr(
                seq_A[:, k*batch_size:(k+1)*batch_size].to(device),
                seq_B[:, k*batch_size:(k+1)*batch_size].to(device),
                axis=0
            ).cpu() for k in range(math.ceil(seq_A.shape[1] / batch_size))
        ], axis=0).numpy()
    else:
        return corr(
            seq_A.to(device),
            seq_B.to(device),
            axis=0
        ).cpu().numpy()


@torch.no_grad()
def cross_context_corrs(sequence_pair, batch_size=None, device='cpu'):
    return [
        batch_corr(
            seq_A,
            seq_B,
            batch_size=batch_size,
            device=device
        ) for seq_A, seq_B in sequence_pair
    ]


@torch.no_grad()
def estimate_integration_window(cc_corrs, segment_durs, threshold=0.75):
    """
    Find when correlations (`cc_corrs`) cross the specified `threshold`.

    corrs: correlation matrix for all segment durations, with shape [segments x channels]
    """
    if not isinstance(segment_durs, Iterable):
        cc_corrs = [cc_corrs]
        segment_durs = [segment_durs]
    
    cc_corrs = np.stack([np.nanmax(c, axis=0) for c in cc_corrs], axis=0)

    seglens = np.log(segment_durs)
    x_intrp = np.linspace(seglens.min(), seglens.max(), 1000)
    
    integration_windows = np.zeros(cc_corrs.shape[1])
    for j in range(cc_corrs.shape[1]):
        y_intrp = interp1d(
            seglens,
            np.convolve(np.pad(cc_corrs[:, j], [(1, 1)], 'edge'), [0.15, 0.7, 0.15], 'valid')
        )(x_intrp)
        
        passthresh = np.where(y_intrp >= threshold)[0]
        integration_windows[j] = round(np.exp(x_intrp[passthresh[0]]), 3) if len(passthresh) > 0 else np.nan
    
    return integration_windows
