import os
import numpy as np

import torch
import torchaudio


def wav2spec_fx(out_sr, freqbins=128, top_db=70):
    """
    Returns a function that converts the audio waveform to Mel-spectrogram.

    out_sr: sampling rate of output spectrogram, in Hz.
    freqbins: number of frequency bins in the output spectrogram.
    top_db: maximum difference between highest and lowest power in the spectrogram.
    """
    def func(x, in_sr):
        parser = torch.nn.Sequential(
            torchaudio.transforms.MelSpectrogram(in_sr, n_fft=1024, hop_length=int(in_sr/out_sr), f_min=20, f_max=8_000, n_mels=freqbins, power=2.0),
            torchaudio.transforms.AmplitudeToDB('power', top_db=top_db),
            type("Normalize", (torch.nn.Module,), dict(forward=lambda self, x: (x - x.max()).squeeze(0).T.float() / top_db + 1))()
        )

        return parser(x), out_sr

    return func


def _sox_fx(x, in_sr, tfm):
    """
    Returns a function that applies the SoX effects specified in `tfm` to an input signal `x`, assuming a
    sampling rate of `in_sr` for the input.

    x: input audio signal with shape [time, channel].
    in_sr: sampling rate of input waveform, in Hz.
    tfm: list of SoX transformations to be applied to the input.
    """
    x, out_sr = torchaudio.sox_effects.apply_effects_tensor(x.T.float(), in_sr, tfm)
    return x.T, out_sr


def audio_resample_fx(to_sr, channels=1):
    """
    to_sr: target sampling rate, in Hz.
    channels: output channels for audio, set to 1 to convert stereo audio to mono.
    """
    def func(x, in_sr):
        tfm = [['rate', str(to_sr)], ['channels', str(channels)]]
        return _sox_fx(x, in_sr, tfm)

    return func


def audio_tempo_fx(scale_factor, audio_type='s'):
    """
    Returns a function that time-stretches the audio by a fixed scaling factor, without chaning the pitch.
    To optimize performance on different types of audio, the `audio_type` parameter can be set. Supported
    types by SoX are: 's' (speech), 'm' (music), 'l' (linear).

    scale_factor: scaling factor used to stretch audio.
    audio_type: type of input audio, used to optimize stretching parameters for best results.
    """
    def func(x, in_sr):
        tfm = [['tempo', '-'+audio_type, str(scale_factor)], ['rate', str(in_sr)]]
        return _sox_fx(x, in_sr, tfm)

    return func


def audio_speed_fx(scale_factor):
    """
    Returns a function that time-stretches the audio by a fixed scaling factor, without preserving pitch.

    scale_factor: factor by which to scale the audio.
    """
    def func(x, in_sr):
        tfm = [['speed', str(scale_factor)], ['rate', str(in_sr)]]
        return _sox_fx(x, in_sr, tfm)

    return func


def audio_pitch_fx(semitones):
    """
    Returns a function that shifts the pitch of audio, by a fixed number of semitones.

    semitones: amount to shift pitch, in semitones.
    """
    def func(x, in_sr):
        tfm = [['pitch', str(semitones)], ['rate', str(in_sr)]]
        return _sox_fx(x, in_sr, tfm)

    return func


def audio_reverb_fx(reverberance, room_scale):
    """
    Returns a function that adds reverberance to audio.

    reverberance: strength of reverberance.
    room_scale: size of room to simulate reverberance in.
    """
    def func(x, in_sr):
        tfm = [['reverb', '--wet-only', str(reverberance), '50', str(room_scale)], ['channels', '1']]
        return _sox_fx(x, in_sr, tfm)

    return func


def audio_filter_fx(cutoff_low, cutoff_high, bandpass=True):
    """
    Returns a function that applies frequency filtering to audio, including lowpass, highpass, bandpass,
    and bandstop filtering. To perform highpass filtering set `cutoff_high` to None; for a lowpass filter
    set `cutoff_low` to None; for a bandpass filter set both cutoff frequencies and `bandpass` to True;
    for a bandstop filter set both cutoff frequencies and `bandpass` to False.

    cutoff_low: lower cutoff frequency of filtering, in Hz.
    cutoff_high: higher cutoff frequency of filtering, in Hz.
    bandpass: whether to apply bandpass, or bandtop filtering.
    """
    if cutoff_low is None and cutoff_high is None:
        raise ValueError('At least one of `cutoff_low` and `cutoff_high` has to be set.')
    
    def func(x, in_sr):
        if cutoff_low is None:
            tfm = [['sinc', '-'+str(cutoff_high)]]
        elif cutoff_high is None:
            tfm = [['sinc', str(cutoff_low)]]
        elif bandpass:
            tfm = [['sinc', str(cutoff_low)+'-'+str(cutoff_high)]]
        else:
            tfm = [['sinc', str(cutoff_high)+'-'+str(cutoff_low)]]
        
        return _sox_fx(x, in_sr, tfm)
    
    return func


def audio_inject_noise_fx(noise_sources, random=True):
    """
    Returns a function that injects additive noise to audio, selected from a list of noise sources.
    """
    # tfm = [['rate', str(in_sr)], ['channels', '1']]
    # # repeat to cover full speech
    # dur_speech = mix_audio.shape[1] / mix_sr
    # dur_noise = noise_audio.shape[1] / noise_sr
    # count = math.ceil(dur_speech / dur_noise)
    # tfm.append(['repeat', str(count)])
    # # trim to same length as speech
    # tfm.append(['trim', '0', str(dur_speech)])
    # # set volume
    # snr_db = np.random.uniform(*self.mod_config['BG_SNR'])
    # tfm.append(['norm', str(-3 - snr_db)])
    # # process audio
    # noise_audio, noise_sr = torchaudio.sox_effects.apply_effects_tensor(noise_audio, noise_sr, tfm)
    # 
    # mix_audio = (mix_audio + noise_audio[:, :mix_audio.shape[1]]) / np.sqrt(2)
    # mix_audio, mix_sr = torchaudio.sox_effects.apply_effects_tensor(mix_audio, mix_sr, [['norm', '-3']])

    raise NotImplementedError()
