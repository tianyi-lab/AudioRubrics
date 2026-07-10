# Copyright 2025 - Omni-R1 (adapted from PAPO)
# Licensed under the Apache License, Version 2.0

"""Audio corruption functions for perception-aware training (KL-PRCP).

Analog of PAPO's random_patch_blackening for the audio domain.
All functions operate on raw numpy waveforms (float32, 16kHz mono)
and preserve waveform length exactly.
"""

import numpy as np


def random_segment_zeroing(waveform, segment_ms=50, zero_prob=0.6, sr=16000):
    """Zero out random time segments of a waveform.

    Direct analog of PAPO's random_patch_blackening: divides the waveform
    into fixed-size segments and zeros each with probability ``zero_prob``.

    Args:
        waveform: 1-D numpy float32 array.
        segment_ms: Segment duration in milliseconds.
        zero_prob: Probability of zeroing each segment.
        sr: Sample rate (default 16000).

    Returns:
        Corrupted waveform (same shape and dtype as input).
    """
    waveform = waveform.copy()
    segment_samples = int(sr * segment_ms / 1000)
    n_samples = len(waveform)
    for start in range(0, n_samples, segment_samples):
        if np.random.rand() < zero_prob:
            end = min(start + segment_samples, n_samples)
            waveform[start:end] = 0.0
    return waveform


def specaugment_corruption(waveform, num_freq_masks=2, freq_mask_width=20,
                           num_time_masks=2, time_mask_fraction=0.1, sr=16000,
                           n_fft=512, hop_length=160):
    """SpecAugment-style corruption via STFT masking and reconstruction.

    Computes the STFT, applies frequency and time masks, then reconstructs
    via iSTFT.  More expensive than segment zeroing but better aligned with
    the model's mel-spectrogram front-end.

    Args:
        waveform: 1-D numpy float32 array.
        num_freq_masks: Number of frequency masks to apply.
        freq_mask_width: Maximum width (in bins) of each frequency mask.
        num_time_masks: Number of time masks to apply.
        time_mask_fraction: Maximum fraction of time frames to mask.
        sr: Sample rate.
        n_fft: FFT size.
        hop_length: Hop length for STFT.

    Returns:
        Corrupted waveform (same length as input).
    """
    orig_len = len(waveform)

    # STFT
    stft = np.fft.rfft(
        np.lib.stride_tricks.sliding_window_view(
            np.pad(waveform, (0, n_fft - len(waveform) % n_fft)),
            n_fft,
        )[::hop_length],
    )
    n_time, n_freq = stft.shape

    # Frequency masks
    for _ in range(num_freq_masks):
        width = np.random.randint(1, freq_mask_width + 1)
        start = np.random.randint(0, max(n_freq - width, 1))
        stft[:, start:start + width] = 0.0

    # Time masks
    max_time_width = max(int(n_time * time_mask_fraction), 1)
    for _ in range(num_time_masks):
        width = np.random.randint(1, max_time_width + 1)
        start = np.random.randint(0, max(n_time - width, 1))
        stft[start:start + width, :] = 0.0

    # iSTFT (overlap-add reconstruction)
    frames = np.fft.irfft(stft, n=n_fft)
    reconstructed = np.zeros(n_time * hop_length + n_fft, dtype=np.float32)
    window = np.hanning(n_fft).astype(np.float32)
    window_sum = np.zeros_like(reconstructed)
    for i, frame in enumerate(frames):
        pos = i * hop_length
        reconstructed[pos:pos + n_fft] += frame * window
        window_sum[pos:pos + n_fft] += window ** 2
    window_sum = np.maximum(window_sum, 1e-8)
    reconstructed /= window_sum
    return reconstructed[:orig_len].astype(np.float32)


def random_noise_injection(waveform, noise_level=0.3):
    """Add Gaussian noise scaled to a fraction of the waveform's RMS.

    Args:
        waveform: 1-D numpy float32 array.
        noise_level: Noise amplitude relative to waveform RMS.

    Returns:
        Corrupted waveform (same shape and dtype as input).
    """
    rms = np.sqrt(np.mean(waveform ** 2)) + 1e-8
    noise = np.random.randn(*waveform.shape).astype(np.float32)
    return (waveform + noise_level * rms * noise).astype(np.float32)


def random_audio_corruption(waveform, method="segment_zeroing", **kwargs):
    """Dispatcher for audio corruption methods.

    Args:
        waveform: 1-D numpy float32 array (16kHz mono).
        method: One of "segment_zeroing", "specaugment", "noise".
        **kwargs: Forwarded to the chosen corruption function.

    Returns:
        Corrupted waveform (same shape and dtype as input).
    """
    if method == "segment_zeroing":
        return random_segment_zeroing(waveform, **kwargs)
    elif method == "specaugment":
        return specaugment_corruption(waveform, **kwargs)
    elif method == "noise":
        return random_noise_injection(waveform, **kwargs)
    else:
        raise ValueError(f"Unknown audio corruption method: {method}")
