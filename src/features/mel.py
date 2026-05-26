import numpy as np


def _hz_to_mel(frequencies, htk=False):
    frequencies = np.asanyarray(frequencies, dtype=np.float64)
    if htk:
        return 2595.0 * np.log10(1.0 + frequencies / 700.0)

    f_sp = 200.0 / 3
    mels = frequencies.copy()
    mels /= f_sp
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    log_t = frequencies >= min_log_hz
    mels[log_t] = min_log_mel + np.log(frequencies[log_t] / min_log_hz) / logstep
    return mels


def _mel_to_hz(mels, htk=False):
    mels = np.asanyarray(mels, dtype=np.float64)
    if htk:
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

    f_sp = 200.0 / 3
    freqs = mels.copy()
    freqs *= f_sp
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    log_t = mels >= min_log_mel
    freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
    return freqs


def _mel_frequencies(n_mels, fmin, fmax, htk=False):
    min_mel = _hz_to_mel(fmin, htk=htk)
    max_mel = _hz_to_mel(fmax, htk=htk)
    return _mel_to_hz(np.linspace(min_mel, max_mel, n_mels), htk=htk)


def build_mel_basis(sampling_rate, n_fft, num_mels, fmin, fmax, htk=False, norm="slaney"):
    if fmax is None:
        fmax = float(sampling_rate) / 2
    fftfreqs = np.fft.rfftfreq(n=n_fft, d=1.0 / sampling_rate)
    mel_f = _mel_frequencies(num_mels + 2, fmin, fmax, htk=htk)
    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)

    lower = -ramps[:-2] / fdiff[:-1, np.newaxis]
    upper = ramps[2:] / fdiff[1:, np.newaxis]
    weights = np.maximum(0, np.minimum(lower, upper))
    if norm == "slaney":
        enorm = 2.0 / (mel_f[2 : num_mels + 2] - mel_f[:num_mels])
        weights *= enorm[:, np.newaxis]
    elif norm is not None:
        raise ValueError(f"Unsupported mel norm: {norm}")
    return weights.astype(np.float32, copy=False)
