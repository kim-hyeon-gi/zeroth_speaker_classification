from glob import iglob

import librosa
import numpy as np
from tqdm.notebook import tqdm


def get_numpy_from_nonfixed_2d_array(aa, fixed_length, padding_value=0):
    rows = []
    for a in aa:
        rows.append(
            np.pad(a, (0, fixed_length), "constant", constant_values=padding_value)[
                :fixed_length
            ]
        )
    return rows


def make_padding_dataset_using_librosa():
    dataset = []
    labels = []
    for i in tqdm(range(104, 219)):
        for file in iglob("./dataset/" + str(i) + "/*.flac", recursive=True):
            scale, sr = librosa.load(file)
            mel_spectrogram = librosa.feature.melspectrogram(
                y=scale,
                n_fft=512,
                hop_length=512,
                n_mels=40,
            )
            log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
            if np.shape(log_mel_spectrogram)[1] > 500:
                log_mel_spectrogram = log_mel_spectrogram[:, :500]
            dataset.append(
                get_numpy_from_nonfixed_2d_array(log_mel_spectrogram, 500, 0)
            )
            labels.append(i)

    dataset = np.array(dataset)
    labels = np.array(labels)
    labels = labels - 104
    np.save("./padding_dataset.npy", dataset)
    np.save("./padding_labels.npy", labels)


def make_dataset_using_log_mel_spectrogram():
    sample_rate = 16000
    NFFT = 512
    nfilt = 40
    frame_size = 0.025
    frame_stride = 0.01
    frame_length, frame_step = (
        frame_size * sample_rate,
        frame_stride * sample_rate,
    )
    low_freq_mel = 0
    high_freq_mel = 2595 * np.log10(1 + (sample_rate / 2) / 700)  # Convert Hz to Mel
    mel_points = np.linspace(
        low_freq_mel, high_freq_mel, nfilt + 2
    )  # Equally spaced in Mel scale
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)  # Convert Mel to Hz
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)
    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    hamming_window = np.array(
        [
            0.54 - 0.46 * np.cos((2 * np.pi * n) / (frame_length - 1))
            for n in range(int(frame_length))
        ]
    )
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])  # left
        f_m = int(bin[m])  # center
        f_m_plus = int(bin[m + 1])  # right
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

    dataset = []
    labels = []
    for i in tqdm(range(104, 219)):
        for file in iglob("./dataset/" + str(i) + "/*.flac", recursive=True):
            signal, sample_rate = librosa.load(file)

            signal_length = len(signal)
            frame_length = int(round(frame_length))
            frame_step = int(round(frame_step))
            num_frames = int(
                np.ceil(float(np.abs(signal_length - frame_length)) / frame_step)
            )
            signal_length = len(signal)
            frame_length = int(round(frame_length))
            frame_step = int(round(frame_step))
            num_frames = int(
                np.ceil(float(np.abs(signal_length - frame_length)) / frame_step)
            )
            pad_signal_length = num_frames * frame_step + frame_length
            z = np.zeros((pad_signal_length - signal_length))
            pad_signal = np.append(signal, z)
            indices = (
                np.tile(np.arange(0, frame_length), (num_frames, 1))
                + np.tile(
                    np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)
                ).T
            )
            frames = pad_signal[indices.astype(np.int32, copy=False)]
            frames *= hamming_window

            dft_frames = np.fft.rfft(frames, NFFT)
            mag_frames = np.absolute(dft_frames)
            pow_frames = (1.0 / NFFT) * ((mag_frames) ** 2)

            filter_banks = np.dot(pow_frames, fbank.T)
            filter_banks = np.where(
                filter_banks == 0, np.finfo(float).eps, filter_banks
            )  # Numerical Stability
            filter_banks = 20 * np.log10(filter_banks)  # dB
            if len(filter_banks) < 500:
                continue
            dataset.append(filter_banks[:500].tolist())
            labels.append(i)
    dataset = np.array(dataset)
    labels = np.array(labels)
    labels = labels - 104
    np.save("./no_padding_dataset.npy", dataset)
    np.save("./no_padding_labels.npy", labels)


make_dataset_using_log_mel_spectrogram()
