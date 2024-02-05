import numpy as np
import pandas as pd
import scipy


def segmentation(data, size_of_segment, step_size):
    # Size of each segment must have a number 2^n+1 of data points
    # List where we will store the segments of the signal
    segments = []

    # Iterate over the signal with a step size corresponding to the overlap
    for i in range(0, len(data), step_size):
        # We need to handle the case of the last segment which might not have enough values for segmentation.
        # In that case, we don't store it
        if i + size_of_segment < len(data):
            segment = data[i:i + size_of_segment]
            segments.append(segment)
    return segments


def smooth(sig, n):
    # Make an array containing only zeros and with length = length_of_signal + 2*n
    extremes_zeros = np.zeros(len(sig) + 2 * n)

    for i in range(len(sig)):
        if i < n:
            # mirror the start of signal
            extremes_zeros[i] = sig[n - i]
            # mirror the end of signal
            extremes_zeros[len(sig) + n + i] = sig[len(sig) - i - 2]
        # fill the remaining with the signal itself
        extremes_zeros[i + n] = sig[i]

    # Build the array
    smoothen_signal = np.zeros(len(sig))

    # Fill the array
    for i in range(n, len(extremes_zeros) - n):
        # Calculate the mean of the neighours - we have to look at the surrounding neighbours,
        # thus we divide the total by 2 look back and further
        mean_neighbours = np.mean(extremes_zeros[i - n // 2:i + n // 2])
        smoothen_signal[i - n] = mean_neighbours

    return smoothen_signal


def data_process(data):
    """Signal Processing for Synthesis"""
    "Normalise Data"
    mean_value = np.mean(data)  # Calculate the mean of the signal
    abs_signal = np.abs(data)  # Calculate the absolute of the signal
    max_abs_signal = np.max(abs_signal)  # Get the maximum value of the absolute signal
    normalised_signal = (data - mean_value) / max_abs_signal

    "Denoise Signals"
    smoothen_signal = smooth(normalised_signal, 10)

    "Remove Baseline Wander"
    baseline_wander = smooth(smoothen_signal, 40)
    filtered_signal = smoothen_signal - baseline_wander

    "Remove minimum"
    interval_signal = filtered_signal - min(filtered_signal)

    return interval_signal


def quantization(signal, d):
    # Defining a given range of possible values that the signal can have.
    # d is the number of possible values.
    scaled_signal = signal * d
    rounded_signal = np.around(scaled_signal)

    quantised_signal = np.array(rounded_signal, dtype=int)

    return quantised_signal


def freq_fft(data, fps, limite_inferior, limite_superior):
    fft = np.fft.fft(data)

    frequencias = np.fft.fftfreq(len(data), d=(1/(fps*6)))  # 1/30 = 1/(5*6)

    condition = (frequencias > limite_inferior) & (frequencias <= limite_superior)

    indices = np.where(condition)
    freq_desejadas = frequencias[indices]
    fft_result = fft[indices]

    fft_result_positivo = np.abs(fft_result)

    return freq_desejadas, fft_result_positivo


def save_signals_to_csv(filename, time, signals):
    data = {'Time': time}
    for i, signal in enumerate(signals):
        data[f'Signal_{i+1}'] = signal

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
