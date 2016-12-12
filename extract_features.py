import numpy as np
from scipy import stats

def _compute_mean(arr):
    return np.mean(arr, axis=0)

def _compute_var(arr):
    return np.var(arr, axis=0)

def _compute_fft(arr):
    return np.real(np.fft.fft(arr, axis=0)[0:2]).flatten()

def extract_features(win):

    # Store total features

    x = np.array([])

    # Append each sub-feature to the total vector

    mag = np.array(win.data['magnetometer'])[:, 1:4]
    magResolved = np.array(map(lambda x: ((x[0] ** 2) + (x[1] ** 2) + (x[2] ** 2)) ** 0.5, mag))
    bar = np.array(win.data['barometer'])[:, 1:2]
    light = np.array(win.data['light'])[:, 1:2]

    x = np.append(x, _compute_mean(mag))
    x = np.append(x, _compute_mean(magResolved))
    x = np.append(x, _compute_mean(bar))
    x = np.append(x, _compute_mean(light))
    x = np.append(x, _compute_var(mag))
    x = np.append(x, _compute_var(magResolved))
    x = np.append(x, _compute_var(bar))
    x = np.append(x, _compute_var(light))
    x = np.append(x, _compute_fft(mag))
    x = np.append(x, _compute_fft(magResolved))
    x = np.append(x, _compute_fft(bar))
    x = np.append(x, _compute_fft(light))

    return x

def extract_labels(win):
    mag = np.array(win.data['magnetometer'])[:, 4]
    bar = np.array(win.data['barometer'])[:, 2]
    light = np.array(win.data['light'])[:, 2]
    labels = mag
    labels = np.append(labels, bar)
    return stats.mode(np.append(labels, light)).mode[0]
