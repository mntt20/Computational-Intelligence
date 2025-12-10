# Supervised Spike Sorting with Wavelet Denoising and CNN
import numpy as np
import scipy.io as spio
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.preprocessing import StandardScaler
import pywt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization

def wavelet_denoise(signal, wavelet='db4', level=3, threshold_mode='soft', threshold_scale=1.0):
    # Decompose signal
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    # Estimate noise level from highest-frequency detail
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = threshold_scale * sigma * np.sqrt(2 * np.log(len(signal)))
    # Threshold details
    denoised_coeffs = [coeffs[0]] + [
        pywt.threshold(c, value=uthresh, mode=threshold_mode)
        for c in coeffs[1:]
    ]
    # Reconstruct
    return pywt.waverec(denoised_coeffs, wavelet)

# --- Step 1: Load and Denoise D1 ---
mat = spio.loadmat('CI\CW C\Coursework C Datasets\D1.mat', squeeze_me=True)
d = mat['d']
Index = mat['Index']
Class = mat['Class']
window_size = 250

# Wavelet denoising
d_denoised = wavelet_denoise(d, wavelet='db4', level=3, threshold_mode='soft', threshold_scale=1.0)

# Bandpass filter (if desired, but some researchers skip linear filter after wavelet denoise[3])
fs = 25000
b, a = butter(3, [300/(fs/2), 3000/(fs/2)], btype='band')
d_filt = filtfilt(b, a, d_denoised)

# --- Spike extraction and supervised training as before ---
segments = []
labels = []
for idx, cls in zip(Index, Class):
    if idx-window_size >= 0 and idx+window_size < len(d_filt):
        segments.append(d_filt[idx-window_size:idx+window_size])
        labels.append(cls)
segments = np.array(segments)
labels = np.array(labels) - 1

scaler = StandardScaler()
segments_std = scaler.fit_transform(segments)
segments_std = segments_std[..., np.newaxis]

model = Sequential([
    Conv1D(32, 5, activation='relu', input_shape=(window_size*2,1)),
    BatchNormalization(),
    MaxPooling1D(2),
    Conv1D(64, 5, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(5, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(segments_std, labels, epochs=30, batch_size=64, verbose=2)

# --- Step 2: Apply wavelet denoising to D2-D6 and sort spikes ---
for ds in ['D2','D3','D4','D5','D6']:
    mat = spio.loadmat(f'CI\CW C\Coursework C Datasets\{ds}.mat', squeeze_me=True)
    d_raw = mat['d']
    d_denoised = wavelet_denoise(d_raw, wavelet='db4', level=3, threshold_mode='soft', threshold_scale=1.0)
    d_filt = filtfilt(b, a, d_denoised)
    thr = 4*np.std(d_filt)
    indices, _ = find_peaks(np.abs(d_filt), height=thr, distance=100)
    test_segments, valid_indices = [], []
    for idx in indices:
        if idx-window_size >= 0 and idx+window_size < len(d_filt):
            test_segments.append(d_filt[idx-window_size:idx+window_size])
            valid_indices.append(idx)
    if len(test_segments) == 0: continue
    test_segments = np.array(test_segments)
    test_segments_std = scaler.transform(test_segments)[..., np.newaxis]
    pred_probs = model.predict(test_segments_std)
    pred_classes = np.argmax(pred_probs, axis=1) + 1
    spio.savemat(f'WaveletCNN_{ds}.mat', {'Index':np.array(valid_indices), 'Class':pred_classes})
