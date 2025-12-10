import numpy as np
import scipy.io as spio
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization

### ------ 1. D1 Training Data Preparation ------ ###
# Load D1
mat = spio.loadmat('CI\CW C\Coursework C Datasets\D1.mat', squeeze_me=True)
d = mat['d']
Index = mat['Index']
Class = mat['Class']
window_size = 250

# Preprocess: Butterworth Bandpass filter
fs = 25000
b, a = butter(3, [300/(fs/2), 3000/(fs/2)], btype='band')
d_filt = filtfilt(b, a, d)

# Extract spike segments and labels
segments, labels = [], []
for idx, cls in zip(Index, Class):
    if idx-window_size >= 0 and idx+window_size < len(d_filt):
        segments.append(d_filt[idx-window_size:idx+window_size])
        labels.append(cls)
segments = np.array(segments)
labels = np.array(labels) - 1  # For zero-based classes

# Standardise
scaler = StandardScaler()
segments_std = scaler.fit_transform(segments)
segments_std = segments_std[..., np.newaxis]  # [samples, timesteps, channels]

### ------ 2. Train CNN ------ ###
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

### ------ 3. Predict on New Data (D2–D6) ------ ###
for ds in ['D2','D3','D4','D5','D6']:
    mat = spio.loadmat(f'CI\CW C\Coursework C Datasets\{ds}.mat', squeeze_me=True)
    d = mat['d']
    d_filt = filtfilt(b, a, d)
    # Detect spikes – can adjust threshold for noisier sets
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
    pred_classes = np.argmax(pred_probs, axis=1) + 1 # Back to 1-based labels
    # Save
    spio.savemat(f'SupervisedCNN_{ds}.mat', {'Index':np.array(valid_indices), 'Class':pred_classes})
