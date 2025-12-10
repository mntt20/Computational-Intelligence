import scipy.io as spio
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, MaxPooling1D, Flatten, Dense, Dropout
from sklearn.preprocessing import StandardScaler

# --- Step 1: Load D1 and train CNN (full data) ---
mat = spio.loadmat('CI/CW C/Coursework C Datasets/D1.mat', squeeze_me=True)
d1 = mat['d']
Index1 = mat['Index']
Class1 = mat['Class']
window_size = 250

# Filtering for D1 using Butterworth filter
fs = 25000
b, a = butter(N=3, Wn=[300/(fs/2), 3000/(fs/2)], btype='band')
d1_filt = filtfilt(b, a, d1)

# Extract and label all spike segments from D1
segments = []
labels = []
for idx, cls in zip(Index1, Class1):
    if idx-window_size >= 0 and idx+window_size < len(d1_filt):
        segment = d1_filt[idx-window_size:idx+window_size]
        segments.append(segment)
        labels.append(cls)
segments = np.array(segments)
labels = np.array(labels) - 1  # 0-based class labels for keras

# Standardise the input
scaler = StandardScaler()
segments_std = scaler.fit_transform(segments)

# Reshape for CNN: [samples, timesteps, channels]
X_train = segments_std[..., np.newaxis]
y_train = labels

# Define CNN
model = Sequential([
    Conv1D(32, 5, activation='relu', input_shape=(window_size*2, 1)),
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

# Train CNN using all of D1
model.fit(X_train, y_train, epochs=30, batch_size=64, verbose=2)

# --- Step 2: Detect and classify spikes in D2-D6 ---
for ds in ['D2','D3','D4','D5','D6']:
    mat = spio.loadmat(f'CI/CW C/Coursework C Datasets/{ds}.mat', squeeze_me=True)
    d = mat['d']
    # Filtering
    d_filt = filtfilt(b, a, d)
    # Spike detection
    thr = 4 * np.std(d_filt)
    indices, _ = find_peaks(np.abs(d_filt), height=thr, distance=100)

    # Extract spike segments
    segments = []
    valid_indices = []
    for idx in indices:
        if idx-window_size >= 0 and idx+window_size < len(d_filt):
            segment = d_filt[idx-window_size:idx+window_size]
            segments.append(segment)
            valid_indices.append(idx)
    if len(segments) == 0:
        continue
    segments = np.array(segments)
    segments_std = scaler.transform(segments)
    X_new = segments_std[..., np.newaxis]

    # Predict class labels using CNN
    pred_probs = model.predict(X_new)
    pred_classes = np.argmax(pred_probs, axis=1) + 1  # back to 1-based class labels

    # Save results
    output_dict = {'Index': np.array(valid_indices), 'Class': pred_classes}
    spio.savemat(f'CNN_output_{ds}.mat', output_dict)
    print(f"Saved CNN_output_{ds}.mat with {len(valid_indices)} spikes.")

