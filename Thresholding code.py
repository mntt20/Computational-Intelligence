import numpy as np
import scipy.io as spio
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization


### ------ 0. Helper functions ------ ###

def robust_threshold(x, k=4.5):
    """
    Robust amplitude threshold based on median absolute deviation (MAD).
    """
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad > 0:
        sigma = 1.4826 * mad   # Gaussian-equivalent sigma
    else:
        sigma = np.std(x)
    return k * sigma

def per_segment_zscore(segments):
    """
    Z-score normalisation per segment (window).
    segments: array [n_segments, n_samples]
    """
    seg_mean = segments.mean(axis=1, keepdims=True)
    seg_std  = segments.std(axis=1, keepdims=True) + 1e-8
    return (segments - seg_mean) / seg_std


### ------ 1. D1 Training Data Preparation ------ ###

# Load D1
mat = spio.loadmat(r'CI\CW C\Coursework C Datasets\D1.mat', squeeze_me=True)
d = mat['d']
Index = mat['Index']
Class = mat['Class']
window_size = 250

# Preprocess: Butterworth Bandpass filter
fs = 25000
b, a = butter(3, [300/(fs/2), 3000/(fs/2)], btype='band')
d_filt = filtfilt(b, a, d)

# Extract spike segments and labels (as given by D1)
segments, labels = [], []
for idx, cls in zip(Index, Class):
    if idx - window_size >= 0 and idx + window_size < len(d_filt):
        segments.append(d_filt[idx-window_size:idx+window_size])
        labels.append(cls)

segments = np.array(segments)          # [n_segments, 2*window_size]
labels = np.array(labels) - 1          # zero-based labels 0..4

# Step 3: per-segment normalisation (amplitude invariance)
segments_norm = per_segment_zscore(segments)

# Global standardisation (kept, but now on per-segment-normalised data)
scaler = StandardScaler()
segments_std = scaler.fit_transform(segments_norm)
segments_std = segments_std[..., np.newaxis]   # [samples, timesteps, channels]


### ------ 2. Train CNN with class weights (Step 4) ------ ###

# Compute class weights to handle imbalance
classes_unique = np.unique(labels)
class_weights_arr = compute_class_weight(
    class_weight='balanced',
    classes=classes_unique,
    y=labels
)
class_weights = dict(zip(classes_unique, class_weights_arr))

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

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    segments_std,
    labels,
    epochs=30,
    batch_size=64,
    verbose=2,
    class_weight=class_weights   # Step 4
)


### ------ 3. Predict on New Data (D2–D6) with improved detection (Steps 1 & 2) ------ ###

peak_width = 40    # samples around rough peak to refine centre

for ds in ['D2', 'D3', 'D4', 'D5', 'D6']:
    mat = spio.loadmat(fr'CI\CW C\Coursework C Datasets\{ds}.mat', squeeze_me=True)
    d = mat['d']

    # Band-pass filter using same filter as D1
    d_filt = filtfilt(b, a, d)

    # Step 1: robust, per-dataset threshold based on MAD
    thr = robust_threshold(d_filt, k=4.5)

    # Detect candidate peaks over full signal (batch, not streaming)
    # Reduce distance a bit so close spikes are not discarded; must still be > spike width.
    indices, properties = find_peaks(
        np.abs(d_filt),
        height=thr,
        distance=40
    )

    test_segments = []
    valid_indices = []

    # Step 2: local recentering of each detected peak
    for p in indices:
        # local window to refine true maximum
        left = max(p - peak_width // 2, 0)
        right = min(p + peak_width // 2, len(d_filt))
        local = d_filt[left:right]

        if local.size == 0:
            continue

        # Refine to the true extremum in this small window
        local_idx = np.argmax(np.abs(local))
        idx = left + local_idx

        # Now extract fixed window around refined centre
        if idx - window_size >= 0 and idx + window_size < len(d_filt):
            segment = d_filt[idx-window_size:idx+window_size]
            test_segments.append(segment)
            valid_indices.append(idx)

    if len(test_segments) == 0:
        # Nothing detected for this dataset
        continue

    test_segments = np.array(test_segments)

    # Step 3 again: per-segment normalisation at test time
    test_segments_norm = per_segment_zscore(test_segments)

    # Apply same global scaler learned on training data
    test_segments_std = scaler.transform(test_segments_norm)
    test_segments_std = test_segments_std[..., np.newaxis]

    # Predict classes
    pred_probs = model.predict(test_segments_std, verbose=0)
    pred_classes = np.argmax(pred_probs, axis=1) + 1   # back to 1-based labels

    # Save predictions
    spio.savemat(
        fr'{ds}.mat',
        {'Index': np.array(valid_indices), 'Class': pred_classes}
    )
