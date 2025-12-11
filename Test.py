import numpy as np
import scipy.io as spio
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
import pywt

### ------ Helper functions ------ ###

def wavelet_denoise(signal, wavelet='db4', level=3,
                    threshold_mode='soft', threshold_scale=1.0):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = threshold_scale * sigma * np.sqrt(2 * np.log(len(signal)))
    denoised_coeffs = [coeffs[0]] + [
        pywt.threshold(c, value=uthresh, mode=threshold_mode)
        for c in coeffs[1:]
    ]
    return pywt.waverec(denoised_coeffs, wavelet)

def per_segment_zscore(segments):
    seg_mean = segments.mean(axis=1, keepdims=True)
    seg_std  = segments.std(axis=1, keepdims=True) + 1e-8
    return (segments - seg_mean) / seg_std

def build_features(segments):
    seg_norm = per_segment_zscore(segments)
    x1 = seg_norm
    energy = per_segment_zscore(seg_norm**2)
    X = np.stack([x1, energy], axis=-1)   # [N, T, 2]
    return X

### ------ 1. D1: same detector as D2–D6 ------ ###

mat = spio.loadmat(r'D1.mat', squeeze_me=True)
d = mat['d']
Index_gt = mat['Index']        # ground-truth indices (1-based or 0-based?)
Class_gt = mat['Class']
window_size = 250
fs = 25000

# preprocess D1
d_denoised = wavelet_denoise(d)
b, a = butter(3, [300/(fs/2), 3000/(fs/2)], btype='band')
d_filt = filtfilt(b, a, d_denoised)

# run the SAME detector as for D2–D6
thr = 2.5 * np.std(d_filt)       # slightly relaxed
indices_det, _ = find_peaks(np.abs(d_filt), height=thr, distance=40)

# match detections to nearest ground-truth spike within tolerance
tolerance = 10                 # samples
segments = []
labels = []

Index_gt = np.asarray(Index_gt).astype(int).ravel()
Class_gt = np.asarray(Class_gt).astype(int).ravel()

for idx_det in indices_det:
    # find closest ground-truth spike
    diffs = np.abs(Index_gt - idx_det)
    j = np.argmin(diffs)
    if diffs[j] <= tolerance:
        cls = Class_gt[j]
        if idx_det-window_size >= 0 and idx_det+window_size < len(d_filt):
            seg = d_filt[idx_det-window_size:idx_det+window_size]
            segments.append(seg)
            labels.append(cls)

segments = np.array(segments)
labels = np.array(labels) - 1      # 0..4
print("D1: detected-for-training segments:", segments.shape[0])

# build 2-channel features
X_train = build_features(segments)
y_train = labels

### ------ 2. Train CNN with class weights ------ ###

classes_unique = np.unique(y_train)
class_weights_arr = compute_class_weight(
    class_weight='balanced',
    classes=classes_unique,
    y=y_train
)
class_weights = dict(zip(classes_unique, class_weights_arr))
print("D1 class counts (detected):", dict(zip(*np.unique(y_train, return_counts=True))))
print("Class weights:", class_weights)

model = Sequential([
    Conv1D(32, 5, activation='relu', input_shape=(window_size*2, 2)),
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

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train,
          epochs=30,
          batch_size=64,
          verbose=2,
          class_weight=class_weights)

### ------ 3. D2–D6: same detector + classifier ------ ###

for ds in ['D2','D3','D4','D5','D6']:
    mat = spio.loadmat(fr'{ds}.mat', squeeze_me=True)
    d_raw = mat['d']

    d_denoised = wavelet_denoise(d_raw)
    d_filt = filtfilt(b, a, d_denoised)

    # dataset-specific settings
    if ds in ['D2', 'D3', 'D4']:
        thr = 2.0 * np.std(d_filt)   # lower threshold
        min_dist = 30                # allow closer spikes
    else:
        thr = 2.5 * np.std(d_filt)
        min_dist = 40

    indices, _ = find_peaks(
        np.abs(d_filt),
        height=thr,
        distance=min_dist
    )

    test_segments, valid_indices = [], []
    for idx in indices:
        if idx-window_size >= 0 and idx+window_size < len(d_filt):
            test_segments.append(d_filt[idx-window_size:idx+window_size])
            valid_indices.append(idx)

    print(f"{ds}: detected {len(valid_indices)} spikes")

    if len(test_segments) == 0:
        continue

    test_segments = np.array(test_segments)
    X_test = build_features(test_segments)

    pred_probs = model.predict(X_test, verbose=0)
    pred_classes = np.argmax(pred_probs, axis=1) + 1

    # diagnostics
    unique, counts = np.unique(pred_classes, return_counts=True)
    print(f"{ds}: class distribution {dict(zip(unique, counts))}")

    spio.savemat(f'Output_{ds}.mat',
                 {'Index': np.array(valid_indices), 'Class': pred_classes})
