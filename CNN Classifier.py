## Code a CNN to classify spikes from different neurons
import scipy.io as spio
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from scipy.signal import butter, filtfilt

# --- Load D1.mat ---
try:
    mat = spio.loadmat('CI\\CW C\\Coursework C Datasets\\D1.mat', squeeze_me=True)
except FileNotFoundError:
    print("The specified .mat file was not found.")
    exit()

d = mat['d']
Index = mat['Index']
Class = mat['Class']

# --- Signal Processing: Bandpass Filter ---
def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

sampling_rate = 25000  # 25 kHz
lowcut = 300  # Hz
highcut = 5000  # Hz
d_filtered = bandpass_filter(d, lowcut, highcut, sampling_rate)

# --- Extract Spike Waveforms ---
window_size = 250  # samples before and after spike peak
spike_waveforms = []
spike_labels = []

for idx, (spike_time, spike_class) in enumerate(zip(Index, Class)):
    start = spike_time - window_size
    end = spike_time + window_size
    if start >= 0 and end < len(d_filtered):
        waveform = d_filtered[start:end]
        spike_waveforms.append(waveform)
        spike_labels.append(spike_class - 1)  # Convert to 0-based indexing

spike_waveforms = np.array(spike_waveforms)
spike_labels = np.array(spike_labels)

# --- Normalize Waveforms ---
scaler = StandardScaler()
spike_waveforms = scaler.fit_transform(spike_waveforms)

# --- Reshape for CNN (samples, timesteps, channels) ---
spike_waveforms = spike_waveforms.reshape(spike_waveforms.shape[0], spike_waveforms.shape[1], 1)

# --- One-hot encode labels ---
num_classes = 5
spike_labels = to_categorical(spike_labels, num_classes)

# --- Split into train/test (90%/10%) ---
X_train, X_test, y_train, y_test = train_test_split(
    spike_waveforms, spike_labels, test_size=0.1, stratify=spike_labels, random_state=42
)

# --- Build 1D CNN Model ---
model = Sequential([
    Conv1D(32, kernel_size=5, activation='relu', input_shape=(500, 1)),
    MaxPooling1D(pool_size=2),
    Conv1D(64, kernel_size=5, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --- Train the Model ---
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# --- Evaluate on Test Set ---
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.4f}")

# --- Plot Training History ---
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()
