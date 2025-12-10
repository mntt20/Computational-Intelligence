import scipy.io as spio
import numpy as np 
import matplotlib.pyplot as plt

# Load the .mat file
try:
     mat = spio.loadmat('CI\CW C\Coursework C Datasets\D1.mat', squeeze_me=True)
except FileNotFoundError:
        print("The specified .mat file was not found.")
        exit()

d = mat['d']
Index = mat['Index']
Class = mat['Class']

#--- Plotting 3 Examples for each of the 5 classes ---

sampling_rate = 25000 # 25 kHz
window_size = 250 # Samples to plot before and after the spike peak

#Loop through each class ID (1 to 5)

for class_id in range(1, 6):
    # Find indices of spikes belonging to the current class
    # 'indicies_of_class' will hold the positions in Class array
    # (e.g., 0,5,12) where class matches class_id  
    indices_of_class = np.where(Class == class_id)[0]   

    # Check if there are at least 3 examples for the class
    if len(indices_of_class) < 3:
        print(f"Warning: Found fewer than 3 examples for class {class_id}. Plotting {len(indices_of_class)}")
        examples_to_plot = len(indices_of_class)
    else:
        # Just take the first 3 examples
        examples_to_plot = indices_of_class[:3]

    if len(examples_to_plot) == 0:
        print(f"No examples found for class {class_id}.")
        continue # Skip to the next class if no examples are found

    # 2. Create a new figure window for this class
    # (3 rows, 1 column)
    fig, axes = plt.subplots(nrows=3,ncols=1,figsize=(10, 8), sharex=True)
    fig.suptitle(f'First 3 Examples of Class {class_id} Spikes', fontsize=16)

    # 3. Loop through the 3 examples and plot each
    for i, spike_index in enumerate(examples_to_plot):

        # Get the spike time from Index vector
        spike_time = Index[spike_index]

        # Define the window around the spike
        start_sample = spike_time - window_size
        end_sample = spike_time + window_size

        # Basic boundary check to avoid plotting near the signal edge
        if start_sample < 0 or end_sample > len(d):
            axes[i].set_title(f'Spike at sample {spike_time} is too close to edge to plot.')
            continue

        # Get the segment of the signal data (y-axis)
        d_segment = d[start_sample:end_sample]  # Extract the spike segment

        # Create time vector for this segment
        num_samples_in_segment = len(d_segment)
        time_vector = np.linspace(-window_size/sampling_rate, window_size/sampling_rate, num = num_samples_in_segment)

        # 4. Plot on the correct subplot (ax)
        ax = axes[i]
        ax.plot(time_vector, d_segment, color='blue')
        ax.set_ylabel('Amplitude (mV)')
        ax.set_title(f'Spike Occurence #{i+1} at Sample {spike_time}')

    # Add a single x-label to the bottom plot
    axes[-1].set_xlabel('Time (seconds) relative to spike peak')

    #Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Show the plots
    # Code will pause here until the window is closed
    # Then proceeds to the next class and open a new window
    plt.show()




        