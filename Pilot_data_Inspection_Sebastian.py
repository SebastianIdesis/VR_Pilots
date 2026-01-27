# Data inspection Sebastian
import pyxdf
import numpy as np

# Load the XDF file
#file_path = "Javad_Subject1_DryRun.xdf"
#file_path = "Computer_Sebastian_sub-P001_Roller_Coaster.xdf"
#file_path = "sub-P001_ses-S001_task-Default_run-001_eeg.xdf"
file_path = "sub-P055_ses-S001_task-Default_run-001_eeg.xdf"

streams, header = pyxdf.load_xdf(file_path)

# Print the labels and shape of each stream
print(f"Number of streams: {len(streams)}")
print("\nStream labels and shapes:")
for i, stream in enumerate(streams):
    stream_info = stream['info']
    label = stream_info['name'][0] if 'name' in stream_info else 'Unnamed'
    stream_type = stream_info['type'][0] if 'type' in stream_info else 'Unknown'
    
    # Get the shape of time_series and time_stamps
    if 'time_series' in stream:
        ts = stream['time_series']
        if hasattr(ts, 'shape'):
            time_series_shape = ts.shape
        else:
            time_series_shape = f"list of length {len(ts)}"
    else:
        time_series_shape = 'N/A'
    
    if 'time_stamps' in stream:
        stamps = stream['time_stamps']
        if hasattr(stamps, 'shape'):
            time_stamps_shape = stamps.shape
        else:
            time_stamps_shape = f"list of length {len(stamps)}"
    else:
        time_stamps_shape = 'N/A'
    
    print(f"\nStream {i}: {label} (Type: {stream_type})")
    print(f"  time_series shape: {time_series_shape}")
    print(f"  time_stamps shape: {time_stamps_shape}")
    print(f"  Stream keys: {list(stream.keys())}")

# Detailed exploration
print("\n" + "="*80)
print("DETAILED EXPLORATION")
print("="*80)

# Find streams by their names/types instead of assuming indices
eeg_stream = None
markers_stream = None
fms_stream = None
fms_streams = []  # Collect all FMS streams

for i, stream in enumerate(streams):
    stream_info = stream['info']
    label = stream_info['name'][0] if 'name' in stream_info else 'Unnamed'
    stream_type = stream_info['type'][0] if 'type' in stream_info else 'Unknown'
    
    # Identify EEG stream
    if stream_type == 'EEG' or 'actiCHamp' in label:
        eeg_stream = stream
        print(f"Found EEG stream at index {i}: {label}")
    
    # Identify markers stream
    if 'Game_Markers' in label or (stream_type == 'Markers' and 'Game' in label):
        markers_stream = stream
        print(f"Found Game_Markers stream at index {i}: {label}")
    
    # Identify FMS stream - collect all of them
    if 'FMS_Score' in label or (stream_type == 'Survey' and 'FMS' in label):
        fms_streams.append((i, label, stream))
        data_count = len(stream['time_series']) if 'time_series' in stream else 0
        print(f"Found FMS_Score stream at index {i}: {label} - {data_count} data points")

# Select the FMS stream with actual data
if len(fms_streams) > 0:
    print(f"\nFound {len(fms_streams)} FMS_Score stream(s)")
    for idx, label, stream in fms_streams:
        data_count = len(stream['time_series']) if 'time_series' in stream else 0
        if data_count > 0 and fms_stream is None:
            fms_stream = stream
            print(f"Using FMS stream at index {idx} with {data_count} data points")
    
    if fms_stream is None and len(fms_streams) > 0:
        # If all are empty, just use the first one
        fms_stream = fms_streams[0][2]
        print(f"All FMS streams are empty, using first one at index {fms_streams[0][0]}")

if eeg_stream is None:
    print("ERROR: Could not find EEG stream!")
if markers_stream is None:
    print("ERROR: Could not find Game_Markers stream!")
if fms_stream is None:
    print("ERROR: Could not find FMS_Score stream!")

# a) EEG Channels
print("\na) EEG Channels:")
if eeg_stream and 'info' in eeg_stream and 'desc' in eeg_stream['info']:
    desc = eeg_stream['info']['desc'][0]
    if 'channels' in desc:
        channels = desc['channels'][0]['channel']
        print(f"Number of channels: {len(channels)}")
        print("Channel labels:")
        for idx, ch in enumerate(channels):
            label = ch['label'][0] if 'label' in ch else f'Channel {idx}'
            print(f"  {idx}: {label}")
else:
    print("Channel information not found in stream info")

# b) Game_Markers
print("\nb) Game_Markers:")
if markers_stream and 'time_series' in markers_stream and len(markers_stream['time_series']) > 0:
    markers = markers_stream['time_series']
    # Flatten the markers if they are lists
    if isinstance(markers[0], list):
        markers_flat = [m[0] if m else '' for m in markers]
    else:
        markers_flat = markers
    
    unique_markers = sorted(set(markers_flat))
    print(f"Total number of markers: {len(markers_flat)}")
    print(f"Number of unique markers: {len(unique_markers)}")
    print("Unique markers:")
    for marker in unique_markers:
        count = markers_flat.count(marker)
        print(f"  '{marker}': {count} occurrences")
else:
    print("No marker data found")

# c) FMS_Score
print("\nc) FMS_Score:")
if fms_stream and 'time_series' in fms_stream and len(fms_stream['time_series']) > 0:
    import numpy as np
    scores = np.array(fms_stream['time_series'])
    unique_values = np.unique(scores)
    print(f"Number of data points: {len(scores)}")
    print(f"Unique values: {unique_values}")
    print(f"Range: [{scores.min()}, {scores.max()}]")
    print(f"Mean: {scores.mean():.2f}")
    print(f"Std: {scores.std():.2f}")
else:
    print("No survey data found")

# Check timestamps for alignment
print("\n" + "="*80)
print("TIMESTAMP INFORMATION FOR ALIGNMENT")
print("="*80)

print("\nEEG timestamps:")
if eeg_stream and 'time_stamps' in eeg_stream and len(eeg_stream['time_stamps']) > 0:
    eeg_times = eeg_stream['time_stamps']
    print(f"  Number of timestamps: {len(eeg_times)}")
    print(f"  First timestamp: {eeg_times[0]:.3f}")
    print(f"  Last timestamp: {eeg_times[-1]:.3f}")
    print(f"  Duration: {eeg_times[-1] - eeg_times[0]:.3f} seconds")
    
    # Calculate sampling rate
    eeg_intervals = np.diff(eeg_times)
    eeg_sampling_rate = 1.0 / np.mean(eeg_intervals)
    print(f"  Sampling rate: {eeg_sampling_rate:.2f} Hz")
    print(f"  Mean interval: {np.mean(eeg_intervals)*1000:.2f} ms")
else:
    print("  No timestamps found")

print("\nFMS_Score timestamps:")
if fms_stream and 'time_stamps' in fms_stream and len(fms_stream['time_stamps']) > 0:
    fms_times = fms_stream['time_stamps']
    print(f"  Number of timestamps: {len(fms_times)}")
    print(f"  First timestamp: {fms_times[0]:.3f}")
    print(f"  Last timestamp: {fms_times[-1]:.3f}")
    print(f"  Duration: {fms_times[-1] - fms_times[0]:.3f} seconds")
    
    # Calculate sampling rate (if more than 1 sample)
    if len(fms_times) > 1:
        fms_intervals = np.diff(fms_times)
        fms_sampling_rate = 1.0 / np.mean(fms_intervals)
        print(f"  Mean sampling rate: {fms_sampling_rate:.2f} Hz")
        print(f"  Mean interval: {np.mean(fms_intervals):.3f} seconds")
        print(f"  Interval range: [{fms_intervals.min():.3f}, {fms_intervals.max():.3f}] seconds")
        print(f"  All intervals: {fms_intervals}")
    else:
        print("  Only one sample - cannot calculate sampling rate")
    print(f"  Sample timestamps (first 10): {fms_times[:10]}")
else:
    print("  No timestamps found")

print("\nTime alignment:")
if eeg_stream and fms_stream and len(eeg_stream['time_stamps']) > 0 and len(fms_stream['time_stamps']) > 0:
    eeg_start = eeg_stream['time_stamps'][0]
    eeg_end = eeg_stream['time_stamps'][-1]
    fms_start = fms_stream['time_stamps'][0]
    fms_end = fms_stream['time_stamps'][-1]
    
    print(f"  EEG range: [{eeg_start:.3f}, {eeg_end:.3f}]")
    print(f"  FMS range: [{fms_start:.3f}, {fms_end:.3f}]")
    print(f"  Overlap: [{max(eeg_start, fms_start):.3f}, {min(eeg_end, fms_end):.3f}]")

# Create aligned figure
print("\n" + "="*80)
print("CREATING ALIGNED FIGURE")
print("="*80)

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

if not eeg_stream or not markers_stream or not fms_stream:
    print("ERROR: Cannot create figure - missing required streams")
else:
    # Get channel indices for AUX_1 and AUX_2
    channels = eeg_stream['info']['desc'][0]['channels'][0]['channel']
    channel_labels = [ch['label'][0] for ch in channels]
    aux1_idx = channel_labels.index('AUX_1')
    aux2_idx = channel_labels.index('AUX_2')

    # Extract data
    eeg_data = np.array(eeg_stream['time_series'])
    eeg_times = eeg_stream['time_stamps']
    aux1_data = eeg_data[:, aux1_idx]
    aux2_data = eeg_data[:, aux2_idx]

    # Calculate heart rate from PPG (AUX_2)
    print("\nExtracting heart rate from PPG signal...")
    # Find peaks in PPG signal - adjust parameters as needed
    # Using a minimum distance based on expected HR (assuming 40-180 bpm)
    sampling_rate = 1 / np.mean(np.diff(eeg_times))  # Calculate sampling rate
    min_peak_distance = int(sampling_rate * 0.4)  # Minimum 0.4s between peaks (150 bpm max)

    peaks, _ = find_peaks(aux2_data, distance=min_peak_distance, prominence=np.std(aux2_data)*0.5)
    peak_times = eeg_times[peaks]

    # Calculate instantaneous heart rate
    if len(peaks) > 1:
        ibi = np.diff(peak_times)  # Inter-beat intervals in seconds
        hr = 60.0 / ibi  # Convert to beats per minute
        hr_times = peak_times[1:]  # Time points for HR (one less than peaks)
        print(f"Detected {len(peaks)} peaks")
        print(f"Heart rate range: {hr.min():.1f} - {hr.max():.1f} bpm")
    else:
        hr = np.array([])
        hr_times = np.array([])
        print("Not enough peaks detected for heart rate calculation")

    fms_data = np.array(fms_stream['time_series']).flatten()
    fms_times = fms_stream['time_stamps']

    # Extract Game Markers
    markers_data = markers_stream['time_series']
    markers_times = markers_stream['time_stamps']
    # Flatten markers if they are lists
    if len(markers_data) > 0 and isinstance(markers_data[0], list):
        markers_labels = [m[0] if m else '' for m in markers_data]
    else:
        markers_labels = markers_data

    # Make time relative - start from 0 at the beginning of EEG recording
    time_offset = eeg_times[0]
    eeg_times_rel = eeg_times - time_offset
    fms_times_rel = fms_times - time_offset
    markers_times_rel = markers_times - time_offset

    # Create figure with 3 panels
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Panel 1: AUX_1
    axes[0].plot(eeg_times_rel, aux1_data, 'b-', linewidth=0.5)
    axes[0].set_ylabel('AUX_1 (EDA)', fontsize=12)
    axes[0].set_title('Aligned EEG Auxiliary Channels and FMS Score', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Heart Rate from PPG
    hr_times_rel = hr_times - time_offset
    axes[1].plot(hr_times_rel, hr, 'g-', linewidth=1.5, marker='o', markersize=3)
    axes[1].set_ylabel('Heart Rate (bpm)', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([max(0, hr.min()-10), hr.max()+10] if len(hr) > 0 else [40, 120])

    # Panel 3: FMS_Score
    axes[2].plot(fms_times_rel, fms_data, 'ro-', markersize=8, linewidth=2, label='FMS Score')
    axes[2].set_ylabel('FMS Score', fontsize=12)
    axes[2].set_xlabel('Time (seconds)', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    # Add vertical lines at FMS timestamps on all panels for reference
    for fms_time in fms_times_rel:
        for ax in axes:
            ax.axvline(fms_time, color='red', alpha=0.2, linestyle='--', linewidth=0.5)

    # Add vertical lines for Game Markers with different colors by type
    print(f"\nAdding {len(markers_times_rel)} game markers to the figure...")
    
    # Extract event type (before the | separator) for consistent coloring
    def get_event_type(label):
        """Extract the event type from marker label (part before |)"""
        if isinstance(label, str) and '|' in label:
            return label.split('|')[0]
        return label
    
    # Define color mapping for marker types
    marker_colors = {}
    unique_marker_types = sorted(set(get_event_type(label) for label in markers_labels))
    color_palette = ['orange', 'purple', 'green', 'red', 'blue', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for marker_type, color in zip(unique_marker_types, color_palette):
        marker_colors[marker_type] = color
    
    print(f"Marker type color mapping:")
    for marker_type, color in marker_colors.items():
        print(f"  '{marker_type}': {color}")
    
    for i, (marker_time, marker_label) in enumerate(zip(markers_times_rel, markers_labels)):
        event_type = get_event_type(marker_label)
        color = marker_colors.get(event_type, 'orange')
        for ax in axes:
            ax.axvline(marker_time, color=color, alpha=0.6, linestyle='-', linewidth=1.5)
        # Add label to top panel only to avoid clutter
        if i < 20:  # Limit labels to avoid overcrowding
            axes[0].text(marker_time, axes[0].get_ylim()[1], marker_label, 
                        rotation=90, va='top', ha='right', fontsize=8, alpha=0.7)

    plt.tight_layout()
    plt.savefig('aligned_signals.png', dpi=150, bbox_inches='tight')
    print("\nFigure saved as 'aligned_signals.png'")
    plt.show()