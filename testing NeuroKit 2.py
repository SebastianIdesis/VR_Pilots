import neurokit2 as nk

# ============================================================================
# PROCESSING FLAGS - Set to control which analyses to run
# ============================================================================
PROCESS_AUX_SENSORS = True  # Set to True to process EDA, PPG, Respiration
PROCESS_EEG = False           # Set to False to skip EEG analysis
# ============================================================================

# Download example data
#data = nk.data("bio_eventrelated_100hz")

# Preprocess the data (filter, find peaks, etc.)
#processed_data, info = nk.bio_process(ecg=data["ECG"], rsp=data["RSP"], eda=data["EDA"], sampling_rate=100)

# Compute relevant features
#results = nk.bio_analyze(processed_data, sampling_rate=100)

import pyxdf
import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path("Results")
RESULTS_DIR.mkdir(exist_ok=True)

#file_path = "sub-P055_ses-S001_task-Default_run-001_eeg.xdf"
#file_path = "sub-P99_ses-S001_task-Default_run-001_eeg.xdf"
file_path = "./Data/P99_Sebastian_From_Javad.xdf"

# Extract subject ID from file name (after "sub-P")
file_name = Path(file_path).name
subject_id = "unknown"
if "sub-P" in file_name:
    subject_id = file_name.split("sub-P", 1)[1].split("_", 1)[0]
subject_tag = f"sub-P{subject_id}"

streams, header = pyxdf.load_xdf(file_path)

for i, stream in enumerate(streams):
    stream_info = stream['info']
    label = stream_info['name'][0] if 'name' in stream_info else 'Unnamed'
    stream_type = stream_info['type'][0] if 'type' in stream_info else 'Unknown'
    
    print(f"Stream {i}: Type='{stream_type}', Name='{label}'")
    
    # Identify EEG stream
    if stream_type == 'EEG' or 'actiCHamp' in label:
        eeg_stream = stream
        print(f"  → Found EEG stream at index {i}")
    
    # Identify Marker stream (including actiCHamp markers)
    if (stream_type == 'Markers' or 
        'marker' in label.lower() or 
        'trigger' in label.lower() or
        'actichampmarker' in label.lower().replace(' ', '')):
        marker_stream = stream
        print(f"  → Found Marker stream at index {i}: '{label}'")

    # Identify Shimmer stream
    if 'shimmer' in label.lower() or 'shimmer' in stream_type.lower():
        shimmer_stream = stream
        print(f"  → Found Shimmer stream at index {i}: '{label}'")

# ============================================================================
# EXTRACT MARKERS
# ============================================================================
if 'marker_stream' in locals():
    print("\n" + "="*70)
    print("MARKER STREAM INFORMATION")
    print("="*70)
    
    # Get marker data
    marker_timestamps = marker_stream['time_stamps']
    marker_data = marker_stream['time_series']
    
    # Convert to numpy array if it's a list
    if isinstance(marker_data, list):
        marker_data = np.array(marker_data)
    
    # Markers can be stored as strings or numbers
    # Extract marker labels
    if marker_data.dtype == object or marker_data.dtype.kind in ['U', 'S']:
        # String markers
        marker_labels = [str(m[0]) if isinstance(m, (list, np.ndarray)) and len(m) > 0 else str(m) for m in marker_data]
    else:
        # Numeric markers
        marker_labels = [str(int(m[0])) if isinstance(m, (list, np.ndarray)) and len(m) > 0 else str(int(m)) for m in marker_data]
    
    print(f"\nTotal markers found: {len(marker_timestamps)}")
    print(f"\nFirst 20 markers:")
    print(f"{'Time (s)':<15} {'Marker Label':<30}")
    print("-" * 60)
    for i in range(min(20, len(marker_timestamps))):
        print(f"{marker_timestamps[i]:<15.3f} {marker_labels[i]:<30}")
    
    # Get unique marker types
    unique_markers = list(set(marker_labels))
    print(f"\nUnique marker types: {unique_markers}")
    print(f"\nMarker counts:")
    for marker_type in unique_markers:
        count = marker_labels.count(marker_type)
        print(f"  {marker_type}: {count}")
    
    # Store in a convenient structure
    markers = {
        'timestamps': marker_timestamps,
        'labels': marker_labels,
        'unique_types': unique_markers
    }
else:
    print("\n⚠️  No marker stream found in the XDF file")
    markers = None

# ============================================================================
# EXPLORE SHIMMER STREAM
# ============================================================================
if 'shimmer_stream' in locals():
    print("\n" + "="*70)
    print("SHIMMER STREAM INFORMATION")
    print("="*70)

    shimmer_info = shimmer_stream['info']
    shimmer_label = shimmer_info['name'][0] if 'name' in shimmer_info else 'Unnamed'
    shimmer_type = shimmer_info['type'][0] if 'type' in shimmer_info else 'Unknown'
    shimmer_srate = float(shimmer_info['nominal_srate'][0]) if 'nominal_srate' in shimmer_info else None

    shimmer_ts = shimmer_stream.get('time_series', None)
    shimmer_times = shimmer_stream.get('time_stamps', None)

    print(f"Name: {shimmer_label}")
    print(f"Type: {shimmer_type}")
    if shimmer_srate is not None:
        print(f"Sampling rate: {shimmer_srate} Hz")

    if shimmer_ts is not None:
        shimmer_ts = np.array(shimmer_ts)
        print(f"Data shape: {shimmer_ts.shape}")
    if shimmer_times is not None and len(shimmer_times) > 0:
        print(f"Time range: {shimmer_times[0]:.3f}s to {shimmer_times[-1]:.3f}s")

    # Try to extract channel labels and units (if available)
    channel_labels = []
    channel_units = []
    try:
        channel_info = shimmer_info['desc'][0]['channels'][0]['channel']
        for ch in channel_info:
            channel_labels.append(ch['label'][0] if 'label' in ch else 'Unknown')
            channel_units.append(ch['unit'][0] if 'unit' in ch else 'N/A')
    except Exception as e:
        print(f"⚠️  Could not read Shimmer channel metadata: {e}")

    if channel_labels:
        print(f"Channels ({len(channel_labels)}):")
        for label, unit in zip(channel_labels, channel_units):
            print(f"  - {label} ({unit})")
    else:
        print("No channel metadata found for Shimmer stream.")

    # Identify empty vs recorded channels
    if shimmer_ts is not None:
        def classify_channel(data, std_eps=1e-3, range_eps=1e-3):
            if data.size == 0:
                return "empty"
            if np.all(np.isnan(data)):
                return "empty"
            finite = data[np.isfinite(data)]
            if finite.size == 0:
                return "empty"
            if np.nanstd(finite) < std_eps or (np.nanmax(finite) - np.nanmin(finite)) < range_eps:
                return "empty"
            return "has data"

        n_channels = shimmer_ts.shape[1] if shimmer_ts.ndim > 1 else 1
        empty_channels = []
        data_channels = []

        for ch_idx in range(n_channels):
            ch_data = shimmer_ts[:, ch_idx] if shimmer_ts.ndim > 1 else shimmer_ts
            status = classify_channel(ch_data)
            ch_label = channel_labels[ch_idx] if ch_idx < len(channel_labels) else f"Channel_{ch_idx+1}"
            if status == "empty":
                empty_channels.append(ch_label)
            else:
                data_channels.append(ch_label)

        print("\nShimmer channel data check:")
        print(f"  - Channels with data: {len(data_channels)}")
        print(f"  - Empty/flat channels: {len(empty_channels)}")
        if data_channels:
            print("  Channels with data:")
            for ch in data_channels:
                print(f"    - {ch}")
        if empty_channels:
            print("  Empty/flat channels:")
            for ch in empty_channels:
                print(f"    - {ch}")

    # Quick plot for selected Shimmer channels
    selected_shimmer_channels = [
        "GSR_Skin_Resistance",
        "GSR_Skin_Conductance",
        "GSR_Range",
        "PPG_A13",
        "PPGtoHR",
        "PPG_IBI",
    ]

    if shimmer_ts is not None and shimmer_times is not None and len(shimmer_times) > 0:
        if channel_labels:
            available = {label: idx for idx, label in enumerate(channel_labels)}
            selected_indices = [available[ch] for ch in selected_shimmer_channels if ch in available]
            selected_labels = [ch for ch in selected_shimmer_channels if ch in available]

            if selected_indices:
                def remove_spike_jumps(values, percentile=99.5, scale=3.0):
                    values = values.astype(float).copy()
                    values[~np.isfinite(values)] = np.nan
                    diffs = np.abs(np.diff(values))
                    diffs = diffs[np.isfinite(diffs)]
                    if diffs.size == 0:
                        return values
                    threshold = np.nanpercentile(diffs, percentile) * scale
                    jump_idx = np.where(np.abs(np.diff(values)) > threshold)[0]
                    for idx in jump_idx:
                        values[idx] = np.nan
                        if idx + 1 < values.size:
                            values[idx + 1] = np.nan
                    return values

                fig, axes = plt.subplots(len(selected_indices), 1, figsize=(12, 2.5 * len(selected_indices)), sharex=True)
                if len(selected_indices) == 1:
                    axes = [axes]

                for ax, ch_idx, ch_label in zip(axes, selected_indices, selected_labels):
                    cleaned = remove_spike_jumps(shimmer_ts[:, ch_idx])
                    ax.plot(shimmer_times, cleaned, linewidth=0.7)
                    ax.set_title(f"Shimmer: {ch_label}", fontsize=10, fontweight='bold')
                    ax.set_ylabel("Amplitude", fontsize=9)
                    ax.grid(True, alpha=0.3)

                axes[-1].set_xlabel("Time (s)", fontsize=9)
                plt.tight_layout()
                shimmer_plot_path = RESULTS_DIR / f"{subject_tag}_shimmer_selected_channels.png"
                plt.savefig(shimmer_plot_path, dpi=300)
                plt.show()
                print(f"\nSaved Shimmer selected channels plot to: {shimmer_plot_path}")
            else:
                print("\n⚠️  None of the selected Shimmer channels were found in this stream")
        else:
            print("\n⚠️  Cannot plot selected Shimmer channels without channel labels")

    # Keep only agreed Shimmer channels and store as external EDA/PPG
    eda_data_EXT = {'channels': [], 'data': [], 'labels': []}
    ppg_data_EXT = {'channels': [], 'data': [], 'labels': []}

    if shimmer_ts is not None and shimmer_times is not None and len(shimmer_times) > 0 and channel_labels:
        shimmer_label_to_idx = {label: idx for idx, label in enumerate(channel_labels)}

        shimmer_eda_label = "GSR_Skin_Conductance"
        shimmer_ppg_label = "PPG_A13"

        if shimmer_eda_label in shimmer_label_to_idx:
            idx = shimmer_label_to_idx[shimmer_eda_label]
            eda_data_EXT['channels'].append(idx)
            eda_data_EXT['labels'].append(shimmer_eda_label)
            eda_data_EXT['data'].append(shimmer_ts[:, idx])

        if shimmer_ppg_label in shimmer_label_to_idx:
            idx = shimmer_label_to_idx[shimmer_ppg_label]
            ppg_data_EXT['channels'].append(idx)
            ppg_data_EXT['labels'].append(shimmer_ppg_label)
            ppg_data_EXT['data'].append(shimmer_ts[:, idx])

        if eda_data_EXT['data']:
            eda_data_EXT['data'] = np.array(eda_data_EXT['data']).T
            eda_data_EXT['time_stamps'] = shimmer_times
            eda_data_EXT['sampling_rate'] = shimmer_srate if shimmer_srate is not None else None
            print(f"\nShimmer EDA_EXT: {eda_data_EXT['labels']}")
            print(f"Shimmer EDA_EXT shape: {eda_data_EXT['data'].shape}")
        else:
            print("\n⚠️  Shimmer EDA_EXT channel not found")

        if ppg_data_EXT['data']:
            ppg_data_EXT['data'] = np.array(ppg_data_EXT['data']).T
            ppg_data_EXT['time_stamps'] = shimmer_times
            ppg_data_EXT['sampling_rate'] = shimmer_srate if shimmer_srate is not None else None
            print(f"\nShimmer PPG_EXT: {ppg_data_EXT['labels']}")
            print(f"Shimmer PPG_EXT shape: {ppg_data_EXT['data'].shape}")
        else:
            print("\n⚠️  Shimmer PPG_EXT channel not found")
else:
    print("\n⚠️  No Shimmer stream found in the XDF file")

# Extract channel information and separate signal types
if 'eeg_stream' in locals():
    # Get channel labels and metadata
    channel_info = eeg_stream['info']['desc'][0]['channels'][0]['channel']
    channel_labels = [ch['label'][0] for ch in channel_info]
    
    # Extract units for each channel (key for identifying signal types!)
    channel_units = []
    for ch in channel_info:
        if 'unit' in ch:
            channel_units.append(ch['unit'][0])
        else:
            channel_units.append('N/A')
    
    print(f"\nTotal channels found: {len(channel_labels)}")
    print(f"Channel labels: {channel_labels}")
    print(f"\n{'Channel':<15} {'Unit':<15} {'Type Hint'}")
    print("-" * 60)
    for label, unit in zip(channel_labels, channel_units):
        if 'AUX' in label.upper():
            # Identify signal type based on units
            type_hint = ''
            unit_upper = unit.upper()
            if 'SIEMENS' in unit_upper or 'µS' in unit or 'uS' in unit_upper or 'MICROSIEMENS' in unit_upper:
                type_hint = '→ likely EDA/GSR'
            elif 'BPM' in unit_upper or unit == 'unitless' or 'V' in unit_upper:
                type_hint = '→ likely PPG or RESP'
            print(f"{label:<15} {unit:<15} {type_hint}")
    
    # Get time series data
    time_series = eeg_stream['time_series']
    time_stamps = eeg_stream['time_stamps']
    
    # Get sampling rate
    sampling_rate = float(eeg_stream['info']['nominal_srate'][0])
    print(f"Sampling rate: {sampling_rate} Hz")
    
    # Initialize dictionaries to store separated signals
    eeg_data = {'channels': [], 'data': [], 'labels': []}
    eda_data = {'channels': [], 'data': [], 'labels': []}
    ppg_data = {'channels': [], 'data': [], 'labels': []}
    resp_data = {'channels': [], 'data': [], 'labels': []}
    
    # First, identify all AUX channels with their units
    aux_channels = []
    for idx, label in enumerate(channel_labels):
        if 'AUX' in label.upper():
            aux_channels.append((idx, label, channel_units[idx]))
    
    print(f"\nFound {len(aux_channels)} AUX channels:")
    
    # Try to automatically assign based on units
    eda_assigned = False
    ppg_assigned = False
    resp_assigned = False
    
    for idx, label, unit in aux_channels:
        unit_upper = unit.upper()
        assigned = False
        
        # EDA/GSR: Look for microsiemens or similar units
        if not eda_assigned and ('SIEMENS' in unit_upper or 'µS' in unit or 'US' in unit_upper):
            eda_data['channels'].append(idx)
            eda_data['labels'].append(label)
            eda_data['data'].append(time_series[:, idx])
            print(f"  {label} (unit: {unit}) → EDA/GSR (detected by unit)")
            eda_assigned = True
            assigned = True
        
        # If not assigned yet, use order-based heuristic
        if not assigned:
            if not eda_assigned:
                eda_data['channels'].append(idx)
                eda_data['labels'].append(label)
                eda_data['data'].append(time_series[:, idx])
                print(f"  {label} (unit: {unit}) → EDA/GSR (by order)")
                eda_assigned = True
            elif not ppg_assigned:
                ppg_data['channels'].append(idx)
                ppg_data['labels'].append(label)
                ppg_data['data'].append(time_series[:, idx])
                print(f"  {label} (unit: {unit}) → PPG (by order)")
                ppg_assigned = True
            elif not resp_assigned:
                resp_data['channels'].append(idx)
                resp_data['labels'].append(label)
                resp_data['data'].append(time_series[:, idx])
                print(f"  {label} (unit: {unit}) → Respiration (by order)")
                resp_assigned = True
    
    print("\n⚠️  HOW TO VERIFY ASSIGNMENTS:")
    print("  1. Check channel UNITS in the metadata (most reliable)")
    print("     - EDA/GSR: microsiemens (µS)")
    print("     - PPG: often unitless or voltage")
    print("     - RESP: often unitless or liters")
    print("  2. Check manufacturer documentation for AUX channel mapping")
    print("  3. Verify by signal characteristics:")
    print("     - EDA: slow-changing, ~0-20 µS range")
    print("     - PPG: fast oscillations matching heart rate (~60-100 BPM)")
    print("     - RESP: slow oscillations matching breathing (~12-20 breaths/min)")
    
    # Separate remaining channels (EEG channels)
    for idx, label in enumerate(channel_labels):
        if 'AUX' not in label.upper():
            eeg_data['channels'].append(idx)
            eeg_data['labels'].append(label)
            eeg_data['data'].append(time_series[:, idx])
    
    # Convert lists to numpy arrays for easier processing
    if eeg_data['data']:
        eeg_data['data'] = np.array(eeg_data['data']).T  # Shape: (n_samples, n_channels)
        eeg_data['time_stamps'] = time_stamps
        eeg_data['sampling_rate'] = sampling_rate
        print(f"\nEEG channels: {len(eeg_data['labels'])} - {eeg_data['labels']}")
        print(f"EEG data shape: {eeg_data['data'].shape}")
    
    if eda_data['data']:
        eda_data['data'] = np.array(eda_data['data']).T
        eda_data['time_stamps'] = time_stamps
        eda_data['sampling_rate'] = sampling_rate
        print(f"\nEDA channels: {len(eda_data['labels'])} - {eda_data['labels']}")
        print(f"EDA data shape: {eda_data['data'].shape}")
    
    if ppg_data['data']:
        ppg_data['data'] = np.array(ppg_data['data']).T
        ppg_data['time_stamps'] = time_stamps
        ppg_data['sampling_rate'] = sampling_rate
        print(f"\nPPG channels: {len(ppg_data['labels'])} - {ppg_data['labels']}")
        print(f"PPG data shape: {ppg_data['data'].shape}")
    
    if resp_data['data']:
        resp_data['data'] = np.array(resp_data['data']).T
        resp_data['time_stamps'] = time_stamps
        resp_data['sampling_rate'] = sampling_rate
        print(f"\nRESP channels: {len(resp_data['labels'])} - {resp_data['labels']}")
        print(f"RESP data shape: {resp_data['data'].shape}")
else:
    print("No EEG stream found!")

# ============================================================================
# PROCESS EDA SIGNAL
# ============================================================================
if PROCESS_AUX_SENSORS and len(eda_data['labels']) > 0:
    print("\n" + "="*70)
    print("PROCESSING EDA SIGNAL")
    print("="*70)
    
    # Extract the EDA signal (first column)
    eda_signal = eda_data['data'][:, 0]
    
    # Process the EDA signal
    signals, info = nk.eda_process(eda_signal, sampling_rate=int(eda_data['sampling_rate']))
    
    # Display information about the processed signal
    print(f"\nProcessed signal columns: {list(signals.columns)}")
    print(f"\nSignal shape: {signals.shape}")
    print(f"\nNumber of SCR peaks detected: {info['SCR_Peaks'].sum()}")
    
    # Display first few rows of the processed data
    print("\nFirst 10 rows of processed signals:")
    print(signals.head(10))
    
    # Display info dictionary keys
    print(f"\nInfo dictionary keys: {list(info.keys())}")
    
    # Display peak information
    if info['SCR_Peaks'].sum() > 0:
        peak_indices = np.where(info['SCR_Peaks'] == 1)[0]
        peak_times = eda_data['time_stamps'][peak_indices]
        print(f"\nSCR Peak times (first 10): {peak_times[:10]}")
    
    # ========================================================================
    # RESTING TRIAL: RollerCoasterBaselineStarted -> RollerCoasterBaselineFinished
    # ========================================================================
    baseline_phasic = None
    baseline_timestamps = None
    baseline_start_time = None
    baseline_end_time = None
    
    if markers is not None:
        baseline_start_indices = [i for i, label in enumerate(markers['labels'])
                                  if 'RollerCoasterBaselineStarted' in label]
        baseline_end_indices = [i for i, label in enumerate(markers['labels'])
                                if 'RollerCoasterBaselineFinished' in label]
        
        if baseline_start_indices and baseline_end_indices:
            baseline_start_time = markers['timestamps'][baseline_start_indices[0]]
            baseline_end_time = markers['timestamps'][baseline_end_indices[0]]
            
            if baseline_end_time > baseline_start_time:
                baseline_mask = (eda_data['time_stamps'] >= baseline_start_time) & (eda_data['time_stamps'] <= baseline_end_time)
                baseline_phasic = signals['EDA_Phasic'][baseline_mask].to_numpy()
                baseline_timestamps = eda_data['time_stamps'][baseline_mask]
                print(f"\nResting trial window: {baseline_start_time:.3f}s to {baseline_end_time:.3f}s")
                print(f"Resting phasic samples: {len(baseline_phasic)}")
            else:
                print("\n⚠️  Baseline end time occurs before start time. Check marker order.")
        else:
            print("\n⚠️  Baseline markers not found: RollerCoasterBaselineStarted/Finished")
    else:
        print("\n⚠️  No markers available for baseline extraction")
    
    # Save baseline figure if we have the window
    if baseline_phasic is not None:
        fig_baseline = plt.figure(figsize=(12, 4))
        plt.plot(baseline_timestamps, baseline_phasic, color='purple', linewidth=1)
        plt.title('Resting Trial: EDA Phasic (Baseline)', fontsize=12, fontweight='bold')
        plt.xlabel('Time (s)', fontsize=10)
        plt.ylabel('Amplitude (μS)', fontsize=10)
        plt.grid(True, alpha=0.3)
        baseline_fig_path = RESULTS_DIR / f'{subject_tag}_eda_resting_phasic_baseline.png'
        plt.tight_layout()
        plt.savefig(baseline_fig_path, dpi=300)
        plt.show()
        print(f"\nSaved resting baseline figure to: {baseline_fig_path}")
    
    # ========================================================================
    # TRIAL FIGURE: 6 panels (RollerCoasterStarted -> RollerCoasterFinished)
    # ========================================================================
    trial_windows = []
    if markers is not None:
        start_times = [markers['timestamps'][i] for i, label in enumerate(markers['labels'])
                       if 'RollerCoasterStarted' in label]
        end_times = [markers['timestamps'][i] for i, label in enumerate(markers['labels'])
                     if 'RollerCoasterFinished' in label]
        
        # Pair starts and ends in chronological order
        start_times_sorted = sorted(start_times)
        end_times_sorted = sorted(end_times)
        
        for start_time in start_times_sorted:
            end_candidates = [t for t in end_times_sorted if t > start_time]
            if end_candidates:
                end_time = end_candidates[0]
                trial_windows.append((start_time, end_time))
                end_times_sorted.remove(end_time)
    
    if trial_windows:
        # Sort by start time and keep first six trials
        trial_windows = sorted(trial_windows, key=lambda x: x[0])
        trial_windows = trial_windows[:6]
        
        fig_trials, axes_trials = plt.subplots(1, 6, figsize=(24, 4), sharey=True)
        if len(trial_windows) == 1:
            axes_trials = [axes_trials]
        
        for idx, (start_t, end_t) in enumerate(trial_windows):
            mask = (eda_data['time_stamps'] >= start_t) & (eda_data['time_stamps'] <= end_t)
            trial_time = eda_data['time_stamps'][mask]
            trial_phasic = signals['EDA_Phasic'][mask]
            
            axes_trials[idx].plot(trial_time, trial_phasic, color='red', linewidth=0.8)
            axes_trials[idx].set_title(f'Trial {idx + 1}', fontsize=10, fontweight='bold')
            axes_trials[idx].set_xlabel('Time (s)', fontsize=9)
            axes_trials[idx].grid(True, alpha=0.3)
        
        axes_trials[0].set_ylabel('Amplitude (μS)', fontsize=9)
        plt.suptitle('EDA Phasic Component per Trial (RollerCoasterStarted → RollerCoasterFinished)',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        trials_fig_path = RESULTS_DIR / f'{subject_tag}_eda_phasic_trials_6panel.png'
        plt.savefig(trials_fig_path, dpi=300)
        plt.show()
        print(f"\nSaved trial figure to: {trials_fig_path}")
    else:
        print("\n⚠️  No RollerCoasterStarted/Finished trial pairs found")
    
    # Create comprehensive visualization
    fig_eda = plt.figure(figsize=(14, 12))
    
    # Filter markers to only show RollerCoasterStarted
    if markers is not None:
        roller_coaster_indices = [i for i, label in enumerate(markers['labels']) if 'RollerCoasterStarted' in label]
        roller_coaster_times = [markers['timestamps'][i] for i in roller_coaster_indices]
    
    # Plot 1: Raw and Clean signals
    ax1 = plt.subplot(4, 1, 1)
    ax1.plot(eda_data['time_stamps'], signals['EDA_Raw'], 'gray', alpha=0.5, label='Raw', linewidth=0.8)
    ax1.plot(eda_data['time_stamps'], signals['EDA_Clean'], 'b-', label='Clean', linewidth=1)
    # Add RollerCoasterStarted markers
    if markers is not None and roller_coaster_times:
        for t in roller_coaster_times:
            ax1.axvline(x=t, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    ax1.set_title('Raw vs Clean EDA Signal', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Amplitude (μS)', fontsize=10)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Tonic component
    ax2 = plt.subplot(4, 1, 2)
    ax2.plot(eda_data['time_stamps'], signals['EDA_Tonic'], 'g-', linewidth=1)
    # Add RollerCoasterStarted markers
    if markers is not None and roller_coaster_times:
        for t in roller_coaster_times:
            ax2.axvline(x=t, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    ax2.set_title('Tonic Component (Slow Baseline)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Amplitude (μS)', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Phasic component with peaks
    ax3 = plt.subplot(4, 1, 3)
    ax3.plot(eda_data['time_stamps'], signals['EDA_Phasic'], 'r-', linewidth=1, label='Phasic')
    # Mark SCR peaks
    if info['SCR_Peaks'].sum() > 0:
        peak_indices = np.where(info['SCR_Peaks'] == 1)[0]
        ax3.scatter(eda_data['time_stamps'][peak_indices], signals['EDA_Phasic'].iloc[peak_indices], 
                   c='red', s=50, zorder=5, label=f'SCR Peaks (n={len(peak_indices)})')
    # Add RollerCoasterStarted markers
    if markers is not None and roller_coaster_times:
        for i, t in enumerate(roller_coaster_times):
            ax3.axvline(x=t, color='red', linestyle='--', alpha=0.7, linewidth=1.5,
                       label='RollerCoasterStarted' if i == 0 else '')
    ax3.set_title('Phasic Component (Event-Related Responses)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Amplitude (μS)', fontsize=10)
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: All components together
    ax4 = plt.subplot(4, 1, 4)
    ax4.plot(eda_data['time_stamps'], signals['EDA_Clean'], 'b-', label='Clean Signal', linewidth=1, alpha=0.7)
    ax4.plot(eda_data['time_stamps'], signals['EDA_Tonic'], 'g-', label='Tonic', linewidth=1.5)
    ax4.plot(eda_data['time_stamps'], signals['EDA_Phasic'], 'r-', label='Phasic', linewidth=1, alpha=0.7)
    # Add RollerCoasterStarted markers
    if markers is not None and roller_coaster_times:
        for t in roller_coaster_times:
            ax4.axvline(x=t, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    ax4.set_title('All Components Overlaid', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Time (s)', fontsize=10)
    ax4.set_ylabel('Amplitude (μS)', fontsize=10)
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('EDA Signal Processing Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
else:
    print("\nNo EDA data available for processing")

# ============================================================================
# PROCESS SHIMMER EDA (EXT) SIGNAL
# ============================================================================
if PROCESS_AUX_SENSORS and 'eda_data_EXT' in locals() and len(eda_data_EXT['labels']) > 0:
    print("\n" + "="*70)
    print("PROCESSING SHIMMER EDA (EXT) SIGNAL")
    print("="*70)

    eda_ext_signal = eda_data_EXT['data'][:, 0]
    if eda_data_EXT.get('sampling_rate') is None:
        print("⚠️  Shimmer sampling rate missing; cannot process EDA_EXT")
    else:
        signals_ext, info_ext = nk.eda_process(eda_ext_signal, sampling_rate=int(eda_data_EXT['sampling_rate']))
        print(f"\nProcessed EDA_EXT columns: {list(signals_ext.columns)}")
        print(f"\nEDA_EXT signal shape: {signals_ext.shape}")
        print(f"\nNumber of SCR peaks detected: {info_ext['SCR_Peaks'].sum()}")

        # Resting baseline window (if markers exist)
        baseline_phasic_ext = None
        baseline_timestamps_ext = None
        if markers is not None and baseline_start_time is not None and baseline_end_time is not None:
            baseline_mask_ext = (eda_data_EXT['time_stamps'] >= baseline_start_time) & (eda_data_EXT['time_stamps'] <= baseline_end_time)
            baseline_phasic_ext = signals_ext['EDA_Phasic'][baseline_mask_ext].to_numpy()
            baseline_timestamps_ext = eda_data_EXT['time_stamps'][baseline_mask_ext]

        if baseline_phasic_ext is not None and len(baseline_phasic_ext) > 0:
            fig_baseline_ext = plt.figure(figsize=(12, 4))
            plt.plot(baseline_timestamps_ext, baseline_phasic_ext, color='purple', linewidth=1)
            plt.title('Shimmer EDA_EXT: Resting Trial Phasic (Baseline)', fontsize=12, fontweight='bold')
            plt.xlabel('Time (s)', fontsize=10)
            plt.ylabel('Amplitude (μS)', fontsize=10)
            plt.grid(True, alpha=0.3)
            baseline_fig_path_ext = RESULTS_DIR / f'{subject_tag}_eda_EXT_resting_phasic_baseline.png'
            plt.tight_layout()
            plt.savefig(baseline_fig_path_ext, dpi=300)
            plt.show()
            print(f"\nSaved Shimmer EDA_EXT baseline figure to: {baseline_fig_path_ext}")

        # Trial windows (same as main EDA)
        if trial_windows:
            fig_trials_ext, axes_trials_ext = plt.subplots(1, 6, figsize=(24, 4), sharey=True)
            if len(trial_windows) == 1:
                axes_trials_ext = [axes_trials_ext]

            for idx, (start_t, end_t) in enumerate(trial_windows):
                mask_ext = (eda_data_EXT['time_stamps'] >= start_t) & (eda_data_EXT['time_stamps'] <= end_t)
                trial_time_ext = eda_data_EXT['time_stamps'][mask_ext]
                trial_phasic_ext = signals_ext['EDA_Phasic'][mask_ext]

                axes_trials_ext[idx].plot(trial_time_ext, trial_phasic_ext, color='red', linewidth=0.8)
                axes_trials_ext[idx].set_title(f'Trial {idx + 1}', fontsize=10, fontweight='bold')
                axes_trials_ext[idx].set_xlabel('Time (s)', fontsize=9)
                axes_trials_ext[idx].grid(True, alpha=0.3)

            axes_trials_ext[0].set_ylabel('Amplitude (μS)', fontsize=9)
            plt.suptitle('Shimmer EDA_EXT Phasic per Trial (RollerCoasterStarted → RollerCoasterFinished)',
                         fontsize=12, fontweight='bold')
            plt.tight_layout()
            trials_fig_path_ext = RESULTS_DIR / f'{subject_tag}_eda_EXT_phasic_trials_6panel.png'
            plt.savefig(trials_fig_path_ext, dpi=300)
            plt.show()
            print(f"\nSaved Shimmer EDA_EXT trial figure to: {trials_fig_path_ext}")

        # Comprehensive visualization
        fig_eda_ext = plt.figure(figsize=(14, 12))

        if markers is not None:
            roller_coaster_indices_ext = [i for i, label in enumerate(markers['labels']) if 'RollerCoasterStarted' in label]
            roller_coaster_times_ext = [markers['timestamps'][i] for i in roller_coaster_indices_ext]

        ax1 = plt.subplot(4, 1, 1)
        ax1.plot(eda_data_EXT['time_stamps'], signals_ext['EDA_Raw'], 'gray', alpha=0.5, label='Raw', linewidth=0.8)
        ax1.plot(eda_data_EXT['time_stamps'], signals_ext['EDA_Clean'], 'b-', label='Clean', linewidth=1)
        if markers is not None and roller_coaster_times_ext:
            for t in roller_coaster_times_ext:
                ax1.axvline(x=t, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
        ax1.set_title('Shimmer EDA_EXT: Raw vs Clean', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Amplitude (μS)', fontsize=10)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        ax2 = plt.subplot(4, 1, 2)
        ax2.plot(eda_data_EXT['time_stamps'], signals_ext['EDA_Tonic'], 'g-', linewidth=1)
        if markers is not None and roller_coaster_times_ext:
            for t in roller_coaster_times_ext:
                ax2.axvline(x=t, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
        ax2.set_title('Shimmer EDA_EXT: Tonic', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Amplitude (μS)', fontsize=10)
        ax2.grid(True, alpha=0.3)

        ax3 = plt.subplot(4, 1, 3)
        ax3.plot(eda_data_EXT['time_stamps'], signals_ext['EDA_Phasic'], 'r-', linewidth=1, label='Phasic')
        if info_ext['SCR_Peaks'].sum() > 0:
            peak_indices_ext = np.where(info_ext['SCR_Peaks'] == 1)[0]
            ax3.scatter(eda_data_EXT['time_stamps'][peak_indices_ext], signals_ext['EDA_Phasic'].iloc[peak_indices_ext],
                        c='red', s=50, zorder=5, label=f'SCR Peaks (n={len(peak_indices_ext)})')
        if markers is not None and roller_coaster_times_ext:
            for i, t in enumerate(roller_coaster_times_ext):
                ax3.axvline(x=t, color='red', linestyle='--', alpha=0.7, linewidth=1.5,
                            label='RollerCoasterStarted' if i == 0 else '')
        ax3.set_title('Shimmer EDA_EXT: Phasic', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Amplitude (μS)', fontsize=10)
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)

        ax4 = plt.subplot(4, 1, 4)
        ax4.plot(eda_data_EXT['time_stamps'], signals_ext['EDA_Clean'], 'b-', label='Clean', linewidth=1, alpha=0.7)
        ax4.plot(eda_data_EXT['time_stamps'], signals_ext['EDA_Tonic'], 'g-', label='Tonic', linewidth=1.5)
        ax4.plot(eda_data_EXT['time_stamps'], signals_ext['EDA_Phasic'], 'r-', label='Phasic', linewidth=1, alpha=0.7)
        if markers is not None and roller_coaster_times_ext:
            for t in roller_coaster_times_ext:
                ax4.axvline(x=t, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
        ax4.set_title('Shimmer EDA_EXT: All Components', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Time (s)', fontsize=10)
        ax4.set_ylabel('Amplitude (μS)', fontsize=10)
        ax4.legend(loc='upper right')
        ax4.grid(True, alpha=0.3)

        plt.suptitle('Shimmer EDA_EXT Signal Processing Results', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
else:
    print("\nNo Shimmer EDA_EXT data available for processing")

# ============================================================================
# PROCESS PPG SIGNAL
# ============================================================================
if PROCESS_AUX_SENSORS and len(ppg_data['labels']) > 0:
    print("\n" + "="*70)
    print("PROCESSING PPG SIGNAL")
    print("="*70)
    
    # Extract the PPG signal (first column)
    ppg_signal = ppg_data['data'][:, 0]
    
    # Process the PPG signal
    ppg_signals, ppg_info = nk.ppg_process(ppg_signal, sampling_rate=int(ppg_data['sampling_rate']))
    
    # Display information about the processed signal
    print(f"\nProcessed PPG signal columns: {list(ppg_signals.columns)}")
    print(f"\nPPG signal shape: {ppg_signals.shape}")
    
    # Visualize the PPG processing
    nk.ppg_plot(ppg_signals, ppg_info)
    
    # ========================================================================
    # RESTING TRIAL: Heart Rate Baseline
    # ========================================================================
    baseline_hr = None
    baseline_hr_timestamps = None
    
    if markers is not None and baseline_start_time is not None and baseline_end_time is not None:
        baseline_mask_ppg = (ppg_data['time_stamps'] >= baseline_start_time) & (ppg_data['time_stamps'] <= baseline_end_time)
        baseline_hr = ppg_signals['PPG_Rate'][baseline_mask_ppg].to_numpy()
        baseline_hr_timestamps = ppg_data['time_stamps'][baseline_mask_ppg]
        print(f"\nResting trial HR window: {baseline_start_time:.3f}s to {baseline_end_time:.3f}s")
        print(f"Resting HR samples: {len(baseline_hr)}")
        print(f"Mean HR in baseline: {baseline_hr.mean():.1f} BPM")
    
    # Save baseline figure if we have the window
    if baseline_hr is not None:
        fig_hr_baseline = plt.figure(figsize=(12, 4))
        plt.plot(baseline_hr_timestamps, baseline_hr, color='green', linewidth=1)
        plt.title('Resting Trial: Heart Rate (Baseline)', fontsize=12, fontweight='bold')
        plt.xlabel('Time (s)', fontsize=10)
        plt.ylabel('Heart Rate (BPM)', fontsize=10)
        plt.grid(True, alpha=0.3)
        hr_baseline_fig_path = RESULTS_DIR / f'{subject_tag}_hr_resting_baseline.png'
        plt.tight_layout()
        plt.savefig(hr_baseline_fig_path, dpi=300)
        plt.show()
        print(f"\nSaved HR resting baseline figure to: {hr_baseline_fig_path}")
    
    # ========================================================================
    # TRIAL FIGURE: 6 panels (Heart Rate per trial)
    # ========================================================================
    if trial_windows:
        fig_hr_trials, axes_hr_trials = plt.subplots(1, 6, figsize=(24, 4), sharey=True)
        if len(trial_windows) == 1:
            axes_hr_trials = [axes_hr_trials]
        
        for idx, (start_t, end_t) in enumerate(trial_windows):
            mask_ppg = (ppg_data['time_stamps'] >= start_t) & (ppg_data['time_stamps'] <= end_t)
            trial_time_ppg = ppg_data['time_stamps'][mask_ppg]
            trial_hr = ppg_signals['PPG_Rate'][mask_ppg]
            
            axes_hr_trials[idx].plot(trial_time_ppg, trial_hr, color='green', linewidth=0.8)
            axes_hr_trials[idx].set_title(f'Trial {idx + 1}', fontsize=10, fontweight='bold')
            axes_hr_trials[idx].set_xlabel('Time (s)', fontsize=9)
            axes_hr_trials[idx].grid(True, alpha=0.3)
        
        axes_hr_trials[0].set_ylabel('Heart Rate (BPM)', fontsize=9)
        plt.suptitle('Heart Rate per Trial (RollerCoasterStarted → RollerCoasterFinished)',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        hr_trials_fig_path = RESULTS_DIR / f'{subject_tag}_hr_trials_6panel.png'
        plt.savefig(hr_trials_fig_path, dpi=300)
        plt.show()
        print(f"\nSaved HR trial figure to: {hr_trials_fig_path}")
else:
    print("\nNo PPG data available for processing")

# ============================================================================
# PROCESS SHIMMER PPG (EXT) SIGNAL
# ============================================================================
if PROCESS_AUX_SENSORS and 'ppg_data_EXT' in locals() and len(ppg_data_EXT['labels']) > 0:
    print("\n" + "="*70)
    print("PROCESSING SHIMMER PPG (EXT) SIGNAL")
    print("="*70)

    ppg_ext_signal = ppg_data_EXT['data'][:, 0]
    if ppg_data_EXT.get('sampling_rate') is None:
        print("⚠️  Shimmer sampling rate missing; cannot process PPG_EXT")
    else:
        ppg_signals_ext, ppg_info_ext = nk.ppg_process(ppg_ext_signal, sampling_rate=int(ppg_data_EXT['sampling_rate']))
        print(f"\nProcessed PPG_EXT columns: {list(ppg_signals_ext.columns)}")
        print(f"\nPPG_EXT signal shape: {ppg_signals_ext.shape}")

        nk.ppg_plot(ppg_signals_ext, ppg_info_ext)

        baseline_hr_ext = None
        baseline_hr_timestamps_ext = None
        if markers is not None and baseline_start_time is not None and baseline_end_time is not None:
            baseline_mask_ppg_ext = (ppg_data_EXT['time_stamps'] >= baseline_start_time) & (ppg_data_EXT['time_stamps'] <= baseline_end_time)
            baseline_hr_ext = ppg_signals_ext['PPG_Rate'][baseline_mask_ppg_ext].to_numpy()
            baseline_hr_timestamps_ext = ppg_data_EXT['time_stamps'][baseline_mask_ppg_ext]

        if baseline_hr_ext is not None and len(baseline_hr_ext) > 0:
            fig_hr_baseline_ext = plt.figure(figsize=(12, 4))
            plt.plot(baseline_hr_timestamps_ext, baseline_hr_ext, color='green', linewidth=1)
            plt.title('Shimmer PPG_EXT: Resting Trial Heart Rate (Baseline)', fontsize=12, fontweight='bold')
            plt.xlabel('Time (s)', fontsize=10)
            plt.ylabel('Heart Rate (BPM)', fontsize=10)
            plt.grid(True, alpha=0.3)
            hr_baseline_fig_path_ext = RESULTS_DIR / f'{subject_tag}_hr_EXT_resting_baseline.png'
            plt.tight_layout()
            plt.savefig(hr_baseline_fig_path_ext, dpi=300)
            plt.show()
            print(f"\nSaved Shimmer PPG_EXT baseline figure to: {hr_baseline_fig_path_ext}")

        if trial_windows:
            fig_hr_trials_ext, axes_hr_trials_ext = plt.subplots(1, 6, figsize=(24, 4), sharey=True)
            if len(trial_windows) == 1:
                axes_hr_trials_ext = [axes_hr_trials_ext]

            for idx, (start_t, end_t) in enumerate(trial_windows):
                mask_ppg_ext = (ppg_data_EXT['time_stamps'] >= start_t) & (ppg_data_EXT['time_stamps'] <= end_t)
                trial_time_ppg_ext = ppg_data_EXT['time_stamps'][mask_ppg_ext]
                trial_hr_ext = ppg_signals_ext['PPG_Rate'][mask_ppg_ext]

                axes_hr_trials_ext[idx].plot(trial_time_ppg_ext, trial_hr_ext, color='green', linewidth=0.8)
                axes_hr_trials_ext[idx].set_title(f'Trial {idx + 1}', fontsize=10, fontweight='bold')
                axes_hr_trials_ext[idx].set_xlabel('Time (s)', fontsize=9)
                axes_hr_trials_ext[idx].grid(True, alpha=0.3)

            axes_hr_trials_ext[0].set_ylabel('Heart Rate (BPM)', fontsize=9)
            plt.suptitle('Shimmer PPG_EXT: Heart Rate per Trial (RollerCoasterStarted → RollerCoasterFinished)',
                         fontsize=12, fontweight='bold')
            plt.tight_layout()
            hr_trials_fig_path_ext = RESULTS_DIR / f'{subject_tag}_hr_EXT_trials_6panel.png'
            plt.savefig(hr_trials_fig_path_ext, dpi=300)
            plt.show()
            print(f"\nSaved Shimmer PPG_EXT trial figure to: {hr_trials_fig_path_ext}")
else:
    print("\nNo Shimmer PPG_EXT data available for processing")

# ============================================================================
# PROCESS RESPIRATION SIGNAL
# ============================================================================
if PROCESS_AUX_SENSORS and len(resp_data['labels']) > 0:
    print("\n" + "="*70)
    print("PROCESSING RESPIRATION SIGNAL")
    print("="*70)
    
    # Extract the Respiration signal (first column)
    rsp_signal = resp_data['data'][:, 0]
    
    # Process the Respiration signal
    rsp_signals, rsp_info = nk.rsp_process(rsp_signal, sampling_rate=int(resp_data['sampling_rate']))
    
    # Display information about the processed signal
    print(f"\nProcessed RSP signal columns: {list(rsp_signals.columns)}")
    print(f"\nRSP signal shape: {rsp_signals.shape}")
    
    # Visualize the Respiration processing
    nk.rsp_plot(rsp_signals, rsp_info)
    
    # ========================================================================
    # RESTING TRIAL: Respiration Baseline
    # ========================================================================
    baseline_rsp = None
    baseline_rsp_timestamps = None
    
    if markers is not None and baseline_start_time is not None and baseline_end_time is not None:
        baseline_mask_rsp = (resp_data['time_stamps'] >= baseline_start_time) & (resp_data['time_stamps'] <= baseline_end_time)
        baseline_rsp = rsp_signals['RSP_Clean'][baseline_mask_rsp].to_numpy()
        baseline_rsp_timestamps = resp_data['time_stamps'][baseline_mask_rsp]
        print(f"\nResting trial RSP window: {baseline_start_time:.3f}s to {baseline_end_time:.3f}s")
        print(f"Resting RSP samples: {len(baseline_rsp)}")
    
    # Save baseline figure if we have the window (using Respiration Rate)
    if baseline_rsp is not None:
        baseline_rsp_rate = rsp_signals['RSP_Rate'][baseline_mask_rsp].to_numpy()
        
        fig_rsp_baseline = plt.figure(figsize=(12, 4))
        plt.plot(baseline_rsp_timestamps, baseline_rsp_rate, color='teal', linewidth=1)
        plt.title('Resting Trial: Respiration Rate (Baseline)', fontsize=12, fontweight='bold')
        plt.xlabel('Time (s)', fontsize=10)
        plt.ylabel('Breathing Rate (breaths/min)', fontsize=10)
        plt.grid(True, alpha=0.3)
        mean_baseline_rate = np.nanmean(baseline_rsp_rate)
        plt.axhline(y=mean_baseline_rate, color='red', linestyle='--', linewidth=1, alpha=0.7, label=f'Mean: {mean_baseline_rate:.2f} breaths/min')
        plt.legend()
        rsp_baseline_fig_path = RESULTS_DIR / f'{subject_tag}_rsp_resting_rate_baseline.png'
        plt.tight_layout()
        plt.savefig(rsp_baseline_fig_path, dpi=300)
        plt.show()
        print(f"\nSaved RSP resting baseline figure to: {rsp_baseline_fig_path}")
        print(f"Mean baseline breathing rate: {mean_baseline_rate:.2f} breaths/min")
    
    # ========================================================================
    # TRIAL FIGURE: 6 panels (Respiration Rate per trial)
    # ========================================================================
    if trial_windows:
        fig_rsp_trials, axes_rsp_trials = plt.subplots(1, 6, figsize=(24, 4), sharey=True)
        if len(trial_windows) == 1:
            axes_rsp_trials = [axes_rsp_trials]
        
        for idx, (start_t, end_t) in enumerate(trial_windows):
            mask_rsp = (resp_data['time_stamps'] >= start_t) & (resp_data['time_stamps'] <= end_t)
            trial_time_rsp = resp_data['time_stamps'][mask_rsp]
            trial_rsp_rate = rsp_signals['RSP_Rate'][mask_rsp]
            
            axes_rsp_trials[idx].plot(trial_time_rsp, trial_rsp_rate, color='teal', linewidth=0.8)
            
            # Add mean rate line for this trial
            trial_mean_rate = np.nanmean(trial_rsp_rate)
            axes_rsp_trials[idx].axhline(y=trial_mean_rate, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
            
            axes_rsp_trials[idx].set_title(f'Trial {idx + 1}\nMean: {trial_mean_rate:.1f} br/min', fontsize=9, fontweight='bold')
            axes_rsp_trials[idx].set_xlabel('Time (s)', fontsize=9)
            axes_rsp_trials[idx].grid(True, alpha=0.3)
        
        axes_rsp_trials[0].set_ylabel('Breathing Rate (breaths/min)', fontsize=9)
        plt.suptitle('Respiration Rate per Trial (RollerCoasterStarted → RollerCoasterFinished)',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        rsp_trials_fig_path = RESULTS_DIR / f'{subject_tag}_rsp_rate_trials_6panel.png'
        plt.savefig(rsp_trials_fig_path, dpi=300)
        plt.show()
        print(f"\nSaved RSP trial figure to: {rsp_trials_fig_path}")
else:
    print("\nNo Respiration data available for processing")

# ============================================================================
# EEG ANALYSIS - PREPROCESSING, ERPs, POWER SPECTRAL, AND CONNECTIVITY
# ============================================================================
import mne
from scipy import signal
from scipy.stats import zscore

if PROCESS_EEG and len(eeg_data['labels']) > 0:
    print("\n" + "="*70)
    print("EEG PREPROCESSING AND ANALYSIS")
    print("="*70)
    
    # Create MNE Info object
    ch_names = eeg_data['labels']
    ch_types = ['eeg'] * len(ch_names)
    sfreq = int(eeg_data['sampling_rate'])
    
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    
    # Create Raw object (data should be in channels x timepoints format)
    raw = mne.io.RawArray(eeg_data['data'].T, info)
    print(f"\nRaw EEG data created: {raw.n_times} timepoints, {raw.info['nchan']} channels")
    
    # ========================================================================
    # PREPROCESSING - MANDATORY STEPS
    # ========================================================================
    print("\n" + "-"*70)
    print("PREPROCESSING")
    print("-"*70)
    
    # 1. Set standard 10-20 electrode positions (if available)
    try:
        raw.set_montage('standard_1020')
        print("\n✓ Standard 10-20 electrode positions set")
    except:
        print("\n⚠ Could not set standard montage - channels may not match 10-20 system")
    
    # 2. High-pass filter (remove DC drift and very low frequencies)
    raw.filter(l_freq=0.5, h_freq=None, verbose=False)
    print("✓ High-pass filter applied (0.5 Hz)")
    
    # 3. Notch filter (remove line noise at 50 Hz)
    raw.notch_filter(freqs=50, verbose=False)
    print("✓ Notch filter applied (50 Hz line noise)")
    
    # 4. Re-reference to common average reference
    raw.set_eeg_reference('average', verbose=False)
    print("✓ Common Average Reference applied")
    
    # 5. Automated bad channel detection (channels with high variance compared to others)
    print("\n" + "-"*70)
    print("BAD CHANNEL DETECTION")
    print("-"*70)
    
    # Calculate variance for each channel
    channel_vars = np.var(raw.get_data(), axis=1)
    
    # Compare channels to each other (z-score across channels)
    z_scores = zscore(channel_vars)
    
    print("\nChecking variance across channels...")
    print(f"{'Channel':<10} {'Variance':<15} {'Z-score':<10} {'Status'}")
    print("-" * 60)
    
    bad_chans = []
    for i, (ch_name, var, z) in enumerate(zip(raw.ch_names, channel_vars, z_scores)):
        status = "OK"
        if z > 3:  # Channel has much higher variance than others
            bad_chans.append(ch_name)
            status = "BAD (high)"
        elif z < -3:  # Channel has much lower variance (flat)
            bad_chans.append(ch_name)
            status = "BAD (flat)"
        
        if status != "OK" or i < 5:  # Show first 5 channels + all bad ones
            print(f"{ch_name:<10} {var:<15.2f} {z:<10.2f} {status}")
    
    if len(bad_chans) > 0:
        print(f"\n✓ Marked {len(bad_chans)} bad channel(s): {bad_chans}")
        raw.info['bads'].extend(bad_chans)
        
        # Interpolate bad channels
        raw.interpolate_bads(reset_bads=True)
        print(f"✓ Bad channels interpolated from neighbors")
    else:
        print("\n✓ No bad channels detected")
    
    # 6. Low-pass filter (remove high-frequency noise)
    raw.filter(l_freq=None, h_freq=40, verbose=False)
    print("✓ Low-pass filter applied (40 Hz)")
    
    # 7. Independent Component Analysis (ICA) for artifact removal
    print("\n" + "-"*70)
    print("RUNNING ICA (Independent Component Analysis)")
    print("-"*70)
    print("Fitting ICA to remove eye movements and muscle artifacts...")
    print("This may take a few minutes...")
    
    from mne.preprocessing import ICA
    
    # Fit ICA (use 20 components or n_channels-1, whichever is smaller)
    n_components = min(20, raw.info['nchan'] - 1)
    ica = ICA(n_components=n_components, random_state=42, max_iter=800)
    ica.fit(raw)
    
    print(f"✓ ICA fitted with {n_components} components")
    
    # Automatically detect EOG (eye) artifacts using frontal channels FP1 and FP2
    # These channels are closest to the eyes and pick up eye movement artifacts
    eog_channels = ['Fp1', 'Fp2']  # Standard 10-20 frontal channels
    
    try:
        eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=eog_channels, threshold=2.5)
        
        if len(eog_indices) > 0:
            print(f"✓ Detected {len(eog_indices)} ICA component(s) with eye artifacts: {eog_indices}")
            ica.exclude = eog_indices
        else:
            print("✓ No clear eye artifact components detected")
    except Exception as e:
        print(f"⚠ Could not auto-detect EOG components: {e}")
        print("  Continuing without automatic EOG removal")
    
    # Apply ICA to remove artifacts
    raw = ica.apply(raw)
    print(f"✓ ICA applied - removed {len(ica.exclude)} component(s)")
    
    print(f"\n✓ Preprocessing complete. Final data shape: {raw.get_data().shape}")
    
    # ========================================================================
    # CREATE EPOCHS BASED ON MARKERS
    # ========================================================================
    if markers is not None:
        print("\n" + "-"*70)
        print("EPOCH CREATION")
        print("-"*70)
        
        # IMPORTANT: Align marker timestamps with EEG timestamps
        # Different streams may have different time references, so we need to synchronize them
        eeg_start_time = eeg_data['time_stamps'][0]
        eeg_end_time = eeg_data['time_stamps'][-1]
        eeg_duration = eeg_end_time - eeg_start_time
        
        # Find the offset by matching marker times to EEG times
        marker_timestamps = markers['timestamps']
        
        # Most markers should fall within the EEG recording window
        # Find markers that are close to the EEG time range
        marker_offset = None
        
        # Strategy: find the most common offset by checking which marker would align with EEG start
        # We look at the first marker that logically should be near the start
        first_valid_marker_idx = 0
        marker_offset = eeg_start_time - marker_timestamps[first_valid_marker_idx]
        
        print(f"\nTimestamp synchronization:")
        print(f"  - EEG start time: {eeg_start_time:.2f} s")
        print(f"  - EEG end time: {eeg_end_time:.2f} s")
        print(f"  - EEG duration: {eeg_duration:.2f} s")
        print(f"  - Marker start time (raw): {marker_timestamps[0]:.2f} s")
        print(f"  - Marker end time (raw): {marker_timestamps[-1]:.2f} s")
        print(f"  - Calculated offset: {marker_offset:.2f} s")
        
        # Adjust marker timestamps to align with EEG
        adjusted_marker_timestamps = marker_timestamps + marker_offset
        print(f"  - First marker (adjusted): {adjusted_marker_timestamps[0]:.2f} s")
        print(f"  - Last marker (adjusted): {adjusted_marker_timestamps[-1]:.2f} s")
        
        # Create events array for MNE
        # Events are: [sample_index, duration, event_id]
        events_list = []
        event_id_dict = {}
        event_counter = {}
        
        for i, (timestamp, label) in enumerate(zip(adjusted_marker_timestamps, markers['labels'])):
            # Extract just the event name (before the pipe character if metadata is present)
            event_name = label.split('|')[0] if '|' in label else label
            
            # Convert timestamp to sample index using EEG time reference
            # sample_index = (timestamp - eeg_start_time) * sfreq
            relative_time = timestamp - eeg_start_time
            sample_idx = int(relative_time * sfreq)
            
            if 0 <= sample_idx < raw.n_times:
                # Create unique event ID for each unique marker type
                if event_name not in event_counter:
                    event_counter[event_name] = 0
                    event_id_dict[event_name] = len(event_id_dict) + 1
                
                events_list.append([sample_idx, 0, event_id_dict[event_name]])
        
        events = np.array(events_list) if events_list else np.array([]).reshape(0, 3)
        print(f"\nTotal events found: {len(events)}")
        print(f"Event types: {list(event_id_dict.keys())}")
        
        # Create epochs around RollerCoasterStarted markers
        # Baseline: -2 to 0 seconds (before event - just 2 seconds for shorter correction window)
        # Post-event: 0 to 10 seconds (after event) to capture full response
        if len(events) > 0 and 'RollerCoasterStarted' in event_id_dict:
            tmin, tmax = -10, 15  # -10 to +15 seconds (for full epoch window)
            baseline_win = (-2, 0)  # Baseline correction window: -2 to 0 seconds
            
            epochs_all = mne.Epochs(raw, events, event_id=event_id_dict['RollerCoasterStarted'],
                                   tmin=tmin, tmax=tmax, baseline=baseline_win, preload=True, verbose=False)
            
            # Limit to first 6 trials
            epochs = epochs_all[:6]
            
            print(f"\n✓ Epochs created: {len(epochs)} events (limited to first 6 trials)")
            print(f"  - Epoch window: {tmin} to {tmax} s")
            print(f"  - Baseline correction: {baseline_win[0]} to {baseline_win[1]} s")
            
            # ====================================================================
            # EVENT-RELATED POTENTIALS (ERPs) - All trials in one figure
            # ====================================================================
            print("\n" + "-"*70)
            print("COMPUTING EVENT-RELATED POTENTIALS (ERPs)")
            print("-"*70)
            
            # Prepare list to store all evoked responses
            all_evoked = []
            evoked_labels = []
            
            # First, get ERP for the baseline resting trial if available
            if 'RollerCoasterBaselineStarted' in event_id_dict:
                print("\nProcessing baseline/resting trial...")
                
                # Find baseline events in the events array
                baseline_event_indices = np.where(events[:, 2] == event_id_dict['RollerCoasterBaselineStarted'])[0]
                print(f"  Found {len(baseline_event_indices)} baseline event(s)")
                
                if len(baseline_event_indices) > 0:
                    try:
                        # Don't apply baseline correction to baseline trial - show raw voltage
                        baseline_trial_epochs = mne.Epochs(raw, events, 
                                                           event_id=event_id_dict['RollerCoasterBaselineStarted'],
                                                           tmin=0, tmax=15, baseline=None, 
                                                           preload=True, verbose=False, reject=None)
                        
                        if len(baseline_trial_epochs) > 0:
                            baseline_evoked = baseline_trial_epochs.average()
                            baseline_evoked_window = baseline_evoked.copy().crop(tmin=0, tmax=10)
                            all_evoked.append(baseline_evoked_window)
                            evoked_labels.append('Baseline')
                            print(f"✓ Baseline trial found and added")
                        else:
                            print("⚠ Baseline epochs were created but empty after processing")
                    except Exception as e:
                        print(f"⚠ Error creating baseline epochs: {e}")
                else:
                    print("⚠ No baseline events found in events array")
            else:
                print("\n⚠ RollerCoasterBaselineStarted not in event types")
            
            # Now get ERPs for each RollerCoasterStarted trial
            print(f"\nProcessing {len(epochs)} RollerCoasterStarted trials...")
            
            for trial_idx in range(len(epochs)):
                # Extract single epoch and convert to Epochs object for averaging
                single_epoch_data = epochs[trial_idx:trial_idx+1]
                evoked_trial = single_epoch_data.average()
                evoked_trial_window = evoked_trial.copy().crop(tmin=0, tmax=10)
                
                all_evoked.append(evoked_trial_window)
                evoked_labels.append(f'Trial {trial_idx + 1}')
            
            # Create a figure with time segments (columns) and trials (rows)
            time_segments = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10)]  # 2-second windows
            n_trials = len(all_evoked)
            n_times = len(time_segments)
            
            # Diagnostic: Check if baseline trial has different scale
            print(f"\nDiagnostic - Amplitude ranges:")
            for idx, (evoked, label) in enumerate(zip(all_evoked, evoked_labels)):
                data_range = evoked.get_data()
                print(f"  {label}: min={data_range.min():.2f}, max={data_range.max():.2f}, std={data_range.std():.2f} µV")
            
            # First, find global min/max across all segments and trials for consistent colorbar
            # Exclude baseline and Trial 2 (with artifacts) from colorbar scaling
            all_segment_data = []
            all_segment_data_clean = []  # Exclude baseline and Trial 2
            
            for idx, (evoked, label) in enumerate(zip(all_evoked, evoked_labels)):
                for t_start, t_end in time_segments:
                    segment_data = evoked.copy().crop(tmin=t_start, tmax=t_end)
                    segment_mean = segment_data.get_data().mean(axis=1)
                    all_segment_data.append(segment_mean)
                    
                    # Skip baseline and Trial 2 for colorbar scaling
                    if label != 'Baseline' and label != 'Trial 2':
                        all_segment_data_clean.append(segment_mean)
            
            all_segment_data = np.array(all_segment_data)
            
            # Use clean trials for colorbar (exclude baseline and Trial 2)
            if len(all_segment_data_clean) > 0:
                scaling_data = np.array(all_segment_data_clean)
                print(f"\n✓ Using clean trials for colorbar scaling (excluding baseline and Trial 2)")
            else:
                scaling_data = all_segment_data
                print(f"\n✓ Using all data for colorbar scaling")
            
            vmin = np.percentile(scaling_data, 2)  # 2nd percentile for robustness
            vmax = np.percentile(scaling_data, 98)  # 98th percentile
            
            print(f"✓ Colorbar range: {vmin:.2f} to {vmax:.2f} µV")
            
            fig = plt.figure(figsize=(18, 3.5 * n_trials))
            
            for trial_idx, (evoked, label) in enumerate(zip(all_evoked, evoked_labels)):
                for time_idx, (t_start, t_end) in enumerate(time_segments):
                    ax = plt.subplot(n_trials, n_times, trial_idx * n_times + time_idx + 1)
                    
                    # Get data for this time segment
                    segment_data = evoked.copy().crop(tmin=t_start, tmax=t_end)
                    segment_mean = segment_data.get_data().mean(axis=1)
                    
                    # Plot topomap with shared colorbar scale
                    mne.viz.plot_topomap(
                        segment_mean,
                        segment_data.info,
                        axes=ax,
                        show=False,
                        contours=False,
                        vlim=(vmin, vmax)
                    )
                    
                    # Set titles
                    if trial_idx == 0:
                        ax.set_title(f'{t_start}-{t_end} s', fontsize=10, fontweight='bold', pad=5)
                    if time_idx == 0:
                        ax.set_ylabel(label, fontsize=10, fontweight='bold', labelpad=10)
            
            # Add a colorbar
            sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=plt.Normalize(vmin=vmin, vmax=vmax))
            sm.set_array([])
            cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
            cbar = fig.colorbar(sm, cax=cbar_ax)
            cbar.set_label('Voltage (µV)', fontsize=10)
            
            plt.suptitle('Event-Related Potentials (ERP) - Trials vs Time Segments (0-10 s post-stimulus)', 
                        fontsize=13, fontweight='bold', y=0.995)
            plt.tight_layout(rect=[0, 0, 0.91, 0.99])
            plt.savefig(RESULTS_DIR / f'{subject_tag}_erp_all_trials_time_segments.png', dpi=300, bbox_inches='tight')
            plt.show()
            print(f"\n✓ Saved ERP time-segment analysis to: {RESULTS_DIR / f'{subject_tag}_erp_all_trials_time_segments.png'}")
            print(f"✓ Generated {n_trials}x{n_times} grid ({n_trials} trials, {n_times} time segments)")
            print(f"✓ All panels use same colorbar scale (vmin={vmin:.2f}, vmax={vmax:.2f} µV)")
            
            # ====================================================================
            # POWER SPECTRAL DENSITY (PSD) - Frequency bands
            # ====================================================================
            print("\n" + "-"*70)
            print("COMPUTING POWER SPECTRAL DENSITY (PSD)")
            print("-"*70)
            
            # Compute PSD for baseline and trial periods
            baseline_epochs = epochs.copy().crop(tmin=-5, tmax=0)
            trial_epochs = epochs.copy().crop(tmin=0, tmax=10)
            
            psd_baseline = baseline_epochs.compute_psd(method='welch', fmin=0.5, fmax=40)
            psd_trial = trial_epochs.compute_psd(method='welch', fmin=0.5, fmax=40)
            
            print(f"✓ PSD computed for baseline and trial periods")
            
            # Skip detailed PSD topomaps due to Spectrum object data access issues
            # Instead, proceed to connectivity and heatmap analysis which provides similar insights
            
            # ====================================================================
            # CONNECTIVITY ANALYSIS - Phase-Locking Value (PLV)
            # ====================================================================
            print("\n" + "-"*70)
            print("COMPUTING CONNECTIVITY (Phase-Locking Value)")
            print("-"*70)
            
            from scipy.signal import hilbert
            
            # Compute PLV for a specific frequency band (Alpha - most common)
            fmin, fmax = 8, 12  # Alpha band
            
            # Filter data to alpha band
            data_baseline = baseline_epochs.get_data()
            data_trial = trial_epochs.get_data()
            
            # Apply band-pass filter
            sos = signal.butter(4, [fmin, fmax], btype='band', fs=sfreq, output='sos')
            data_baseline_filt = np.array([signal.sosfilt(sos, d) for d in data_baseline.reshape(data_baseline.shape[0] * data_baseline.shape[1], -1)])
            data_baseline_filt = data_baseline_filt.reshape(data_baseline.shape)
            
            data_trial_filt = np.array([signal.sosfilt(sos, d) for d in data_trial.reshape(data_trial.shape[0] * data_trial.shape[1], -1)])
            data_trial_filt = data_trial_filt.reshape(data_trial.shape)
            
            # Compute phase
            phase_baseline = np.angle(hilbert(data_baseline_filt, axis=2))
            phase_trial = np.angle(hilbert(data_trial_filt, axis=2))
            
            # Compute PLV (mean absolute value of phase differences)
            def compute_plv(phase_data):
                """Compute Phase-Locking Value between all channel pairs"""
                n_epochs, n_chans, n_times = phase_data.shape
                plv_matrix = np.zeros((n_chans, n_chans))
                
                for i in range(n_chans):
                    for j in range(i+1, n_chans):
                        phase_diff = phase_data[:, i, :] - phase_data[:, j, :]
                        plv = np.abs(np.mean(np.exp(1j * phase_diff)))
                        plv_matrix[i, j] = plv
                        plv_matrix[j, i] = plv
                
                return plv_matrix
            
            plv_baseline = compute_plv(phase_baseline)
            plv_trial = compute_plv(phase_trial)
            
            print(f"✓ PLV computed for Alpha band ({fmin}-{fmax} Hz)")
            
            # Visualize PLV as heatmaps
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            im0 = axes[0].imshow(plv_baseline, cmap='hot', aspect='auto', vmin=0, vmax=1)
            axes[0].set_title('Connectivity (PLV) - Baseline', fontsize=12, fontweight='bold')
            axes[0].set_xlabel('Channel', fontsize=10)
            axes[0].set_ylabel('Channel', fontsize=10)
            axes[0].set_xticks(range(0, len(ch_names), 4))
            axes[0].set_yticks(range(0, len(ch_names), 4))
            plt.colorbar(im0, ax=axes[0], label='PLV')
            
            im1 = axes[1].imshow(plv_trial, cmap='hot', aspect='auto', vmin=0, vmax=1)
            axes[1].set_title('Connectivity (PLV) - Trial', fontsize=12, fontweight='bold')
            axes[1].set_xlabel('Channel', fontsize=10)
            axes[1].set_ylabel('Channel', fontsize=10)
            axes[1].set_xticks(range(0, len(ch_names), 4))
            axes[1].set_yticks(range(0, len(ch_names), 4))
            plt.colorbar(im1, ax=axes[1], label='PLV')
            
            plt.suptitle(f'Phase-Locking Value (Alpha Band: {fmin}-{fmax} Hz)', 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(RESULTS_DIR / f'{subject_tag}_connectivity_plv_heatmap.png', dpi=300, bbox_inches='tight')
            plt.show()
            print(f"✓ Saved connectivity heatmaps to: {RESULTS_DIR / f'{subject_tag}_connectivity_plv_heatmap.png'}")
            
            # ====================================================================
            # HEATMAPS - Time x Channel for full window
            # ====================================================================
            print("\n" + "-"*70)
            print("CREATING HEATMAPS - TIME x CHANNEL")
            print("-"*70)
            
            # Average epochs for heatmap visualization
            avg_data_baseline = baseline_epochs.average().get_data()
            avg_data_trial = trial_epochs.average().get_data()
            
            times_baseline = baseline_epochs.average().times
            times_trial = trial_epochs.average().times
            
            fig, axes = plt.subplots(2, 1, figsize=(14, 10))
            
            # Baseline heatmap
            im0 = axes[0].imshow(avg_data_baseline, aspect='auto', cmap='RdBu_r', 
                                origin='lower', extent=[times_baseline[0], times_baseline[-1], 0, len(ch_names)])
            axes[0].set_title('EEG Amplitude Heatmap - Baseline (-10 to 0 s)', fontsize=12, fontweight='bold')
            axes[0].set_ylabel('Channel', fontsize=10)
            axes[0].set_yticks(range(0, len(ch_names), 4))
            axes[0].set_yticklabels(ch_names[::4], fontsize=8)
            cbar0 = plt.colorbar(im0, ax=axes[0], label='Amplitude (µV)')
            
            # Trial heatmap
            im1 = axes[1].imshow(avg_data_trial, aspect='auto', cmap='RdBu_r',
                                origin='lower', extent=[times_trial[0], times_trial[-1], 0, len(ch_names)])
            axes[1].set_title('EEG Amplitude Heatmap - Trial (0 to 15 s post-stimulus)', fontsize=12, fontweight='bold')
            axes[1].set_xlabel('Time (s)', fontsize=10)
            axes[1].set_ylabel('Channel', fontsize=10)
            axes[1].set_yticks(range(0, len(ch_names), 4))
            axes[1].set_yticklabels(ch_names[::4], fontsize=8)
            cbar1 = plt.colorbar(im1, ax=axes[1], label='Amplitude (µV)')
            
            plt.tight_layout()
            plt.savefig(RESULTS_DIR / f'{subject_tag}_eeg_amplitude_heatmaps.png', dpi=300, bbox_inches='tight')
            plt.show()
            print(f"✓ Saved EEG amplitude heatmaps to: {RESULTS_DIR / f'{subject_tag}_eeg_amplitude_heatmaps.png'}")
            
            print("\n" + "="*70)
            print("EEG ANALYSIS COMPLETE")
            print("="*70)
            print("\nGenerated figures:")
            print("  - erp_topographic_map.png (spatial view of ERP)")
            print("  - connectivity_plv_heatmap.png (phase-locking value)")
            print("  - eeg_amplitude_heatmaps.png (time x channel amplitude)")
        else:
            print("\n⚠ 'RollerCoasterStarted' markers not found. Cannot create epochs.")
    else:
        print("\n⚠ No markers available for epoch creation")
else:
    print("\nNo EEG data available for processing")

# Plot the three auxiliary signals
fig, axes = plt.subplots(3, 1, figsize=(12, 10))
fig.suptitle('Auxiliary Physiological Signals', fontsize=16, fontweight='bold')

# Plot EDA
if len(eda_data['labels']) > 0:
    axes[0].plot(eda_data['time_stamps'], eda_data['data'][:, 0], 'b-', linewidth=0.5)
    axes[0].set_title(f'EDA/GSR - {eda_data["labels"][0]}', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Amplitude (μS)', fontsize=10)
    axes[0].grid(True, alpha=0.3)
else:
    axes[0].text(0.5, 0.5, 'No EDA data found', ha='center', va='center', transform=axes[0].transAxes)
    axes[0].set_title('EDA/GSR', fontsize=12, fontweight='bold')

# Plot PPG
if len(ppg_data['labels']) > 0:
    axes[1].plot(ppg_data['time_stamps'], ppg_data['data'][:, 0], 'r-', linewidth=0.5)
    axes[1].set_title(f'PPG - {ppg_data["labels"][0]}', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Amplitude', fontsize=10)
    axes[1].grid(True, alpha=0.3)
else:
    axes[1].text(0.5, 0.5, 'No PPG data found', ha='center', va='center', transform=axes[1].transAxes)
    axes[1].set_title('PPG', fontsize=12, fontweight='bold')

# Plot Respiration
if len(resp_data['labels']) > 0:
    axes[2].plot(resp_data['time_stamps'], resp_data['data'][:, 0], 'g-', linewidth=0.5)
    axes[2].set_title(f'Respiration - {resp_data["labels"][0]}', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Amplitude', fontsize=10)
    axes[2].set_xlabel('Time (s)', fontsize=10)
    axes[2].grid(True, alpha=0.3)
else:
    axes[2].text(0.5, 0.5, 'No Respiration data found', ha='center', va='center', transform=axes[2].transAxes)
    axes[2].set_title('Respiration', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Time (s)', fontsize=10)

plt.tight_layout()
plt.show()

# Now you can preprocess each signal type individually
# Example: Process EDA data
# if eda_data['data'].size > 0:
#     eda_processed = nk.eda_process(eda_data['data'][:, 0], sampling_rate=eda_data['sampling_rate'])

# Simulate 10 seconds of EDA Signal (recorded at 250 samples / second)
#eda_signal = nk.eda_simulate(duration=10, sampling_rate=250, scr_number=3, drift=0.01)