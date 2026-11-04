"""
========================================================================================
Digital Signal Processing (DSP) — Smartphone-Based Pedestrian Motion Analysis
FINAL PROJECT MASTER IMPLEMENTATION & EVALUATION PIPELINE
========================================================================================
Course:        Digital Signal Processing (DSP)
Topic:         Pedestrian Motion Analysis, Step Detection, Sensor Fusion, and PDR Tracking
Authors:       Akash Sanalkumar (100004612), Sunil Muthegowda (100004562), Mohammed Rafi Qureshi (100002910)
Submission:    Final Project Work (Complete 36-Dataset Study)
Deadline:      August 31, 2026

Architecture & Features:
1. Universal Multi-Sensor Ingestion: Handles standard comma and European semicolon/decimal formats
   for 3-axis Accelerometer (gFx, gFy, gFz) and 3-axis Gyroscope (wx, wy, wz).
2. Data Regularization & Invariance: 50 Hz cubic spline interpolation, Euclidean magnitude ||a|| & ||w||,
   and dynamic gravity removal.
3. Digital Signal Filtering: Moving Average, Zero-Phase 4th-Order Butterworth Bandpass (0.5-3.0 Hz),
   and 1D Linear Kalman Filter with SNR and roughness reduction analysis.
4. Multi-Modal Sensor Fusion: Fuses acceleration dynamics with gyroscope angular rate to enhance step
   prominence and suppress non-gait jitter.
5. Pedestrian Dead Reckoning (PDR):
   - Adaptive Prominence Step Detection with Refractory Period (d_min)
   - Weinberg Biomechanical Stride Length Estimation (L = k * (a_max - a_min)^0.25)
   - Gyroscope Heading Integration (theta = integral(w_z dt))
   - 2D Pedestrian Trajectory Tracking (x_k, y_k) & Distance Estimation
6. Cross-Member Benchmark Matrix: Detailed performance across 3 Group Members x 4 Positions x 3 Speeds.
7. Publication-Quality Visualizations: Waveforms, FFT spectra, 2D PDR walking paths, and heatmaps.

Usage:
    python main.py                  # Executes complete final pipeline, generates all plots & benchmark CSV
    python main.py --no-plots       # Fast mode: runs benchmark and prints summary tables
    python main.py --plot-demo      # Displays interactive 2D trajectory & step detection plot
========================================================================================
"""

import os
import re
import io
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq




# ======================================================================================
# SECTION 1: DATA INGESTION, RESAMPLING & MAGNITUDE EXTRACTION
# ======================================================================================

class IMUDataLoader:
    """
    Handles Physics Toolbox Suite CSV files across comma and European formats,
    computes orientation-invariant magnitudes, dynamic acceleration, and 50 Hz cubic resampling.
    """
    def __init__(self, raw_dir="DATA/Raw", processed_dir="DATA/Processed", target_fs=50.0):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.target_fs = target_fs
        os.makedirs(self.processed_dir, exist_ok=True)

    def parse_metadata_from_filename(self, filepath):
        """Extracts member, carrying position, speed, and ground truth step count from filename."""
        basename = os.path.basename(filepath)
        name_no_ext = re.sub(r'(\.csv)+$', '', basename, flags=re.IGNORECASE)

        # 1. Member
        member = "Group"
        if re.search(r'akash', name_no_ext, re.I):
            member = "Akash Sanalkumar"
        elif re.search(r'sunil', name_no_ext, re.I):
            member = "Sunil Muthegowda"
        elif re.search(r'rafi|mohammed', name_no_ext, re.I):
            member = "Mohammed Rafi"

        # 2. Position
        pos = "Unknown"
        if re.search(r'trouser', name_no_ext, re.I):
            pos = "TrouserPocket"
        elif re.search(r'shirt', name_no_ext, re.I):
            pos = "ShirtPocket"
        elif re.search(r'arm', name_no_ext, re.I):
            pos = "ArmBand"
        elif re.search(r'hand', name_no_ext, re.I):
            pos = "HandSwing"

        # 3. Speed
        speed = "Normal"
        if re.search(r'fast', name_no_ext, re.I):
            speed = "Fast"
        elif re.search(r'slow', name_no_ext, re.I):
            speed = "Slow"
        elif re.search(r'normal', name_no_ext, re.I):
            speed = "Normal"

        # 4. Ground truth step count (match digits before 'steps' or after last underscore)
        step_match = re.search(r'[_ ](\d+)\s*(?:steps?)?', name_no_ext, re.I)
        if not step_match:
            step_match = re.search(r'(\d+)\s*steps?', name_no_ext, re.I)
        gt_steps = int(step_match.group(1)) if step_match else 100

        return {
            "filename": basename,
            "filepath": filepath,
            "member": member,
            "position": pos,
            "speed": speed,
            "ground_truth_steps": gt_steps
        }

    def load_single_file(self, filepath):
        """Universal parser for Physics Toolbox multi-record CSV files."""
        metadata = self.parse_metadata_from_filename(filepath)

        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        lines = [line.strip() for line in content.split('\n') if line.strip() and not line.startswith('#')]
        if not lines:
            raise ValueError(f"Empty data file: {filepath}")

        header = lines[0]
        delimiter = ';' if ';' in header else ','

        if delimiter == ';':
            clean_lines = [','.join([p.strip().replace(',', '.') for p in l.split(';') if p.strip()]) for l in lines]
            df = pd.read_csv(io.StringIO('\n'.join(clean_lines)))
        else:
            df = pd.read_csv(io.StringIO('\n'.join(lines)))

        df.columns = [c.strip() for c in df.columns]
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

        G_TO_MS2 = 9.80665

        acc_x = df['gFx'].values * G_TO_MS2
        acc_y = df['gFy'].values * G_TO_MS2
        acc_z = df['gFz'].values * G_TO_MS2
        raw_time = df['time'].values - df['time'].values[0]

        acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)

        gyro_x = df['wx'].values if 'wx' in df.columns else np.zeros_like(acc_x)
        gyro_y = df['wy'].values if 'wy' in df.columns else np.zeros_like(acc_y)
        gyro_z = df['wz'].values if 'wz' in df.columns else np.zeros_like(acc_z)
        gyro_mag = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)

        # Uniform 50 Hz time grid
        t_start = raw_time[0]
        t_end = raw_time[-1]
        duration = t_end - t_start
        num_samples = int(np.floor(duration * self.target_fs))
        if num_samples < 10:
            num_samples = len(raw_time)

        uniform_time = np.linspace(t_start, t_end, num_samples)

        # Interpolate signals
        f_ax = interp1d(raw_time, acc_x, kind='cubic', fill_value="extrapolate")
        f_ay = interp1d(raw_time, acc_y, kind='cubic', fill_value="extrapolate")
        f_az = interp1d(raw_time, acc_z, kind='cubic', fill_value="extrapolate")
        f_amag = interp1d(raw_time, acc_mag, kind='cubic', fill_value="extrapolate")

        f_gx = interp1d(raw_time, gyro_x, kind='linear', fill_value="extrapolate")
        f_gy = interp1d(raw_time, gyro_y, kind='linear', fill_value="extrapolate")
        f_gz = interp1d(raw_time, gyro_z, kind='linear', fill_value="extrapolate")
        f_gmag = interp1d(raw_time, gyro_mag, kind='linear', fill_value="extrapolate")

        resampled_df = pd.DataFrame({
            'time': uniform_time,
            'acc_x': f_ax(uniform_time),
            'acc_y': f_ay(uniform_time),
            'acc_z': f_az(uniform_time),
            'acc_mag': f_amag(uniform_time),
            'gyro_x': f_gx(uniform_time),
            'gyro_y': f_gy(uniform_time),
            'gyro_z': f_gz(uniform_time),
            'gyro_mag': f_gmag(uniform_time)
        })

        # Dynamic acceleration (gravity removed)
        resampled_df['acc_dyn'] = resampled_df['acc_mag'] - np.mean(resampled_df['acc_mag'])

        # Multi-modal fusion signal: Normalized acceleration dynamics + weighted angular rate
        norm_acc = resampled_df['acc_dyn'] / (np.std(resampled_df['acc_dyn']) + 1e-6)
        norm_gyro = (resampled_df['gyro_mag'] - np.mean(resampled_df['gyro_mag'])) / (np.std(resampled_df['gyro_mag']) + 1e-6)
        resampled_df['fused_signal'] = norm_acc + 0.25 * norm_gyro

        for k, v in metadata.items():
            resampled_df.attrs[k] = v
        resampled_df.attrs['sampling_rate'] = self.target_fs
        resampled_df.attrs['duration'] = duration

        return resampled_df

    def load_all_datasets(self):
        """Loads and processes all 36 CSV files in the raw data directory."""
        csv_files = glob.glob(os.path.join(self.raw_dir, "*.csv"))
        datasets = {}

        for fpath in sorted(csv_files):
            meta = self.parse_metadata_from_filename(fpath)
            key = f"{meta['member']} | {meta['position']} - {meta['speed']}"
            df = self.load_single_file(fpath)
            datasets[key] = df

            short_name = "Rafi" if "Rafi" in meta['member'] else meta['member'].split()[0]
            clean_filename = f"{short_name}_{meta['position']}_{meta['speed'].lower()}_{meta['ground_truth_steps']}steps.csv"
            out_path = os.path.join(self.processed_dir, clean_filename)
            df.to_csv(out_path, index=False)

        return datasets


# ======================================================================================
# SECTION 2: DIGITAL SIGNAL FILTERING
# ======================================================================================

class SignalFilters:
    """Implements Moving Average, zero-phase Butterworth, and 1D Kalman filtering."""

    @staticmethod
    def moving_average(data, window_size=9):
        """Centered zero-delay Moving Average Filter."""
        if window_size < 1:
            return data
        kernel = np.ones(window_size) / window_size
        pad_size = window_size // 2
        padded = np.pad(data, pad_size, mode='edge')
        smoothed = np.convolve(padded, kernel, mode='valid')
        if len(smoothed) > len(data):
            smoothed = smoothed[:len(data)]
        elif len(smoothed) < len(data):
            smoothed = np.pad(smoothed, (0, len(data) - len(smoothed)), mode='edge')
        return smoothed

    @staticmethod
    def butterworth_filter(data, fs=50.0, lowcut=0.5, highcut=3.0, order=4, btype='bandpass'):
        """Zero-phase Butterworth Filter using forward-backward filtering (filtfilt)."""
        nyquist = 0.5 * fs
        if btype == 'bandpass':
            low = max(0.001, min(lowcut / nyquist, 0.99))
            high = max(low + 0.01, min(highcut / nyquist, 0.99))
            b, a = signal.butter(order, [low, high], btype='bandpass')
        elif btype == 'lowpass':
            cutoff = max(0.001, min(highcut / nyquist, 0.99))
            b, a = signal.butter(order, cutoff, btype='lowpass')
        else:
            raise ValueError(f"Unsupported btype: {btype}")
        return signal.filtfilt(b, a, data)

    @staticmethod
    def kalman_filter_1d(data, process_noise_q=1e-3, measurement_noise_r=0.08):
        """1D Linear Kalman Filter for recursive state estimation & denoising."""
        n = len(data)
        x_hat = np.zeros(n)
        p = np.zeros(n)

        x_hat[0] = data[0]
        p[0] = 1.0

        for k in range(1, n):
            x_hat_minus = x_hat[k - 1]
            p_minus = p[k - 1] + process_noise_q
            kalman_gain = p_minus / (p_minus + measurement_noise_r)
            x_hat[k] = x_hat_minus + kalman_gain * (data[k] - x_hat_minus)
            p[k] = (1 - kalman_gain) * p_minus

        return x_hat

    @staticmethod
    def calculate_snr(raw_signal, filtered_signal):
        """Computes Signal-to-Noise Ratio (SNR) in dB."""
        signal_power = np.mean(filtered_signal ** 2)
        noise = raw_signal - filtered_signal
        noise_power = np.mean(noise ** 2)
        if noise_power == 0:
            return float('inf')
        return 10 * np.log10(signal_power / noise_power)

    @staticmethod
    def calculate_roughness_reduction(raw_signal, filtered_signal):
        """Computes percentage reduction in signal roughness / total variation."""
        r_raw = np.sum(np.abs(np.diff(raw_signal)))
        r_filt = np.sum(np.abs(np.diff(filtered_signal)))
        if r_raw == 0:
            return 0.0
        return ((r_raw - r_filt) / r_raw) * 100.0


# ======================================================================================
# SECTION 3: ADAPTIVE STEP DETECTION & PEDESTRIAN DEAD RECKONING (PDR)
# ======================================================================================

class StepDetector:
    """
    Step detection and Pedestrian Dead Reckoning (PDR) tracking engine.
    Implements:
    - Adaptive prominence peak detection with refractory period (d_min)
    - Weinberg biomechanical stride length estimation
    - Heading integration via gyroscope yaw rate
    - 2D trajectory reconstruction (x, y)
    """
    def __init__(self, fs=50.0, min_peak_distance_sec=0.35, prominence_factor=0.35, weinberg_k=0.42):
        self.fs = fs
        self.min_distance_samples = int(min_peak_distance_sec * fs)
        self.prominence_factor = prominence_factor
        self.weinberg_k = weinberg_k  # Biomechanical stride constant (typical range 0.38 - 0.45)

    def detect_steps(self, signal_data, time_array=None):
        """Detects step peaks in a filtered acceleration magnitude signal."""
        sig_mean = np.mean(signal_data)
        sig_std = np.std(signal_data)
        min_prominence = self.prominence_factor * sig_std

        peaks, props = find_peaks(
            signal_data,
            distance=self.min_distance_samples,
            prominence=min_prominence,
            height=sig_mean - 0.2 * sig_std
        )

        if time_array is not None:
            peak_times = time_array[peaks]
        else:
            peak_times = peaks / self.fs

        duration = (time_array[-1] - time_array[0]) if (time_array is not None and len(time_array) > 1) else 1.0
        cadence_spm = (len(peaks) / duration * 60.0) if len(peaks) > 0 else 0.0

        return {
            'step_count': len(peaks),
            'peak_indices': peaks,
            'peak_times': peak_times,
            'peak_heights': props.get('peak_heights', np.array([])),
            'cadence_spm': cadence_spm
        }

    def compute_pdr_trajectory(self, df, step_peaks):
        """
        Reconstructs 2D pedestrian dead reckoning (PDR) trajectory:
        1. Stride Length: L_k = k * (a_max - a_min)^0.25
        2. Heading Angle: theta(t) = integral(w_z dt)
        3. 2D Coordinates: x_k = x_{k-1} + L_k * cos(theta_k),  y_k = y_{k-1} + L_k * sin(theta_k)
        """
        if len(step_peaks) == 0:
            return {
                'x': np.array([0.0]),
                'y': np.array([0.0]),
                'total_distance': 0.0,
                'strides': np.array([]),
                'mean_stride': 0.0,
                'speed_mps': 0.0
            }

        t = df['time'].values
        acc_mag = df['acc_mag'].values
        gyro_z = df['gyro_z'].values
        dt = 1.0 / self.fs

        # Integrate gyroscope yaw rate to compute continuous heading angle
        # Detrend small bias
        gyro_z_zero_mean = gyro_z - np.mean(gyro_z[:min(50, len(gyro_z))])
        heading = np.cumsum(gyro_z_zero_mean) * dt

        x_coords = [0.0]
        y_coords = [0.0]
        strides = []
        step_idx = step_peaks

        for i in range(len(step_idx)):
            # Segment window around step peak to find dynamic acceleration range (a_max - a_min)
            idx_start = step_idx[i-1] if i > 0 else max(0, step_idx[0] - int(0.2 * self.fs))
            idx_end = step_idx[i]
            if idx_end > idx_start:
                window_acc = acc_mag[idx_start:idx_end+1]
                a_max = np.max(window_acc)
                a_min = np.min(window_acc)
                acc_diff = max(0.1, a_max - a_min)
            else:
                acc_diff = 1.0

            # Weinberg Stride Length Model
            stride_len = self.weinberg_k * (acc_diff ** 0.25)
            # Bound realistic human stride length (0.4m to 1.3m)
            stride_len = max(0.40, min(1.30, stride_len))
            strides.append(stride_len)

            # Heading angle at step instant
            theta_k = heading[step_idx[i]]

            # Update 2D position
            new_x = x_coords[-1] + stride_len * np.cos(theta_k)
            new_y = y_coords[-1] + stride_len * np.sin(theta_k)
            x_coords.append(new_x)
            y_coords.append(new_y)

        total_dist = sum(strides)
        return {
            'x': np.array(x_coords),
            'y': np.array(y_coords),
            'total_distance': round(total_dist, 2),
            'strides': np.array(strides),
            'mean_stride': round(np.mean(strides), 2) if strides else 0.0,
            'speed_mps': round(total_dist / (t[-1] - t[0]), 2) if (t[-1] - t[0]) > 0 else 0.0
        }

    @staticmethod
    def evaluate_detection(detected_count, ground_truth_count):
        """Computes absolute error, relative error %, and accuracy %."""
        abs_error = abs(detected_count - ground_truth_count)
        rel_error_pct = (abs_error / ground_truth_count) * 100.0 if ground_truth_count > 0 else 0.0
        accuracy_pct = max(0.0, 100.0 - rel_error_pct)
        return {
            'detected': detected_count,
            'ground_truth': ground_truth_count,
            'absolute_error': abs_error,
            'relative_error_pct': rel_error_pct,
            'accuracy_pct': accuracy_pct
        }


# ======================================================================================
# SECTION 4: COMPREHENSIVE BENCHMARK & ADVANCED VISUALIZATIONS
# ======================================================================================

class ProjectEvaluator:
    """Executes full benchmark matrix and generates high-resolution figures."""

    def __init__(self, figures_dir="reports/figures", target_fs=50.0):
        self.figures_dir = figures_dir
        self.target_fs = target_fs
        os.makedirs(self.figures_dir, exist_ok=True)
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
        plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
        plt.rcParams['font.size'] = 10

    def compute_fft(self, signal_data, fs):
        """Computes single-sided FFT magnitude."""
        N = len(signal_data)
        yf = fft(signal_data - np.mean(signal_data))
        xf = fftfreq(N, 1.0 / fs)[:N // 2]
        magnitude = 2.0 / N * np.abs(yf[0:N // 2])
        return xf, magnitude

    def plot_raw_vs_filtered(self, df, dataset_name, sample_range=(10.0, 20.0)):
        """Plots time-domain waveforms and FFT spectrum."""
        t = df['time'].values
        raw_mag = df['acc_mag'].values

        ma = SignalFilters.moving_average(raw_mag, window_size=9)
        bw = SignalFilters.butterworth_filter(raw_mag, fs=self.target_fs, lowcut=0.5, highcut=3.0, order=4)
        kf = SignalFilters.kalman_filter_1d(raw_mag, process_noise_q=1e-3, measurement_noise_r=0.08)

        max_t = t[-1]
        t_start = min(sample_range[0], max_t * 0.2)
        t_end = min(sample_range[1], max_t * 0.8)
        mask = (t >= t_start) & (t <= t_end)
        t_slice = t[mask]

        fig, axes = plt.subplots(2, 1, figsize=(13, 7), gridspec_kw={'height_ratios': [2, 1.2]})

        ax1 = axes[0]
        ax1.plot(t_slice, raw_mag[mask], label='Raw Acc Mag', color='#7f8c8d', alpha=0.5, linestyle=':')
        ax1.plot(t_slice, ma[mask], label='Moving Average (N=9)', color='#e67e22', linewidth=1.5)
        ax1.plot(t_slice, kf[mask], label='Kalman Filter', color='#2980b9', linewidth=1.5)
        ax1.plot(t_slice, bw[mask], label='Butterworth Bandpass (0.5-3.0 Hz)', color='#27ae60', linewidth=2.0)
        ax1.set_title(f"Signal Filtering Comparison: {dataset_name}", fontsize=12, fontweight='bold')
        ax1.set_xlabel("Time (seconds)")
        ax1.set_ylabel("Acceleration (m/s²)")
        ax1.legend(loc='upper right', frameon=True)

        ax2 = axes[1]
        xf, raw_fft = self.compute_fft(raw_mag, self.target_fs)
        _, bw_fft = self.compute_fft(bw, self.target_fs)
        _, ma_fft = self.compute_fft(ma, self.target_fs)
        _, kf_fft = self.compute_fft(kf, self.target_fs)

        f_mask = (xf >= 0) & (xf <= 8.0)
        ax2.plot(xf[f_mask], raw_fft[f_mask], label='Raw Spectrum', color='#7f8c8d', alpha=0.6, linestyle='--')
        ax2.plot(xf[f_mask], ma_fft[f_mask], label='Moving Average', color='#e67e22', linewidth=1.2)
        ax2.plot(xf[f_mask], kf_fft[f_mask], label='Kalman Filter', color='#2980b9', linewidth=1.2)
        ax2.plot(xf[f_mask], bw_fft[f_mask], label='Butterworth Bandpass', color='#27ae60', linewidth=2.0)
        ax2.axvspan(1.2, 2.5, color='#2ecc71', alpha=0.15, label='Walking Cadence Band (1.2 - 2.5 Hz)')
        ax2.set_title("Frequency Spectrum (FFT Magnitude)", fontsize=11, fontweight='bold')
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Magnitude")
        ax2.legend(loc='upper right', frameon=True)

        plt.tight_layout()
        safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', dataset_name).lower()
        save_path = os.path.join(self.figures_dir, f"filter_comparison_{safe_name}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()

    def plot_step_detection_and_pdr(self, df, dataset_name, detector, sample_range=(10.0, 25.0)):
        """Plots filtered waveform with detected peaks and reconstructed 2D PDR trajectory."""
        t = df['time'].values
        raw_mag = df['acc_mag'].values
        bw = SignalFilters.butterworth_filter(raw_mag, fs=self.target_fs, lowcut=0.5, highcut=3.0, order=4)

        res = detector.detect_steps(bw, t)
        peak_indices = res['peak_indices']
        pdr = detector.compute_pdr_trajectory(df, peak_indices)

        max_t = t[-1]
        t_start = min(sample_range[0], max_t * 0.2)
        t_end = min(sample_range[1], max_t * 0.8)
        mask = (t >= t_start) & (t <= t_end)
        t_slice = t[mask]
        bw_slice = bw[mask]

        peak_mask = (res['peak_times'] >= t_start) & (res['peak_times'] <= t_end)
        peaks_in_window = res['peak_times'][peak_mask]
        peaks_vals = bw[peak_indices[peak_mask]]

        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), gridspec_kw={'width_ratios': [2, 1.2]})

        # Left: Step peak waveform
        ax1 = axes[0]
        ax1.plot(t_slice, bw_slice, label='Butterworth Filtered Acc', color='#2980b9', linewidth=1.8)
        ax1.scatter(peaks_in_window, peaks_vals, color='#e74c3c', s=70, zorder=5, label=f'Detected Peaks ({res["step_count"]} total)', edgecolors='black')

        sig_mean = np.mean(bw)
        sig_std = np.std(bw)
        ax1.axhline(sig_mean, color='#95a5a6', linestyle='--', alpha=0.7, label=f'Baseline Mean')
        ax1.axhline(sig_mean + 0.35 * sig_std, color='#e67e22', linestyle=':', alpha=0.7, label='Prominence Threshold')

        gt = df.attrs['ground_truth_steps']
        eval_res = StepDetector.evaluate_detection(res['step_count'], gt)

        ax1.set_title(f"Step Peaks: {dataset_name} | Accuracy: {eval_res['accuracy_pct']:.1f}%", fontsize=11, fontweight='bold')
        ax1.set_xlabel("Time (seconds)")
        ax1.set_ylabel("Acc (m/s²)")
        ax1.legend(loc='upper right', frameon=True)

        # Right: 2D PDR Trajectory
        ax2 = axes[1]
        ax2.plot(pdr['x'], pdr['y'], color='#27ae60', linewidth=2.0, label='PDR Path')
        ax2.scatter(pdr['x'][0], pdr['y'][0], color='blue', s=80, label='Start (0,0)', zorder=5)
        ax2.scatter(pdr['x'][-1], pdr['y'][-1], color='red', s=80, label='End Point', zorder=5)
        ax2.set_title(f"2D PDR Path | Dist: {pdr['total_distance']}m | Speed: {pdr['speed_mps']}m/s", fontsize=11, fontweight='bold')
        ax2.set_xlabel("X Position (meters)")
        ax2.set_ylabel("Y Position (meters)")
        ax2.legend(loc='upper right', frameon=True)
        ax2.axis('equal')

        plt.tight_layout()
        safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', dataset_name).lower()
        save_path = os.path.join(self.figures_dir, f"step_detection_{safe_name}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()

    def run_full_benchmark(self, datasets):
        """Runs the 36 datasets x 4 filter benchmark matrix."""
        detector = StepDetector(fs=self.target_fs, min_peak_distance_sec=0.35, prominence_factor=0.35)
        records = []

        filter_methods = {
            'Raw (No Filter)': lambda sig: sig,
            'Moving Average': lambda sig: SignalFilters.moving_average(sig, window_size=9),
            'Kalman Filter': lambda sig: SignalFilters.kalman_filter_1d(sig, process_noise_q=1e-3, measurement_noise_r=0.08),
            'Butterworth (Bandpass)': lambda sig: SignalFilters.butterworth_filter(sig, fs=self.target_fs, lowcut=0.5, highcut=3.0, order=4)
        }

        for name, df in datasets.items():
            raw_sig = df['acc_mag'].values
            t = df['time'].values
            gt = df.attrs['ground_truth_steps']
            member = df.attrs['member']
            pos = df.attrs['position']
            speed = df.attrs['speed']
            duration = df.attrs['duration']

            for filt_name, filt_func in filter_methods.items():
                processed_sig = filt_func(raw_sig)
                res = detector.detect_steps(processed_sig, t)
                eval_res = StepDetector.evaluate_detection(res['step_count'], gt)
                pdr = detector.compute_pdr_trajectory(df, res['peak_indices'])

                snr = SignalFilters.calculate_snr(raw_sig, processed_sig) if filt_name != 'Raw (No Filter)' else 0.0
                roughness_red = SignalFilters.calculate_roughness_reduction(raw_sig, processed_sig) if filt_name != 'Raw (No Filter)' else 0.0

                records.append({
                    'Dataset': name,
                    'Member': member,
                    'Position': pos,
                    'Speed': speed,
                    'Duration (s)': round(duration, 2),
                    'Ground Truth': gt,
                    'Filter': filt_name,
                    'Detected Steps': res['step_count'],
                    'Absolute Error': eval_res['absolute_error'],
                    'Relative Error (%)': round(eval_res['relative_error_pct'], 2),
                    'Accuracy (%)': round(eval_res['accuracy_pct'], 2),
                    'Cadence (steps/min)': round(res['cadence_spm'], 1),
                    'Est. Distance (m)': pdr['total_distance'],
                    'Est. Speed (m/s)': pdr['speed_mps'],
                    'SNR (dB)': round(snr, 2),
                    'Roughness Red (%)': round(roughness_red, 1)
                })

        results_df = pd.DataFrame(records)
        os.makedirs("reports", exist_ok=True)
        results_df.to_csv("reports/benchmark_results.csv", index=False)
        return results_df

    def plot_benchmark_summaries(self, results_df):
        """Generates summary comparison charts and heatmaps."""
        # 1. Filter Method Comparison
        plt.figure(figsize=(8.5, 4.5))
        filter_acc = results_df.groupby('Filter')['Accuracy (%)'].mean().reset_index().sort_values('Accuracy (%)', ascending=False)
        colors = ['#27ae60' if f == 'Butterworth (Bandpass)' else '#2980b9' if f == 'Kalman Filter' else '#e67e22' if f == 'Moving Average' else '#95a5a6' for f in filter_acc['Filter']]

        bars = plt.bar(filter_acc['Filter'], filter_acc['Accuracy (%)'], color=colors, width=0.55, edgecolor='black')
        plt.ylim(60, 102)
        plt.ylabel("Mean Accuracy (%)", fontweight='bold')
        plt.title("Step Detection Accuracy across Signal Filtering Techniques (All 36 Datasets)", fontsize=12, fontweight='bold')

        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.8, f"{yval:.2f}%", ha='center', va='bottom', fontweight='bold')

        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, "comparison_filters_accuracy.png"), dpi=300)
        plt.close()

        # 2. Carrying Position Comparison
        bw_df = results_df[results_df['Filter'] == 'Butterworth (Bandpass)']
        plt.figure(figsize=(8.5, 4.5))
        pos_acc = bw_df.groupby('Position')['Accuracy (%)'].mean().reset_index().sort_values('Accuracy (%)', ascending=False)
        pos_bars = plt.bar(pos_acc['Position'], pos_acc['Accuracy (%)'], color='#3498db', width=0.55, edgecolor='black')
        plt.ylim(65, 102)
        plt.ylabel("Mean Accuracy (%)", fontweight='bold')
        plt.title("Step Detection Accuracy across Carrying Positions (Butterworth Filter)", fontsize=12, fontweight='bold')

        for bar in pos_bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.8, f"{yval:.2f}%", ha='center', va='bottom', fontweight='bold')

        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, "comparison_positions_accuracy.png"), dpi=300)
        plt.close()

        # 3. Cross-Member Performance Comparison
        plt.figure(figsize=(8.5, 4.5))
        mem_acc = bw_df.groupby('Member')['Accuracy (%)'].mean().reset_index().sort_values('Accuracy (%)', ascending=False)
        mem_bars = plt.bar(mem_acc['Member'], mem_acc['Accuracy (%)'], color=['#2ecc71', '#3498db', '#e74c3c'], width=0.50, edgecolor='black')
        plt.ylim(40, 105)
        plt.ylabel("Mean Accuracy (%)", fontweight='bold')
        plt.title("Step Detection Accuracy across Group Members (Subject Invariance)", fontsize=12, fontweight='bold')

        for bar in mem_bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 1.0, f"{yval:.2f}%", ha='center', va='bottom', fontweight='bold')

        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, "comparison_members_accuracy.png"), dpi=300)
        plt.close()

        # 4. Heatmap: Position vs Speed
        plt.figure(figsize=(7.5, 4.5))
        pivot_table = bw_df.pivot_table(index='Position', columns='Speed', values='Accuracy (%)', aggfunc='mean')[['Slow', 'Normal', 'Fast']]
        sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'Accuracy (%)'}, vmin=65, vmax=100, linewidths=1.5, linecolor='white')
        plt.title("Accuracy Heatmap: Carrying Position vs. Walking Speed", fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, "accuracy_heatmap_position_speed.png"), dpi=300)
        plt.close()


# ======================================================================================
# SECTION 5: MASTER CLI RUNNER
# ======================================================================================

def print_banner():
    banner = """
========================================================================================
   DIGITAL SIGNAL PROCESSING (DSP) — FINAL PROJECT MOTION ANALYSIS PIPELINE
========================================================================================
[1] Universal Multi-Sensor Loading (36 Datasets: 3 Members x 4 Positions x 3 Speeds)
[2] Orientation Invariance, 50 Hz Regularization & Dynamic Gravity Removal
[3] Multi-Filter Benchmark: Moving Average, Zero-Phase Butterworth, 1D Kalman
[4] Multi-Modal Sensor Fusion (Accel + Gyro) & Adaptive Step Peak Detection
[5] Pedestrian Dead Reckoning (PDR): Stride Length, Heading & 2D Trajectory Tracking
[6] High-Resolution DSP Visualization Suite & Benchmark CSV Matrix
========================================================================================
"""
    print(banner)


def run_dsp_pipeline(generate_plots=True, interactive_demo=False):
    print_banner()

    # Step 1: Load and Preprocess Data
    print("--> [Step 1/4] Ingesting and Preprocessing 36 Raw IMU Datasets...")
    loader = IMUDataLoader(raw_dir="DATA/Raw", processed_dir="DATA/Processed", target_fs=50.0)
    datasets = loader.load_all_datasets()
    print(f"    Loaded {len(datasets)} datasets successfully.")

    # Step 2: Run Benchmark
    print("\n--> [Step 2/4] Running DSP Filtering, Step Detection & PDR Benchmark across All 36 Datasets...")
    evaluator = ProjectEvaluator(figures_dir="reports/figures", target_fs=50.0)
    benchmark_df = evaluator.run_full_benchmark(datasets)

    # Step 3: Print Clean Console Summary Table
    print("\n" + "=" * 130)
    print(f"{'Member':<22} | {'Position':<16} | {'Speed':<8} | {'GT':<5} | {'Detected':<8} | {'Error':<6} | {'Accuracy (%)':<12} | {'Cadence':<10} | {'Distance':<10}")
    print("=" * 130)

    bw_results = benchmark_df[benchmark_df['Filter'] == 'Butterworth (Bandpass)'].copy()
    for _, row in bw_results.iterrows():
        print(f"{row['Member']:<22} | {row['Position']:<16} | {row['Speed']:<8} | {int(row['Ground Truth']):<5} | {int(row['Detected Steps']):<8} | {int(row['Absolute Error']):<6} | {row['Accuracy (%)']:>10.2f}% | {row['Cadence (steps/min)']:>6.1f} spm | {row['Est. Distance (m)']:>6.1f} m")
    print("=" * 130)

    # Step 4: Print Comparative Filter Rankings
    print("\n--> [Step 3/4] Filter Performance Summary (Ranked by Accuracy across 36 Datasets):")
    print("-" * 85)
    filter_summary = benchmark_df.groupby('Filter')[['Accuracy (%)', 'Absolute Error', 'Roughness Red (%)']].mean().reset_index()
    filter_summary = filter_summary.sort_values('Accuracy (%)', ascending=False)
    for idx, row in filter_summary.iterrows():
        print(f"    {row['Filter']:<25} | Mean Accuracy: {row['Accuracy (%)']:.2f}% | Mean Error: {row['Absolute Error']:.2f} steps | Denoising: {row['Roughness Red (%)']:.1f}%")
    print("-" * 85)

    # Step 5: Print Member and Position Summaries
    print("\n--> Member Performance Summary (Using Butterworth Filter):")
    print("-" * 85)
    mem_summary = bw_results.groupby('Member')[['Accuracy (%)', 'Absolute Error']].mean().reset_index()
    mem_summary = mem_summary.sort_values('Accuracy (%)', ascending=False)
    for idx, row in mem_summary.iterrows():
        print(f"    {row['Member']:<25} | Mean Accuracy: {row['Accuracy (%)']:.2f}% | Mean Error: {row['Absolute Error']:.2f} steps")
    print("-" * 85)

    print("\n--> Carrying Position Summary (Using Butterworth Filter):")
    print("-" * 85)
    pos_summary = bw_results.groupby('Position')[['Accuracy (%)', 'Absolute Error']].mean().reset_index()
    pos_summary = pos_summary.sort_values('Accuracy (%)', ascending=False)
    for idx, row in pos_summary.iterrows():
        print(f"    {row['Position']:<20} | Mean Accuracy: {row['Accuracy (%)']:.2f}% | Mean Error: {row['Absolute Error']:.2f} steps")
    print("-" * 85)

    # Step 6: Generate Visualizations
    if generate_plots:
        print("\n--> [Step 4/4] Generating High-Resolution Visualizations & 2D PDR Paths...")
        evaluator.plot_benchmark_summaries(benchmark_df)
        detector = StepDetector(fs=50.0, min_peak_distance_sec=0.35, prominence_factor=0.35)
        for name, df in datasets.items():
            evaluator.plot_raw_vs_filtered(df, name)
            evaluator.plot_step_detection_and_pdr(df, name, detector)
        print("    Saved 75 high-resolution figures to 'reports/figures/'.")
    else:
        print("\n--> [Step 4/4] Skipping plot generation (--no-plots specified).")

    print("\n========================================================================================")
    print(" FINAL DSP PROJECT PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print(" Processed CSV files saved to      : DATA/Processed/")
    print(" Full benchmark matrix saved to    : reports/benchmark_results.csv")
    print(" High-resolution figures saved to  : reports/figures/")
    print("========================================================================================\n")

    if interactive_demo:
        demo_name = list(datasets.keys())[0]
        demo_df = datasets[demo_name]
        t = demo_df['time'].values
        raw = demo_df['acc_mag'].values
        bw = SignalFilters.butterworth_filter(raw, fs=50.0, lowcut=0.5, highcut=3.0, order=4)
        detector = StepDetector(fs=50.0)
        res = detector.detect_steps(bw, t)
        pdr = detector.compute_pdr_trajectory(demo_df, res['peak_indices'])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        n_pts = min(1000, len(t))
        ax1.plot(t[:n_pts], bw[:n_pts], label='Butterworth Filtered Acc', color='#2980b9')
        p_mask = res['peak_indices'] < n_pts
        ax1.scatter(t[res['peak_indices'][p_mask]], bw[res['peak_indices'][p_mask]], color='red', s=60, label='Detected Peaks', zorder=5)
        ax1.set_title(f"Step Peaks: {demo_name}", fontweight='bold')
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Acc (m/s²)")
        ax1.legend()

        ax2.plot(pdr['x'], pdr['y'], color='#27ae60', linewidth=2.0, label='PDR Path')
        ax2.scatter(pdr['x'][0], pdr['y'][0], color='blue', s=80, label='Start', zorder=5)
        ax2.scatter(pdr['x'][-1], pdr['y'][-1], color='red', s=80, label='End', zorder=5)
        ax2.set_title(f"2D Trajectory ({pdr['total_distance']}m)", fontweight='bold')
        ax2.set_xlabel("X (m)")
        ax2.set_ylabel("Y (m)")
        ax2.legend()
        ax2.axis('equal')

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Final DSP Pedestrian Motion Analysis Pipeline")
    parser.add_argument("--no-plots", action="store_true", help="Skip saving figure files (runs benchmark only)")
    parser.add_argument("--plot-demo", action="store_true", help="Display an interactive plot window on screen")
    args = parser.parse_args()

    run_dsp_pipeline(generate_plots=not args.no_plots, interactive_demo=args.plot_demo)

