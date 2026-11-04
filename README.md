# DSP — Smartphone-Based Pedestrian Motion Analysis

Digital Signal Processing (DSP) project for smartphone-based pedestrian step detection, sensor fusion, motion analysis, and 2D Pedestrian Dead Reckoning (PDR) tracking.

---

## Authors & Group Members

| Name | Matriculation Number |
| :--- | :--- |
| **Akash Sanalkumar** | 100004612 |
| **Sunil Muthegowda** | 100004562 |
| **Mohammed Rafi Qureshi** | 100002910 |

**Course:** Digital Signal Processing (DSP)  
**Topic:** Smartphone-Based Pedestrian Step Detection & Motion Analysis  

---

## Overview & Key Features

1. **Universal Multi-Sensor Ingestion**: Parses Physics Toolbox Suite CSV files across comma and European semicolon/decimal formats for 3-axis Accelerometer ($gFx, gFy, gFz$) and 3-axis Gyroscope ($wx, wy, wz$).
2. **Data Regularization & Invariance**: 50 Hz cubic spline interpolation, Euclidean magnitude extraction ($\|a\|$ & $\|\omega\|$), and dynamic gravity removal.
3. **Digital Signal Filtering**:
   - Moving Average Filter
   - Zero-Phase 4th-Order Butterworth Bandpass Filter (0.5 – 3.0 Hz)
   - 1D Linear Kalman Filter
   - SNR improvement and roughness reduction analysis
4. **Multi-Modal Sensor Fusion**: Fuses acceleration dynamics with gyroscope angular rate to enhance step prominence and suppress non-gait jitter.
5. **Pedestrian Dead Reckoning (PDR)**:
   - Adaptive prominence peak detection with refractory period lockout
   - Weinberg biomechanical stride length estimation: $L = k \cdot (a_{\max} - a_{\min})^{0.25}$
   - Gyroscope heading integration: $\theta = \int \omega_z \, dt$
   - 2D trajectory reconstruction $(x_k, y_k)$ and total distance estimation
6. **Cross-Member Benchmark Matrix**: Evaluates performance across 3 Subjects $\times$ 4 Carrying Positions $\times$ 3 Walking Speeds (36 datasets total).
7. **Comprehensive Visualizations**: Generates waveform plots, FFT frequency spectra, 2D walking trajectories, and summary evaluation heatmaps.

---

## Project Structure

```text
DSP-Smartphone-Pedestrian-Motion-Analysis/
├── main.py                          # Master DSP pipeline, filtering benchmark & figure generator
├── README.md                        # Project documentation and setup guide
├── README.txt                       # Submission info & structure description
│
├── DATA/
│   ├── Raw/                         # 36 Standardized IMU CSV recordings (3 Subjects)
│   └── Processed/                   # 50 Hz resampled & magnitude-extracted CSVs
│
└── reports/
    ├── Final_Project_Report.pdf     # Final Academic Submission Report (PDF)
    ├── benchmark_results.csv        # Numerical benchmark evaluation matrix (36 datasets)
    └── figures/                     # High-resolution figures (waveforms, FFT, 2D PDR paths)
```

---

## Requirements & Installation

Python 3.8+ is recommended. Install required dependencies:

```bash
pip install numpy pandas matplotlib seaborn scipy
```

---

## Usage

Run the master DSP pipeline:

```bash
# Execute complete pipeline, generate all figures and benchmark CSV
python main.py

# Fast mode: run benchmark analysis and print summary tables without regenerating plots
python main.py --no-plots

# Interactive demo: display 2D trajectory and step detection plots
python main.py --plot-demo
```
