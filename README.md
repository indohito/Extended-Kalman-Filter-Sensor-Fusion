# Extended Kalman Filter Sensor Fusion

## Project Overview

This project implements an Extended Kalman Filter (EKF) for vehicle position tracking using sensor fusion of LiDAR and Radar data. The implementation follows the research paper:

**"Extended Kalman Filter (EKF) Design for Vehicle Position Tracking Using Reliability Function of Radar and Lidar"** by Kim & Park, Sensors 2020.

### Key Features

- **Sensor Fusion**: Combines LiDAR and Radar measurements to track a lead vehicle's position, velocity, and heading
- **Reliability Functions**: Uses sigmoid-based reliability functions to weight sensor measurements based on distance
- **Real-World Data**: Tested on the NuScenes 1.0 mini dataset with real sensor point clouds
- **State Estimation**: Tracks 4D state vector: [x, y, V, θ] (position, velocity, heading)

### Project Structure

```
Extended-Kalman-Filter-Sensor-Fusion/
├── scripts/
│   ├── ekf_script.py              # EKF implementation class
│   ├── test_nuScenes.py           # Main script to run EKF on NuScenes data
│   ├── nuScenes_visualizsation.ipynb  # Visualization notebook
│   └── data/                      # NuScenes dataset directory (download separately)
│       ├── v1.0-mini/            # Metadata JSON files
│       ├── samples/              # Sample sensor data
│       ├── sweeps/               # Sweep sensor data
│       └── maps/                 # Map data
├── figures/                       # Output figures and plots
├── requirements.txt               # Python package dependencies
└── README.md                      # This file
```

## Setup Instructions

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

### Installation

1. **Clone or download this repository**

2. **Install required Python packages**:
   
   Using the requirements file (recommended):
   ```bash
   pip install -r requirements.txt
   ```
   
   Or install manually:
   ```bash
   pip install numpy matplotlib pyquaternion nuscenes-devkit
   ```

   Or if you prefer using a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

### Downloading the NuScenes 1.0 Mini Dataset

**Important**: The NuScenes dataset must be downloaded separately and placed in the correct directory.

1. **Register and Download**:
   - Visit the [NuScenes website](https://www.nuscenes.org/)
   - Create an account and accept the terms of use
   - Navigate to the [download page](https://www.nuscenes.org/download)
   - Download **NuScenes 1.0 mini** dataset

2. **Dataset Contents**:
   The NuScenes 1.0 mini dataset includes:
   - Metadata JSON files (in `v1.0-mini/` folder)
   - Sensor data samples (in `samples/` folder)
   - Sensor sweeps (in `sweeps/` folder)
   - Map data (in `maps/` folder)

3. **Place the Data**:
   After downloading, extract and place the dataset in the following location:
   ```
   Extended-Kalman-Filter-Sensor-Fusion/scripts/data/
   ```

   The final structure should look like:
   ```
   scripts/
   └── data/
       ├── v1.0-mini/
       │   ├── attribute.json
       │   ├── calibrated_sensor.json
       │   ├── category.json
       │   ├── ego_pose.json
       │   ├── instance.json
       │   ├── log.json
       │   ├── map.json
       │   ├── sample_annotation.json
       │   ├── sample_data.json
       │   ├── sample.json
       │   ├── scene.json
       │   ├── sensor.json
       │   └── visibility.json
       ├── samples/
       │   ├── CAM_BACK/
       │   ├── CAM_FRONT/
       │   ├── LIDAR_TOP/
       │   ├── RADAR_FRONT/
       │   └── ... (other sensor folders)
       ├── sweeps/
       │   └── ... (sensor data folders)
       └── maps/
           └── ... (map PNG files)
   ```

   **Note**: The script expects the data to be in `scripts/data/` relative to where you run the script. The `dataroot='data'` parameter in `test_nuScenes.py` assumes you're running from the `scripts/` directory.

## Running the Project

### Running the Main EKF Script

1. **Navigate to the scripts directory**:
   ```bash
   cd Extended-Kalman-Filter-Sensor-Fusion/scripts
   ```

2. **Run the main script**:
   ```bash
   python test_nuScenes.py
   ```

   This will:
   - Load the NuScenes 1.0 mini dataset
   - Initialize the EKF with predefined parameters
   - Track a lead vehicle in the first scene
   - Extract real LiDAR and Radar measurements from point clouds
   - Run the EKF filter and estimate the vehicle state
   - Generate plots showing:
     - Position measurements vs. ground truth
     - EKF state estimates vs. ground truth
     - State estimation errors over time
     - RMSE statistics

3. **View Results**:
   The script will display plots showing:
   - **Position Measurements & EKF vs Ground Truth**: Comparison of LiDAR/Radar measurements and EKF estimates
   - **State Estimation Errors**: Error time series for x, y, velocity, and heading
   - **RMSE Statistics**: Printed to console

### Running the EKF Script Standalone

You can also run the EKF implementation with synthetic data:

```bash
cd Extended-Kalman-Filter-Sensor-Fusion/scripts
python ekf_script.py
```

This will run a simple simulation with synthetic measurements.

### Using the Visualization Notebook

To explore the NuScenes data interactively:

```bash
jupyter notebook nuScenes_visualizsation.ipynb
```

## EKF Parameters

The EKF is configured with the following parameters (can be modified in `test_nuScenes.py`):

- **Process Noise Covariance (Q)**: `diag([0.05, 0.05, 0.02, 0.01])`
- **Measurement Noise Covariance (R)**: `diag([0.5, 0.1, 1.5, 3.0, 0.02])`
- **LiDAR Reliability Parameters**: `α1=0.1, β1=1.0, X_lidar_reli=25.0m`
- **Radar Reliability Parameters**: `α2=0.10, β2=0.5, X_radar_reli=50.0m`

These parameters can be tuned based on your specific use case and sensor characteristics.

## References

- Kim, H., & Park, J. (2020). Extended Kalman Filter (EKF) Design for Vehicle Position Tracking Using Reliability Function of Radar and Lidar. *Sensors*, 20(14), 3856.
- NuScenes Dataset: https://www.nuscenes.org/