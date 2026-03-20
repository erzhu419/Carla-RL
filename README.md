# Carla-RL

Reinforcement Learning codebase for self-driving car in CARLA, using **PyTorch** (migrated from TensorFlow/Keras).

## Architecture

This project uses a **split setup**:
- **Windows**: Runs the CARLA simulator (`CarlaUE4.exe`) for 3D simulation and rendering.
- **WSL2 (Linux)**: Runs the Python training code (`train.py`) using PyTorch + CUDA.

Communication between WSL2 and Windows CARLA is done via TCP. WSL2's **mirrored networking mode** (`networkingMode=mirrored` in `.wslconfig`) is required for `localhost` to route correctly between WSL2 and Windows.

## Setup

### 1. Windows — CARLA Simulator

- Download and extract **CARLA 0.9.16** for Windows.
- Launch `CarlaUE4.exe` and wait until the city map is fully loaded.
- No additional configuration needed — CARLA listens on `localhost:2000` by default.

### 2. WSL2 — Enable Mirrored Networking

Create or edit `C:\Users\<YourUsername>\.wslconfig` on **Windows**:

```ini
[wsl2]
networkingMode=mirrored
```

Then restart WSL2:
```bash
wsl.exe --shutdown
```

### 3. WSL2 — Python Environment

```bash
# Create conda environment (Python 3.10 recommended)
conda create -n CARLA python=3.10
conda activate CARLA

# Install PyTorch with CUDA (adjust cuda version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install CARLA Python API (must match server version 0.9.16)
pip install carla==0.9.16

# Install remaining dependencies
pip install -r requirements.txt
```

### 4. Configure `settings.py`

- `CARLA_HOSTS_TYPE = 'remote'` — tells the script not to auto-launch CARLA.
- `CARLA_HOSTS = [['localhost', 2000, 10], ...]` — connects to Windows CARLA via mirrored localhost.

## Running

Make sure CARLA is running on Windows first, then in WSL2:

```bash
python3 train.py
```

To just play (no training):
```bash
python3 play.py
```

## Customization

- **Models**: Add or modify neural network architectures in `sources/models.py` (PyTorch `nn.Module` subclasses).
- **Settings**: Tune hyperparameters, episode length, image size, etc. in `settings.py`.
- **Reward function**: Edit `sources/carla.py`.

## Tech Stack

| Component | Library |
|---|---|
| Deep Learning | PyTorch |
| Logging | TensorBoard (`torch.utils.tensorboard`) |
| Simulation | CARLA 0.9.16 |
| Vision | OpenCV |
