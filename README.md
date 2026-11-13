# Drowsiness Detection System

This is a real-time Drowsiness Detection System using Deep Learning and Computer Vision. 
It uses a Flask backend, MobileNetV2 for eye/mouth state classification, and MediaPipe for face landmark detection.

## Project Structure
- `app.py`: Flask backend
- `train.py`: Model training script
- `model/`: Directory where trained models (e.g., `eye_model_mobilenet.h5`, `mouth_model_custom_cnn.h5`) will be saved
- `utils/detector.py`: MediaPipe Face Mesh & Region of Interest (ROI) extraction logic
- `static/` & `templates/`: Frontend UI

## Installation

Using `uv` (Recommended - extremely fast):
1. Create a virtual environment and activate it:
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   uv pip install -r requirements.txt
   ```

Using standard Python `pip`:
1. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Training the Models

The full dataset (2,900 images) is included directly in this repository:
- `dataset/eye_data/` containing `Closed/` and `Open/`
- `dataset/mouth_data/` containing `no_yawn/` and `yawn/`

Run the training script for any architecture:
```bash
# MobileNetV3 (Next-Gen Attention)
python train.py --eye_dir dataset/eye_data --mouth_dir dataset/mouth_data --model_type mobilenet_v3

# MobileNetV2 (Edge Baseline)
python train.py --eye_dir dataset/eye_data --mouth_dir dataset/mouth_data --model_type mobilenet

# Custom CNN (Scratch-built)
python train.py --eye_dir dataset/eye_data --mouth_dir dataset/mouth_data --model_type custom_cnn
```
   This will save `eye_model_custom_cnn.h5` and `mouth_model_custom_cnn.h5` into the `model/` directory.

## Running the App

Once models are trained (or if you already have them), start the Flask server:
```bash
python app.py
```
Open your browser and navigate to `http://localhost:5000/`.

## Logic
- **EAR (Eye Aspect Ratio)**: Live computed from MediaPipe landmarks. If `< 0.25` for 15+ frames, it triggers DROWSY status.
- **Eye Model**: Fallback / supplementary neural network predicting Closed vs Open eyes.
- **Mouth Model**: Classifies extracted mouth ROI to detect YAWN (triggers if Yawn > 70% confidence).

## Model Architectures & Performance

The system supports three hot-swappable architectures via the dashboard dropdown:

1. **MobileNetV2 (Production Baseline)**
   - Pre-trained 53-layer model optimized for edge devices.
   - **Eye Model Accuracy**: ~98%
   - **Mouth Model Accuracy**: ~95%+
   - *Best for: High-reliability production deployment and balanced performance.*

2. **MobileNetV3 (Next-Gen Attention & Hardware-Aware)**
   - Google's modern mobile architecture featuring **Squeeze-and-Excitation (SE) attention blocks** and **Hard-Swish** activation functions.
   - **Eye Model Accuracy**: **~100%** (Validation: 1.0000)
   - **Mouth Model Accuracy**: ~66% - 80%
   - *Best for: State-of-the-art edge detection with hardware-level attention.*

3. **Custom CNN (Scratch-Built)**
   - A fully custom Convolutional Neural Network built from scratch.
   - Upgraded with Deep stacking, GlobalAveragePooling, Batch Normalization, scaled Dropout layers, and Dynamic Learning Rate (ReduceLROnPlateau).
   - **Eye Model Accuracy**: ~95%
   - **Mouth Model Accuracy**: ~85%+
   - *Best for: Ultra-lightweight execution and demonstrating scratch-built deep learning capabilities.*
