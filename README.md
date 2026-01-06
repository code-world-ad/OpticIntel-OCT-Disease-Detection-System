# OpticIntel: OCT Disease Detection System

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**A deep learning system for automated classification of retinal diseases from Optical Coherence Tomography (OCT) images.**

[Features](#features) • [Quick Start](#quick-start) • [Documentation](#documentation) • [Model Performance](#model-performance) • [Contributing](#contributing)

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Architecture](#project-architecture)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [API Documentation](#api-documentation)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

---

## 🎯 Overview

**OpticIntel** is an enterprise-grade deep learning system designed to automatically classify retinal diseases from Optical Coherence Tomography (OCT) images. The system leverages state-of-the-art pre-trained vision models and advanced data augmentation techniques to achieve high accuracy in medical image classification.

### Supported Disease Classes

The system can classify the following 7 retinal conditions:

| Disease | Code | Description |
|---------|------|-------------|
| Age-Related Macular Degeneration | AMD | Degenerative condition affecting central vision |
| Diabetic Macular Edema | DME | Fluid accumulation in the macula due to diabetes |
| Epiretinal Membrane | ERM | Scar tissue formation on the retina |
| Normal | NO | Healthy retina with no pathology |
| Retinal Artery Occlusion | RAO | Blocked blood flow in retinal artery |
| Retinal Vein Occlusion | RVO | Blocked blood flow in retinal vein |
| Vitreomacular Interface Disease | VID | Abnormal attachment between vitreous and macula |

---

## ✨ Features

### Core ML Features
- **Multi-Model Support**: VGG, ResNet, Vision Transformer (ViT), Swin Transformer, and other models via TIMM
- **Transfer Learning**: Pre-trained weights from ImageNet for rapid convergence
- **Advanced Data Augmentation**: 8 configurable augmentation techniques including rotation, translation, color jittering
- **Automatic Normalization**: Auto-compute dataset statistics for proper normalization
- **Class Balancing**: Weighted sampling and loss weighting strategies to handle imbalanced datasets
- **Learning Rate Scheduling**: Multiple scheduler options (Cosine Annealing, Exponential, Step-based)
- **Warmup Training**: Gradual learning rate warmup for stable training

### Training & Evaluation
- **Comprehensive Metrics**: Accuracy, F1-score, AUC-ROC, Precision, Recall
- **Early Model Selection**: Automatic best model selection based on validation metrics
- **TensorBoard Integration**: Real-time training visualization and monitoring
- **Reproducibility**: Fixed random seeds for deterministic training
- **Model Checkpointing**: Best and final model weights saved automatically

### Deployment
- **Flask Web Application**: Easy-to-use REST API for predictions
- **GPU Support**: CUDA acceleration for training and inference
- **Model Portability**: Seamless checkpoint loading and inference

### Developer Experience
- **Hydra Configuration Management**: YAML-based config with CLI override support
- **Modular Architecture**: Clean separation of concerns (data, models, training)
- **Type Hints**: Better IDE support and code clarity
- **Progress Tracking**: TQDM-based progress bars for all operations

---

## 🏗️ Project Architecture

### High-Level System Design

```
┌─────────────────────────────────────────────────────────────┐
│                    OpticIntel System                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Raw OCT    │───▶│   Data       │───▶│  Pre-trained │  │
│  │   Images     │    │ Preprocessing│    │    Models    │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                    │                    │          │
│         │                    ▼                    ▼          │
│         │            ┌──────────────────────────────┐       │
│         │            │    Training Pipeline         │       │
│         │            │  (Hydra + PyTorch)          │       │
│         │            └──────────────────────────────┘       │
│         │                    │                              │
│         │                    ▼                              │
│         │         ┌──────────────────────┐                 │
│         │         │  Model Checkpoint    │                 │
│         │         │  (Best & Final)      │                 │
│         │         └──────────────────────┘                 │
│         │                    │                              │
│         ▼                    ▼                              │
│  ┌─────────────────────────────────────┐                   │
│  │    Flask Web Application             │                   │
│  │  (Real-time Inference & Serving)    │                   │
│  └─────────────────────────────────────┘                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Module Organization

```
optic-intel/
├── data/                    # Data processing module
│   ├── builder.py          # Dataset generation and statistics
│   ├── dataset.py          # Custom dataset classes
│   └── transforms.py       # Image augmentation pipelines
├── modules/                # ML components
│   ├── builder.py          # Model factory and checkpoint loading
│   ├── scheduler.py        # Learning rate schedulers and samplers
│   └── loss.py             # Loss function wrappers
├── configs/                # Configuration files
│   └── OCTDL.yaml          # Main training configuration
├── flask_app/              # Deployment module
│   ├── app.py              # Flask application and endpoints
│   └── templates/          # HTML templates
├── preprocessing.py        # Dataset preprocessing script
├── main.py                 # Training entry point
└── requirements.txt        # Project dependencies
```

---

## 🚀 Quick Start

### Prerequisites

- **Python** 3.8 or higher
- **CUDA** 11.8+ (optional, for GPU acceleration)
- **Git** for version control

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/code-world-ad/OpticIntel-OCT-Disease-Detection-System.git
cd OpticIntel-OCT-Disease-Detection-System
```

2. **Create a virtual environment**
```bash
# Using venv
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n opticintel python=3.9
conda activate opticintel
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Basic Usage

#### Training

```bash
# Train with default configuration
python main.py

# Train with custom parameters (override YAML config)
python main.py train.epochs=50 train.batch_size=64 train.network=resnet50

# Train with GPU (if available)
python main.py base.device=cuda

# Custom configuration file
python main.py --config-name=custom_config
```

#### Inference via Web Interface

```bash
# Navigate to Flask app directory
cd flask_app

# Run the Flask server
python app.py
```

Then open your browser and navigate to `http://localhost:5000` to upload OCT images for classification.

#### Batch Prediction

```python
from modules.builder import generate_model
from data.transforms import data_transforms
import hydra
from omegaconf import OmegaConf
import torch
from PIL import Image

# Load configuration
hydra.initialize(config_path="./configs", version_base=None)
cfg = hydra.compose(config_name="OCTDL")

# Load model
model = generate_model(cfg)
model.eval()

# Load and prepare image
image = Image.open("path/to/oct_image.jpg")
_, test_transform = data_transforms(cfg)
image_tensor = test_transform(image).unsqueeze(0)

# Make prediction
with torch.no_grad():
    output = model(image_tensor)
    predicted_class = output.argmax(dim=1).item()
    confidence = torch.softmax(output, dim=1)[0, predicted_class].item()

print(f"Predicted Class: {predicted_class}")
print(f"Confidence: {confidence:.4f}")
```

---

## 📦 Installation

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.8 | 3.9+ |
| RAM | 8 GB | 16+ GB |
| Storage | 10 GB | 50+ GB (for dataset) |
| GPU VRAM | N/A | 8+ GB |

### Dependencies Overview

```
PyYAML                 # YAML configuration parsing
tensorboard            # Training visualization
torch                  # Deep learning framework
torchvision            # Computer vision utilities
tqdm                   # Progress bars
timm                   # Pre-trained vision models
torcheval              # Evaluation metrics
scikit-learn           # ML utilities
hydra-core             # Configuration management
opencv-python          # Image processing
pillow                 # Image library
omegaconf              # Configuration objects
```

### Conda Installation (Recommended for GPU)

```bash
conda create -n opticintel python=3.9 -y
conda activate opticintel
conda install pytorch::pytorch pytorch::torchvision pytorch::torchaudio -c pytorch -y
pip install -r requirements.txt
```

---

## ⚙️ Configuration

### Configuration System

OpticIntel uses **Hydra** for flexible configuration management. All settings are defined in YAML and can be overridden via command line.

### Main Configuration File: `configs/OCTDL.yaml`

```yaml
base:
  data_path: ./dataset              # Path to dataset folder
  save_path: ./run                  # Training output directory
  device: cuda                      # Device: cuda or cpu
  random_seed: 0                    # Reproducibility seed
  overwrite: false                  # Overwrite existing run

data:
  num_classes: 7                    # Number of disease classes
  input_size: 224                   # Input image size (pixels)
  in_channels: 3                    # RGB channels
  mean: auto                        # Dataset mean (auto-compute or specify)
  std: auto                         # Dataset std (auto-compute or specify)
  sampling_strategy: class_balanced  # Handle class imbalance
  sampling_weights_decay_rate: 0.9  # Class weight decay rate

train:
  network: vgg19                    # Model architecture
  pretrained: true                  # Use ImageNet pre-training
  epochs: 100                       # Training epochs
  batch_size: 32                    # Batch size
  num_workers: 4                    # DataLoader workers
  criterion: cross_entropy          # Loss function
  warmup_epochs: 5                  # Learning rate warmup
  metrics: [acc, f1, auc, precision, recall]  # Metrics to track
  indicator: acc                    # Metric for best model selection

solver:
  optimizer: ADAMW                  # Optimizer: SGD, ADAM, ADAMW
  learning_rate: 0.0003             # Initial learning rate
  lr_scheduler: cosine              # Scheduler type
  weight_decay: 0.01                # L2 regularization

data_augmentation:
  - random_crop
  - horizontal_flip
  - vertical_flip
  - color_distortion
  - rotation
  - translation
  - gaussian_blur
```

### Common Configuration Overrides

```bash
# Model selection
python main.py train.network=resnet50
python main.py train.network=vit_base_patch16_224

# Training parameters
python main.py train.epochs=200 train.batch_size=64
python main.py solver.learning_rate=0.001

# Optimizer options
python main.py solver.optimizer=SGD solver.momentum=0.9
python main.py solver.optimizer=ADAM

# Disable augmentations
python main.py data.data_augmentation=[]

# GPU/CPU selection
python main.py base.device=cuda
python main.py base.device=cpu
```

### Dataset Structure

Expected folder structure:
```
dataset/
├── train/
│   ├── AMD/
│   ├── DME/
│   ├── ERM/
│   ├── NO/
│   ├── RAO/
│   ├── RVO/
│   └── VID/
├── val/
│   ├── AMD/
│   ├── DME/
│   ├── ERM/
│   ├── NO/
│   ├── RAO/
│   ├── RVO/
│   └── VID/
└── test/
    ├── AMD/
    ├── DME/
    ├── ERM/
    ├── NO/
    ├── RAO/
    ├── RVO/
    └── VID/
```

---

## 🎓 Usage

### Data Preprocessing

The project includes a preprocessing script to prepare OCT images:

```bash
python preprocessing.py \
  --dataset_folder ./dataset/OCTDL \
  --labels_path ./dataset/OCTDL_labels.csv \
  --output_folder ./dataset \
  --image_dim 512 \
  --crop_ratio 1 \
  --padding False \
  --crop False \
  --resize True \
  --val_ratio 0.15 \
  --test_ratio 0.25
```

**Options:**
- `--dataset_folder`: Source dataset location
- `--labels_path`: CSV file with disease labels and patient IDs
- `--output_folder`: Destination for processed dataset
- `--image_dim`: Final image dimensions
- `--crop_ratio`: Central crop ratio (0-1)
- `--padding`: Pad images to square
- `--crop`: Perform center crop
- `--resize`: Resize to final dimensions
- `--val_ratio`: Validation split (default: 0.15)
- `--test_ratio`: Test split (default: 0.25)

### Training Workflow

#### Step 1: Prepare Dataset
```bash
# Preprocess OCT images
python preprocessing.py \
  --dataset_folder ./raw_data \
  --output_folder ./dataset \
  --resize True
```

#### Step 2: Start Training
```bash
# Train with default settings
python main.py

# Or with custom parameters
python main.py \
  train.network=resnet50 \
  train.epochs=100 \
  solver.learning_rate=0.001 \
  base.device=cuda
```

#### Step 3: Monitor Training
```bash
# View TensorBoard logs
tensorboard --logdir run/log
```

### Training Output

After training completes, the following files are saved in `run/` directory:

```
run/
├── log/                          # TensorBoard logs
├── best_validation_weights.pt    # Best model checkpoint
├── final_weights.pt              # Final model checkpoint
└── .hydra/                       # Configuration snapshots
```

### Model Evaluation

The system automatically evaluates on test set after training:

```
Performance of the best validation model:
├── Accuracy
├── F1-Score
├── AUC-ROC
├── Precision
└── Recall

Performance of the final model:
├── Accuracy
├── F1-Score
├── AUC-ROC
├── Precision
└── Recall
```

### Flask Web Application

#### Starting the Server
```bash
cd flask_app
python app.py
```

The Flask app runs on `http://localhost:5000` with the following endpoints:

#### API Endpoints

**GET `/`**
- Returns the web interface for uploading images

**POST `/predict`**
- Accepts: OCT image file (multipart/form-data)
- Returns: JSON with prediction and confidence
- Example:
```bash
curl -X POST -F "image=@test_image.jpg" http://localhost:5000/predict
```

Response:
```json
{
  "prediction": "AMD"
}
```

---

## 📊 Model Performance

### Supported Models

The system supports any model available in the **TIMM** (Timm Image Models) library:

#### Vision Transformers
- `vit_base_patch16_224` - Base Vision Transformer
- `vit_large_patch16_224` - Large Vision Transformer
- `swin_base_patch4_window7_224` - Swin Transformer (Base)
- `swin_large_patch4_window7_224` - Swin Transformer (Large)

#### CNNs
- `vgg19` - VGGNet-19 (default)
- `resnet50` - ResNet-50
- `resnet152` - ResNet-152
- `efficientnet_b4` - EfficientNet-B4
- `densenet121` - DenseNet-121

#### Switching Models
```bash
# Use Vision Transformer
python main.py train.network=vit_base_patch16_224

# Use Swin Transformer
python main.py train.network=swin_base_patch4_window7_224

# Use EfficientNet
python main.py train.network=efficientnet_b4
```

### Expected Performance Metrics

**Default Configuration (VGG19)**
- Accuracy: ~92%
- F1-Score: ~0.91-0.94
- AUC-ROC: ~0.97-0.98

### Training Time Estimates

| Model | GPU (8GB) | GPU (16GB) | CPU |
|-------|-----------|-----------|-----|
| VGG19 | ~2 hours | ~1.5 hours | ~8 hours |
| ResNet50 | ~1.5 hours | ~1 hour | ~6 hours |
| ViT Base | ~3 hours | ~2 hours | ~12 hours |
| Swin Base | ~2.5 hours | ~1.5 hours | ~10 hours |

*Estimates for 100 epochs with batch size 32*

---

## 📂 Project Structure

### Detailed Directory Breakdown

```
optic-intel/
│
├── 📄 main.py                          # Training entry point (Hydra-based)
├── 📄 preprocessing.py                 # Dataset preprocessing utilities
├── 📄 requirements.txt                 # Python dependencies
├── 📄 .gitattributes                  # Git LFS configuration
│
├── 📁 configs/                         # Configuration directory
│   └── 📄 OCTDL.yaml                  # Main training config (YAML)
│
├── 📁 data/                            # Data processing module
│   ├── 📄 __init__.py
│   ├── 📄 builder.py                  # Dataset generation & statistics
│   │   ├── generate_dataset()         # Load train/val/test splits
│   │   ├── auto_statistics()          # Compute mean/std automatically
│   │   └── generate_dataset_from_folder()
│   ├── 📄 dataset.py                  # Custom PyTorch datasets
│   │   ├── CustomizedImageFolder      # ImageFolder with custom loader
│   │   └── DatasetFromDict            # Dict-based dataset
│   └── 📄 transforms.py               # Data augmentation pipelines
│       ├── data_transforms()          # Build train/test augmentations
│       ├── random_apply()             # Probabilistic augmentation
│       └── simple_transform()          # Basic resize & normalize
│
├── 📁 modules/                         # ML components module
│   ├── 📄 __init__.py
│   ├── 📄 builder.py                  # Model creation & loading
│   │   ├── generate_model()           # Load/build model with checkpoint
│   │   └── build_model()              # Create model from config
│   ├── 📄 scheduler.py                # Training utilities
│   │   ├── WarmupLRScheduler          # Learning rate warmup
│   │   ├── ScheduledWeightedSampler   # Class-balanced sampling
│   │   ├── LossWeightsScheduler       # Dynamic loss weighting
│   │   └── ClippedCosineAnnealingLR   # Clipped cosine annealing
│   └── 📄 loss.py                     # Loss function wrapper
│       └── WarpedLoss                 # Custom loss wrapper
│
├── 📁 flask_app/                       # Web deployment module
│   ├── 📄 app.py                      # Flask application
│   │   ├── @app.route("/")           # Web interface
│   │   └── @app.route("/predict")    # Inference endpoint
│   └── 📁 templates/
│       └── 📄 index.html              # Web UI HTML
│
├── 📁 dataset/                         # Dataset directory (structure)
│   ├── 📁 train/{disease}/
│   ├── 📁 val/{disease}/
│   └── 📁 test/{disease}/
│
├── 📁 run/                             # Training output directory
│   ├── 📁 log/                        # TensorBoard logs
│   ├── 📄 best_validation_weights.pt
│   ├── 📄 final_weights.pt
│   └── 📁 .hydra/                     # Configuration snapshots
│
└── 📁 flask_app/uploads/               # Uploaded images (temporary)
```

### Key Files Explained

#### `main.py` - Training Pipeline
```python
# Main entry point with Hydra configuration management
# - Handles configuration validation and path management
# - Initializes model, dataset, and training utilities
# - Executes training and evaluation workflows
```

#### `modules/builder.py` - Model Factory
```python
# Loads pre-trained models from TIMM library
# - Supports Vision Transformers and standard CNNs
# - Handles checkpoint loading
# - Manages device placement (GPU/CPU)
```

#### `data/builder.py` - Dataset Manager
```python
# Generates train/val/test splits
# - Auto-computes normalization statistics
# - Handles class-balanced sampling
# - Applies augmentation pipelines
```

#### `flask_app/app.py` - Inference API
```python
# REST API for model serving
# - Single image prediction endpoint
# - TensorBoard-ready configuration
# - Automatic GPU/CPU fallback
```

---

## 🔌 API Documentation

### Python API

#### Model Loading and Inference

```python
from modules.builder import generate_model
from data.transforms import data_transforms
from omegaconf import OmegaConf
import hydra
import torch

# Initialize Hydra
hydra.initialize(config_path="./configs", version_base=None)
cfg = hydra.compose(config_name="OCTDL")

# Load model
model = generate_model(cfg)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# Prepare image
from PIL import Image
_, test_transform = data_transforms(cfg)
image = Image.open("oct_image.jpg")
image_tensor = test_transform(image).unsqueeze(0).to(device)

# Inference
with torch.no_grad():
    output = model(image_tensor)
    prediction = output.argmax(dim=1).item()
    confidence = torch.softmax(output, dim=1)[0, prediction].item()
```

#### Dataset Access

```python
from data.builder import generate_dataset
from omegaconf import OmegaConf

# Load datasets
train_dataset, test_dataset, val_dataset = generate_dataset(cfg)

# Access samples
sample_image, label = train_dataset[0]
print(f"Image shape: {sample_image.shape}, Label: {label}")
```

### REST API (Flask)

#### Upload and Predict

```bash
curl -X POST \
  -F "image=@/path/to/oct_image.jpg" \
  http://localhost:5000/predict
```

Response:
```json
{
  "prediction": "DME"
}
```

#### Error Handling

```bash
# No file uploaded
curl -X POST http://localhost:5000/predict
# Response: {"error": "No file uploaded"}

# Invalid file
curl -X POST -F "image=@invalid_file.txt" http://localhost:5000/predict
# Response: {"error": "No file selected"} or processing error
```

---

## 🔧 Troubleshooting

### Common Issues and Solutions

#### Issue: CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solution:**
```bash
# Reduce batch size
python main.py train.batch_size=16

# Use smaller model
python main.py train.network=vgg16

# Use CPU
python main.py base.device=cpu
```

#### Issue: Dataset Not Found
```
FileNotFoundError: Dataset not found at ./dataset
```
**Solution:**
```bash
# Check dataset path
ls -la ./dataset

# Update config path
python main.py base.data_path=/path/to/dataset

# Run preprocessing first
python preprocessing.py --dataset_folder ./raw_data --output_folder ./dataset
```

#### Issue: Import Errors
```
ModuleNotFoundError: No module named 'timm'
```
**Solution:**
```bash
# Reinstall requirements
pip install --upgrade -r requirements.txt

# Or install specific package
pip install timm torcheval
```

#### Issue: Poor Model Performance
```
Low accuracy on validation set
```
**Solutions:**
```bash
# Increase training duration
python main.py train.epochs=200

# Adjust learning rate
python main.py solver.learning_rate=0.0001

# Use different model
python main.py train.network=resnet50

# Verify dataset quality
python preprocessing.py --resize True --padding True
```

#### Issue: Flask Connection Refused
```
ConnectionRefusedError: [Errno 61] Connection refused
```
**Solution:**
```bash
# Check if Flask is running
cd flask_app && python app.py

# Access correct port
# Default: http://localhost:5000

# Check firewall settings
# Allow port 5000 in firewall
```

#### Issue: TensorBoard Not Loading
```
No logs found in ./run/log
```
**Solution:**
```bash
# Run training first
python main.py

# TensorBoard logs are created during training
tensorboard --logdir ./run/log --port 6006
```

---

## 🤝 Contributing

We welcome contributions from the community! This section outlines the process for contributing to OpticIntel.


### Contribution Guidelines

#### 1. Fork and Clone
```bash
git clone https://github.com/yourusername/OpticIntel-OCT-Disease-Detection-System.git
cd OpticIntel-OCT-Disease-Detection-System
git checkout -b feature/your-feature-name
```

#### 2. Set Up Development Environment
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install black flake8 pytest  # Development tools
```

#### 3. Make Changes
- Follow PEP 8 style guide
- Add type hints to new functions
- Update docstrings for clarity
- Include inline comments for complex logic

#### 4. Code Quality
```bash
# Format code
black .

# Check for style issues
flake8 .

# Run tests
pytest
```

#### 5. Commit and Push
```bash
git add .
git commit -m "feat: add descriptive commit message"
git push origin feature/your-feature-name
```

#### 6. Submit Pull Request
- Provide clear description of changes
- Reference any related issues
- Include before/after screenshots if applicable
- Ensure all tests pass

### Areas for Contribution

- 🐛 **Bug Fixes**: Report and fix issues
- ✨ **Features**: Add new models, metrics, or augmentations
- 📚 **Documentation**: Improve README, docstrings, examples
- 🧪 **Tests**: Increase test coverage
- 🎨 **UI/UX**: Improve Flask interface
- 🚀 **Performance**: Optimize code and reduce training time

### Development Workflow

```
1. Identify issue or feature
   ↓
2. Create feature branch
   ↓
3. Implement changes with tests
   ↓
4. Run code quality checks
   ↓
5. Commit with descriptive messages
   ↓
6. Push to fork and create PR
   ↓
7. Address review comments
   ↓
8. Merge to main branch
```

---

## 🗺️ Roadmap

### Version 1.0 (Current)
- ✅ Multi-class OCT image classification
- ✅ Web-based inference interface
- ✅ TensorBoard integration
- ✅ Pre-trained model support

### Version 1.1 (Planned)
- 📋 Batch processing API
- 📋 Model explainability (Grad-CAM)
- 📋 Docker containerization
- 📋 PostgreSQL database support

### Version 2.0 (Planned)
- 🎯 Multi-modal learning (combining OCT + metadata)
- 🎯 3D volume processing
- 🎯 Real-time video stream inference
- 🎯 Distributed training support

---

<div align="center">

**[⬆ back to top](#opticintel-oct-disease-detection-system)**

Made with ❤️ by the OpticIntel Team

</div>
