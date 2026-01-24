# ML for Isotope Identification

A machine learning system for identifying radioactive isotopes from gamma-ray spectra captured by Radiacode scintillation detectors.

## Project Status

✅ **Completed:** Synthetic gamma spectra generation system  
🔲 **Next:** ML model training pipeline  
🔲 **Future:** Real-time inference on Radiacode devices

---

## Overview

This project aims to build a neural network that can identify radioactive isotopes from gamma spectra. Since collecting real gamma spectra requires radioactive sources and is expensive/regulated, we generate **synthetic training data** based on realistic physics models.

### Target Hardware
- **Training:** NVIDIA RTX 5090 GPU
- **Inference:** Radiacode 101, 102, 103, 103G, 110 scintillation detectors

### Data Format
- **Input:** 2D spectrograms (time intervals × 1023 energy channels)
- **Output:** Isotope classification with activity estimation

---

## Synthetic Spectra Generation

### Features
- **82 isotopes** with accurate gamma emission lines
- **Realistic physics:** Gaussian peaks, Poisson noise, Compton continuum, environmental background
- **Multiple detector models:** Radiacode 101, 102, 103, 103G, 110 with correct FWHM and energy ranges
- **Configurable variation:** Activity levels, measurement durations, isotope combinations

### Quick Start

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# or: source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install numpy scipy pillow

# Generate 10 test samples
python -m synthetic_spectra.generate_spectra
```

### Output Structure
```
data/synthetic/spectra/
├── {uuid}_spectrum.npy      # 2D numpy array (time × 1023 channels)
├── {uuid}_spectrum.png      # Visualization
└── labels.json              # Metadata and ground truth labels
```

### Sample Distribution
| Type | Proportion | Description |
|------|------------|-------------|
| Single isotope | 40% | One source + background |
| Dual isotope | 30% | Two sources blended |
| Multi isotope | 20% | 3-5 sources combined |
| Background only | 10% | Environmental only |

### Scaling Up
Edit `synthetic_spectra/generate_spectra.py` to generate larger datasets:
```python
generate_training_batch(
    n_samples=100000,  # Generate 100k samples
    output_dir=Path("data/synthetic/spectra"),
    detector_type="radiacode_103"
)
```

---

## Project Structure

```
ml-for-isotope-identification/
├── README.md                    # This file
├── agents.md                    # AI agent context documentation
├── .gitignore                   # Git ignore rules
├── synthetic_spectra/           # Spectrum generation package
│   ├── __init__.py
│   ├── config.py                # Detector configurations
│   ├── generator.py             # Main generation logic
│   ├── generate_spectra.py      # CLI batch generation
│   ├── ground_truth/
│   │   ├── isotope_data.py      # 82 isotopes database
│   │   └── decay_chains.py      # Decay chain definitions
│   └── physics/
│       └── spectrum_physics.py  # Physics calculations
└── data/                        # Generated data (git-ignored)
    └── synthetic/
        └── spectra/
```

---

## Technical Details

### Detector Specifications
| Model | Crystal | FWHM @ 662 keV | Energy Range | Channels |
|-------|---------|----------------|--------------|----------|
| Radiacode 101 | CsI(Tl) | 9.0% | 20-3000 keV | 1024 |
| Radiacode 102 | CsI(Tl) | 9.5% | 20-3000 keV | 1024 |
| Radiacode 103 | CsI(Tl) | 8.4% | 20-3000 keV | 1024 |
| Radiacode 103G | GAGG(Ce) | 7.4% | 20-3000 keV | 1024 |
| Radiacode 110 | CsI(Tl) | 8.4% | 20-3000 keV | 1024 |

### Physics Model
- **Peak shape:** Gaussian with FWHM scaling as √(E/662)
- **Expected counts:** λ = A × t × I × ε × T
- **Noise:** Poisson counting statistics
- **Background:** Exponential continuum + environmental isotopes (K-40, Pb-214, Bi-214, etc.)

### Isotope Categories
- Natural background (K-40, Ra-226, Rn-222)
- Decay chains (U-238, Th-232, U-235)
- Calibration sources (Am-241, Cs-137, Co-60, Ba-133, Eu-152)
- Medical isotopes (Tc-99m, F-18, I-131, Ga-68)
- Industrial sources (Ir-192, Se-75)
- Reactor fallout (Cs-134, Cs-137, Sr-90)

---

## Development

### Dependencies
```
numpy>=1.24.0
scipy>=1.10.0
pillow>=9.0.0
```

### For AI Agents
See [agents.md](agents.md) for comprehensive documentation on:
- System architecture and design decisions
- Physics model implementation details
- Configuration options and variation strategies
- Key files and modification points

---

## TODO

- [ ] **Push to repository** - Initial commit with generation system
- [ ] Create PyTorch DataLoader for training
- [ ] Implement CNN/Transformer model architecture
- [ ] Add data augmentation pipeline
- [ ] Create training script with logging
- [ ] Add model evaluation metrics
- [ ] Implement real-time inference module
- [ ] Create Radiacode device integration

---

## License

[TBD]

---

## Acknowledgments

- Radiacode for device specifications
- IAEA Nuclear Data Services for isotope data
- NNDC at Brookhaven National Laboratory