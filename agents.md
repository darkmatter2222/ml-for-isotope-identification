# Agents.md - AI Agent Context for Synthetic Gamma Spectra Generation

This document provides comprehensive context for AI agents working on this project. It describes the system architecture, purpose, configuration options, and implementation details of the synthetic gamma spectra generation system.

## Project Purpose

This project generates **synthetic gamma-ray spectra** for training machine learning models to perform **isotope identification**. The goal is to create a neural network that can identify radioactive isotopes from gamma spectra captured by consumer-grade scintillation detectors (Radiacode devices).

### Why Synthetic Data?

1. **Real gamma spectra are expensive/dangerous to collect** - requires radioactive sources, permits, and safety protocols
2. **Need massive datasets** - ML models require 10,000-100,000+ training samples
3. **Controlled variation** - can systematically vary activities, durations, isotope combinations
4. **Ground truth labels** - perfect annotations impossible with real-world data
5. **Reproducibility** - can regenerate datasets with different parameters

## System Architecture

```
synthetic_spectra/
├── __init__.py              # Package initialization
├── config.py                # Detector configurations (Radiacode 101-110)
├── generator.py             # Main SpectrumGenerator class
├── generate_spectra.py      # CLI batch generation script
├── ground_truth/
│   ├── __init__.py
│   ├── isotope_data.py      # 82 isotopes with gamma emission lines
│   └── decay_chains.py      # U-238, Th-232, U-235, Cs-137 chains
└── physics/
    ├── __init__.py
    └── spectrum_physics.py  # Physics calculations for peak generation
```

## Core Concepts

### 1. Spectrum Representation

Each spectrum is a **2D numpy array**:
- **X-axis (columns):** 1023 energy channels mapping to 20 keV - 3000 keV
- **Y-axis (rows):** Time intervals (1-second bins)
- **Values:** Normalized counts [0, 1]

This creates an "image-like" representation suitable for CNN-based models.

### 2. Physics Model

The generation follows realistic gamma spectroscopy physics:

#### Peak Generation (Gaussian)
```
G(E) = (λ / (σ√(2π))) × exp(-(E - E₀)² / (2σ²))
```
Where:
- `E₀` = gamma line energy (keV)
- `σ = FWHM / 2.355` (standard deviation)
- `λ = A × t × I × ε × T` (expected counts)
  - `A` = activity (Bq)
  - `t` = counting time (s)
  - `I` = branching ratio (emission probability)
  - `ε` = detector efficiency
  - `T` = geometric factor

#### FWHM Scaling
```
FWHM(E) = FWHM_662 × √(E / 662)
```
Resolution degrades at higher energies following square-root scaling.

#### Background Model
- **Exponential continuum:** `B(E) = B₀ × exp(-E / E_char)`
- **Environmental isotopes:** K-40, Pb-214, Bi-214, Pb-212, Tl-208, Ac-228
- **Compton continuum:** Simplified model for scattered photons

#### Statistical Noise
**Poisson counting statistics** applied to all counts:
```
observed = Poisson(expected)
```

### 3. Detector Configurations

| Model | Crystal | FWHM @ 662 keV | Energy Range |
|-------|---------|----------------|--------------|
| radiacode_101 | CsI(Tl) | 9.0% | 20-3000 keV |
| radiacode_102 | CsI(Tl) | 9.5% | 20-3000 keV |
| radiacode_103 | CsI(Tl) | 8.4% | 20-3000 keV |
| radiacode_103g | GAGG(Ce) | 7.4% | 20-3000 keV |
| radiacode_110 | CsI(Tl) | 8.4% | 20-3000 keV |

### 4. Isotope Database

**82 isotopes** across categories:
- `NATURAL_BACKGROUND`: K-40, Ra-226, Rn-222
- `PRIMORDIAL`: U-238, U-235, Th-232
- `COSMOGENIC`: Be-7, Na-22, C-14
- `U238_CHAIN`: Pa-234m, Th-234, Ra-226, Pb-214, Bi-214, Pb-210, Po-210
- `TH232_CHAIN`: Ac-228, Ra-224, Pb-212, Bi-212, Tl-208
- `U235_CHAIN`: Pa-231, Th-227, Ra-223, Rn-219, Pb-211, Bi-211
- `CALIBRATION`: Am-241, Ba-133, Cs-137, Co-57, Co-60, Eu-152, Na-22, Mn-54
- `INDUSTRIAL`: Ir-192, Se-75, Cd-109, I-131, Y-90
- `MEDICAL`: Tc-99m, F-18, Ga-67, Ga-68, In-111, I-123, I-125, Tl-201, Lu-177
- `REACTOR_FALLOUT`: Cs-134, Cs-137, I-131, Sr-90, Zr-95, Nb-95, Ru-103, Ru-106, Ce-141, Ce-144
- `ACTIVATION`: Fe-59, Cr-51, Zn-65, Ag-110m, Sb-124, Sb-125

## Configuration & Variation

### SpectrumConfig Parameters

```python
@dataclass
class SpectrumConfig:
    detector_type: str = "radiacode_103"    # Which detector to simulate
    duration_seconds: int = 300              # Total measurement time
    interval_seconds: int = 1                # Time bin size (always 1s)
    include_background: bool = True          # Add environmental background
    background_scale: float = 1.0            # Background intensity multiplier
    noise_enabled: bool = True               # Apply Poisson statistics
    normalize: bool = True                   # Normalize to [0, 1]
```

### Variation in Generated Data

The `generate_training_batch()` function creates varied samples:

| Category | Proportion | Description |
|----------|------------|-------------|
| Single isotope | 40% | One source isotope + background |
| Dual isotope | 30% | Two source isotopes blended |
| Multi isotope | 20% | 3-5 isotopes combined |
| Background only | 10% | Environmental background only |

#### Activity Ranges
- **Activity:** 10-500 Bq (randomized per isotope)
- **Duration:** 60-600 seconds (randomized)
- **Background scale:** 0.5-2.0× (randomized)

#### Isotope Pool
Common isotopes used for generation:
```python
ISOTOPE_POOL = [
    "Am-241", "Ba-133", "Cs-137", "Co-57", "Co-60",
    "Eu-152", "Na-22", "Mn-54", "K-40", "Ra-226",
    "Th-232", "U-238", "I-131", "Tc-99m", "Ir-192"
]
```

## Output Format

### Directory Structure
```
data/synthetic/spectra/
├── {uuid}_spectrum.npy      # Numpy array (time × channels)
├── {uuid}_spectrum.png      # Visualization image
└── labels.json              # Metadata for all samples
```

### Labels JSON Schema
```json
{
  "sample_id": {
    "isotopes": [
      {
        "name": "Cs-137",
        "activity_bq": 123.45,
        "category": "CALIBRATION"
      }
    ],
    "background_isotopes": ["K-40", "Pb-214", ...],
    "detector": "radiacode_103",
    "duration_seconds": 300,
    "num_intervals": 300,
    "background_scale": 1.2,
    "generation_timestamp": "2025-01-24T..."
  }
}
```

### Numpy Array Details
- **Shape:** `(num_intervals, 1023)`
- **Dtype:** `float64`
- **Range:** `[0.0, 1.0]` (normalized)
- **Channel mapping:** `channel_i → 20 + i × (3000-20)/1023 keV`

## Usage

### Generate Test Batch (10 samples)
```bash
python -m synthetic_spectra.generate_spectra
```

### Generate Large Training Set
Edit `generate_spectra.py`:
```python
generate_training_batch(
    n_samples=100000,
    output_dir=Path("data/synthetic/spectra"),
    detector_type="radiacode_103"
)
```

### Programmatic Generation
```python
from synthetic_spectra.generator import SpectrumGenerator, SpectrumConfig, IsotopeSource
from synthetic_spectra.config import RADIACODE_CONFIGS

generator = SpectrumGenerator(RADIACODE_CONFIGS["radiacode_103"])

config = SpectrumConfig(
    duration_seconds=300,
    include_background=True,
    background_scale=1.0
)

sources = [
    IsotopeSource(isotope_name="Cs-137", activity_bq=100.0),
    IsotopeSource(isotope_name="Co-60", activity_bq=50.0)
]

spectrum = generator.generate_spectrum(sources, config)
# spectrum.data is the 2D numpy array
# spectrum.metadata contains all generation parameters
```

## Future Enhancements

### Planned Improvements
1. **Compton edge modeling** - More realistic continuum shapes
2. **Pile-up effects** - High count rate distortions
3. **Gain drift simulation** - Energy calibration shifts over time
4. **Source geometry** - Distance/shielding effects
5. **Decay during measurement** - Short-lived isotope activity changes

### ML Integration Points
- PyTorch `Dataset` class for training
- TensorFlow `tf.data` pipeline
- Data augmentation (noise injection, channel shifts)
- Stratified sampling by isotope/category

## Key Files for Modification

| File | Purpose | When to Modify |
|------|---------|----------------|
| `config.py` | Detector specs | Add new detector types |
| `isotope_data.py` | Isotope database | Add isotopes, update gamma lines |
| `decay_chains.py` | Chain relationships | Add decay chain logic |
| `spectrum_physics.py` | Physics model | Improve realism |
| `generator.py` | Generation logic | Add features, change output format |
| `generate_spectra.py` | Batch generation | Adjust sample distribution |

## Dependencies

```
numpy>=1.24.0      # Array operations
scipy>=1.10.0      # Statistical functions
pillow>=9.0.0      # PNG image generation
```

## References

- Radiacode device specifications: https://radiacode.com/
- Gamma spectroscopy physics: Knoll, "Radiation Detection and Measurement"
- Isotope data: IAEA Nuclear Data Services, NNDC at Brookhaven

---

*This document should be updated whenever significant changes are made to the generation system.*
