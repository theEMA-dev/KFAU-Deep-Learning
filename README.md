# KFAU Deep Learning

This repository contains coursework and assignments for the Deep Learning course at Konya Food and Agriculture University (KFAU).

## Repository Structure

```
├── assignment-1/           # Assignment 1: CNN for MNIST Digit Classification
│   ├── src/                # Source code
│   │   └── cnn.py          # CNN model implementation
│   ├── results/            # Experiment results with different configurations
│   ├── requirements.txt    # Python dependencies
│   └── report.pdf          # Assignment report
```

---

## Assignment 1: CNN for MNIST Digit Classification

### Overview

This assignment implements a **Convolutional Neural Network (CNN)** to classify handwritten digits from the [MNIST dataset](http://yann.lecun.com/exdb/mnist/). The model uses TensorFlow/Keras and evaluates performance using **5-fold cross-validation**.

### Model Architecture

The baseline CNN model consists of:
- **Input**: 28x28x1 grayscale images
- **Conv Block 1**: Conv2D (32 filters, 3x3) → MaxPooling (2x2)
- **Conv Block 2**: Conv2D (64 filters, 3x3)
- **Conv Block 3**: Conv2D (64 filters, 3x3) → MaxPooling (2x2)
- **Dense Layers**: Flatten → Dense (100 units) → Dense (10 units, softmax)

**Optimizer**: SGD with learning rate 0.01 and momentum 0.9

### Experiments

Seven different configurations were tested to analyze the impact of hyperparameters:

| Run | Configuration | Accuracy (Mean ± Std) | Key Changes |
|-----|---------------|----------------------|-------------|
| 1 | Baseline | 99.02% ± 0.07% | 3 Conv blocks, lr=0.01, 10 epochs |
| 2 | Shallow | 98.88% ± 0.18% | 2 Conv blocks, smaller dense layer (60 units) |
| 3 | Deep | 98.97% ± 0.17% | 4 Conv blocks (added 128 filters) |
| 4 | Low LR | 98.61% ± 0.14% | Learning rate = 0.001 |
| 5 | High LR | 24.11% ± 27.17% | Learning rate = 0.1 (unstable training) |
| 6 | More Epochs | 99.16% ± 0.09% | 50 epochs |
| 7 | **Best Config** | **99.17% ± 0.08%** | Baseline + 20 epochs |

### Key Findings

- **Learning rate** significantly impacts training stability. High learning rate (0.1) causes severe training instability.
- **Increased training epochs** (20 epochs) provides the best performance improvement.
- **Network depth** doesn't always improve results; the baseline architecture achieves near-optimal performance.
- All stable configurations achieve >98.5% accuracy on MNIST classification.

### Running the Code

1. **Install dependencies**:
   ```bash
   cd assignment-1
   pip install -r requirements.txt
   ```

2. **Run the training script**:
   ```bash
   python src/cnn.py
   ```

### Results

Each experiment folder in `results/` contains:
- `figure_1.png` - Training/validation loss curves
- `figure_2.png` - Accuracy distribution (box plot)
- `model_graph.png` - Visual representation of the CNN architecture
- `performance_summary.md` - Detailed performance metrics and model configuration

---

## Requirements

- Python 3.x
- TensorFlow
- Keras
- NumPy
- Matplotlib
- scikit-learn

See `assignment-1/requirements.txt` for specific version requirements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
