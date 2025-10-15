 CIFAR-10 Dynamic Mixture-of-Kernels CNN (PyTorch)

A PyTorch image classifier for CIFAR-10 that uses **attention over multiple kernel sizes** inside each block, plus **BatchNorm**, **label smoothing**, **dropout**, and a **CosineAnnealing** LR schedule. The model learns to **soft-select** among 1×1/3×3/5×5/7×7 convolutions per block.

## TL;DR

* 🧠 Architecture: 3 stacked **IntermediateBlocks** (mixture of kernels with attention) → **OutputBlock** (GAP → MLP).
* 📚 Dataset: CIFAR-10 (auto-downloaded).
* 🛠️ Training: Adam, label smoothing, cosine LR.
* 💾 Saves: `best_model.pth` (by validation accuracy); prints train/val curves; evaluates on test set.

---

## Features

* **Dynamic kernel selection** via attention over parallel convs per block.
* **Regularization**: BatchNorm, dropout, label smoothing.
* **Scheduler**: CosineAnnealingLR.
* **Repro-friendly splits**: 90% train / 10% val from the official train set.
* **Plots**: accuracy per epoch, batch-wise training loss.

---

## Project Structure

```
.
├── train.py              
├── README.md
```

---

## Requirements

* Python 3.9+
* PyTorch + torchvision
* matplotlib

Install:

```bash
pip install torch torchvision matplotlib
```

(Optional) Verify CUDA:

```python
import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
```

---

## How it Works

### Data & Transforms

* **Train**: RandomCrop(32,4), HorizontalFlip, ColorJitter, RandomAffine, Normalize.
* **Val/Test**: ToTensor + Normalize.

> Note: CIFAR-10 has 3 channels. In `transforms.Normalize`, using `(0.5,)` broadcasts. For clarity you can use `(0.5, 0.5, 0.5)` for mean and std.

### Model

* **IntermediateBlock(in_channels, conv_layer_configs)**
  Parallel convs (1×1, 3×3, 5×5, 7×7) → BN → ReLU.
  Global average pooling of the **input** feeds a small FC to produce **attention weights** over branches.
  Weighted sum of branch outputs = block output.

* **OutputBlock(in_channels, hidden_dims=[256], num_classes=10)**
  Global Average Pooling → MLP(256, Dropout 0.3) → logits.

* **Stack**:
  `3→64→128→256` channels across 3 blocks → OutputBlock(256→256→10).

### Training

* Loss: `CrossEntropyLoss(label_smoothing=0.1)`
* Optimizer: `Adam(lr=2e-4, weight_decay=1e-4)`
* Scheduler: `CosineAnnealingLR(T_max=50)`
* Epochs: `50`
* Batch size: `64`

---

## Usage

### 1) Run training

If your script file is named `train.py`:

```bash
python train.py
```

What you’ll see:

* Per-epoch Train/Val loss + accuracy.
* Two plots:

  * Accuracy per epoch (train vs val)
  * Training loss per batch
* The best state dict saved to `best_model.pth`.
* Final test accuracy printed at the end.

### 2) Evaluate best model

The script already reloads `best_model.pth` and evaluates on the test set:

```
Final Test Accuracy: XX.XX%
```



## Expected Results

On CIFAR-10 with the provided config, you should typically see:

* **Val accuracy**: ~80–88% after 50 epochs (varies with hardware/seed/augs).
* **Test accuracy**: similar to the best val accuracy.

> Your exact score depends on randomness, augmentations, and GPU.

---

## Customization

**Training params**

* LR / weight decay: in `optimizer = optim.Adam(...)`
* Scheduler horizon: `T_max=50`
* Label smoothing: `nn.CrossEntropyLoss(label_smoothing=0.1)`
* Epochs/batch size: change `epochs` and `batch_size`.

**Architecture**

* Change branch kernels or channels inside each `IntermediateBlock`.
* Add/remove blocks in `self.blocks`.
* Adjust Output MLP via `hidden_dims` in `OutputBlock`.

**Transforms**

* Tune augmentation strength (e.g., `ColorJitter`, `RandomAffine`) for robustness vs. over-regularization.

---

## Repro Tips

* Set seeds to stabilize results (add at top of script):

  ```python
  import random, numpy as np, torch
  random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  ```


## Troubleshooting

**GPU not used**

* Check `print("Using device:", device)` shows `cuda`.
* Install CUDA-enabled PyTorch build from [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

**Training too slow**

* Increase `num_workers` in DataLoader (e.g., 4–8).
* Enable cuDNN benchmark (remove deterministic settings during exploration).

**Over/underfitting**

* Adjust augmentations, dropout (0.2–0.5), weight decay.
* Try a warmup or different scheduler.
* Increase/decrease model width (channels) or number of blocks.

**Matplotlib windows not showing on servers**

* Save plots instead of `plt.show()`:

  ```python
  plt.savefig("acc_per_epoch.png", dpi=200, bbox_inches="tight")
  ```

---

## License

MIT 
---

## Acknowledgements

* [PyTorch](https://pytorch.org/)
* [torchvision CIFAR-10](https://pytorch.org/vision/stable/datasets.html#cifar)
* The dynamic-branch idea is inspired by multi-branch CNNs (e.g., Inception-style kernels) with learned selection.
