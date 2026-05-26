# 📝 Multi-Modal OCR Insurance ID Classifier

![digitizing team](digitizing_team.png)

DigiNsure Inc. is an innovative insurance company focused on enhancing the efficiency of processing claims and customer service interactions. Their newest initiative is digitizing all historical insurance claim documents, which includes improving the labeling of some IDs scanned from paper documents and identifying them as primary or secondary IDs.

To help them in their effort, you'll be using multi-modal learning to train an Optical Character Recognition (OCR) model. To improve the classification, the model uses **images** of the scanned documents as input and their **insurance type** (home, life, auto, health, or other). Integrating different data modalities (such as image and text) enables the model to perform better in complex scenarios, helping to capture more nuanced information. 

This project uses **PyTorch** to build, initialize, and train a neural network capable of processing both image tensors and classification vectors simultaneously.

---

## 📌 Guiding Questions
- How can a neural network architecture ingest both a 2D visual tensor and a 1D categorical text vector?
- What configuration of convolutional, pooling, and dense layers optimizes feature extraction for a $64 \times 64$ grayscale document scan?
- How does the loss converge over 10 training epochs when optimized with Adam and Cross-Entropy Loss?

---

## 📂 The Data

The dataset features dual-input components generated via a processing pipeline (`ProjectDataset`):

| Component | Data Type / Structure | Description |
|-----------|-----------------------|-------------|
| **Image Input** | `Tensor` (1, 64, 64) | Grayscale image patches of scanned IDs resized to $64 \times 64$ pixels. |
| **Type Input** | `Tensor` (5,) | One-hot encoded vector representing the insurance sector: `home`, `life`, `auto`, `health`, or `other`. |
| **Labels** | `Scalar` (0 or 1) | The target classification index mapping to either a `primary_id` (0) or `secondary_id` (1). |

---

## 🛠️ Project Steps

### 1. Defining the OCRModel Class
- Built a custom neural network using `nn.Module` containing an `image_layer` sequential block.
- Configured a 2D convolutional layer (`nn.Conv2d`) with 1 input channel, 16 filters, a $3 \times 3$ kernel, and a padding of 1, followed by a `nn.ReLU` activation and a $2 \times 2$ max-pooling layer.
- Implemented `nn.Flatten()` to transform visual features into a dense vector before passing them to a 128-unit linear layer.
- Designed a parallel `type_layer` mapping the 5-class one-hot categorical input to 10 latent features.
- Created a final `classifier` block that concatenates the 128 visual features and 10 textual features ($128 + 10 = 138$), mapping them down to the final target dimension.

### 2. Setting Up the Training Pipeline
- Initialized the model parameters as `model`.
- Configured the **Adam Optimizer** (`optim.Adam`) with a learning rate of `0.001` to manage weight updates.
- Set up **Cross-Entropy Loss** (`nn.CrossEntropyLoss`) as the training criterion to evaluate prediction errors.

### 3. Executing the Multi-Modal Loop
- Constructed a PyTorch `DataLoader` to handle batch delivery of the mixed multi-modal variables seamlessly.
- Written a structural training loop that zero-centers gradients, executes parallel forward passes, tracks loss backpropagation, and outputs average loss metrics over 10 complete training epochs.

---

## ✅ Key Deliverables
- **`OCRModel`**: A flexible multi-modal neural network architecture handling compound image-text structural inputs.
- An operational mini-batch training loop demonstrating steady loss reduction across consecutive epochs.

---

## 🧰 Skills Used
- Python
- Deep Learning Frameworks (`PyTorch`)
- Computer Vision (Convolutional Neural Networks)
- Multi-Modal Feature Concatenation
- Model Optimization & Training Loops
