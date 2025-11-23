# Reproducing Our Results

## 1. Environment Setup
Create  a python virtual env and install dependencies
- Create env: `python3 -m venv venv`
- Activate: `source venv/bin/activate`
- Install: `pip install -r requirements.txt`

## 2. Dataset Preparation (Download via curl)

our project uses the DIV2K dataset. You can download it directly from the source using curl.

### 1. Create a dataset directory
```bash
mkdir -p DIV2K/train
mkdir -p DIV2K/valid

curl -L https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip -o DIV2K_train_HR.zip
curl -L https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip -o DIV2K_valid_HR.zip

unzip DIV2K_train_HR.zip -d .
unzip DIV2K_valid_HR.zip -d .

mv DIV2K_train_HR/* DIV2K/train/
mv DIV2K_valid_HR/* DIV2K/valid/
```
Final structure 
DIV2K/
  train/
  valid/   

## 3. Training the Models
Our study trains four variants of the UNet denoiser:
1. Baseline UNet
2. Channel-pruned UNet (structured + unstructured pruning)
3. QAT UNet (quantization-aware training to INT8)
4. QAT + pruned UNet

## 4. Evaluation
After training all models, the provided evaluation script loads each checkpoint and:
- rebuilds the correct model architecture
- converts QAT checkpoints into true INT8 models
- computes PSNR, SSIM, and loss
- measures parameter count and sparsity
- benchmarks CPU inference latency
- outputs a full comparison table
Note: benchmarks are not made using onnx exports as onnx has low compatibility with true quantized models.

This reproduces all quantitative results reported in the study.

## 5. Visualization
Our evaluation script comes with visualization functionality to compare the onnx exports of the models