import os, glob, random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

class Div2kDataSet(Dataset):
  def __init__(self, root, split="train", patch_size=128, sigma=25, augment = True):
    self.paths = sorted(glob.glob(os.path.join(root, split, "*.png")))
    self.patch_size = patch_size
    self.sigma = sigma / 255.0
    self.augment = augment

  def __len__(self):
    return len(self.paths)

  def __getitem__(self, idx):
      # ---- load clean image ----
      path = self.paths[idx]
      img = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0

      # ---- random crop ----
      if self.patch_size:
          H, W, _ = img.shape
          top = random.randint(0, H - self.patch_size)
          left = random.randint(0, W - self.patch_size)
          img = img[top:top+self.patch_size, left:left+self.patch_size, :]

      # ---- optional augmentation ----
      if self.augment:
          if random.random() < 0.5: img = np.flip(img, axis=0)   # vertical
          if random.random() < 0.5: img = np.flip(img, axis=1)   # horizontal
          if random.random() < 0.5: img = np.rot90(img)           # 90° rotation
          img = np.ascontiguousarray(img)

      # ---- add Gaussian noise ----
      noise = np.random.normal(0, self.sigma, img.shape)
      noisy = np.clip(img + noise, 0, 1)

      # ---- convert to tensors ----
      clean_t = torch.from_numpy(img).permute(2,0,1)
      noisy_t = torch.from_numpy(noisy).permute(2,0,1)

      return noisy_t, clean_t
      
  
  