# Standard imports
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import h5py
# Torch
import torch
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# Custom
from models import CustomClassifier
from trainer import Trainer


PATH = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]
TARGET_COLS = [
    "bowel_injury", "extravasation_injury",
    "kidney_healthy", "kidney_low", "kidney_high",
    "liver_healthy", "liver_low", "liver_high",
    "spleen_healthy", "spleen_low", "spleen_high",
]


class RSNAData(torch.utils.data.Dataset):
    """To pass to dataloader during training."""
    def __init__(self, image_file_path, train_df, dname="images", transform=None):
        super(RSNAData).__init__()
        self.image_file = h5py.File(image_file_path, "r")
        self.dataset = self.image_file[dname]
        self.image_idx, self.labels = self.process_labels(train_df)
        self.transform = transform

    def __getitem__(self, i):
        idx = self.image_idx[i]
        data = self.dataset[idx, ...]
        if self.transform is not None:
            data = self.transform(data)
        return data, self.labels[idx, ...]

    def __len__(self):
        return len(self.image_idx)

    def __del__(self):
        self.image_file.close()

    def get_max(self, chunk_size=1000):
        """Loop over data to get maximum of image pixel values without putting all in memory."""
        total_max = -np.inf
        for start in tqdm(np.arange(0, len(self), chunk_size),
                          desc="Get max", leave=False):
            _max = np.max(self.dataset[start:start+chunk_size, ...])
            if _max > total_max:
                total_max = _max
        return total_max

    def process_labels(self, train_df):
        """Get image id and labels for train_df's dataset."""
        label_names = ["images_idx"] + TARGET_COLS
        out_data = np.array(train_df[label_names], dtype=np.int8)
        return out_data[:, 0], out_data[:, 1:]


def custom_loss(outputs, labels):
    """Implement BCE on dual outputs and "normal" CE on triple."""
    losses, total_loss = {}, 0.

    # Convert labels to dict-organ format
    targets = {}
    organs = ['bowel', 'extra', 'liver', 'kidney', 'spleen']
    for i, organ in enumerate(organs):
        if organ in ["bowel", "extra"]:
            targets[organ] = labels[:, i].view(-1, 1).to(torch.float32)
        else:
            targets[organ] = labels[:, 2+3*(i-2):2+3*(i-2+1)].to(torch.float32)

    # Calculate loss in every category and add
    bce = nn.BCELoss()
    cross_entropy = nn.CrossEntropyLoss()
    for organ, output in outputs.items():
        if organ in ["bowel", "extra"]:
            losses[organ] = bce(output, targets[organ])
        else:
            losses[organ] = cross_entropy(output, targets[organ])
        total_loss += losses[organ]

    return total_loss, losses


class DataTransform:
    """Apply image transformation to comply with mobilenetv2 requirements."""

    def __init__(self, max_val):
        self.max_val = max_val
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    def __call__(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)

        x = x / self.max_val
        x = x.unsqueeze(0)
        x = x.repeat_interleave(3, dim=0)
        x = self.normalize(x)
        return x


def main(val_frac=.1, verbose=False):
    """Testing."""
    # Hyper-parameters
    img_size = 512  # TODO - make data fit this or at least check
    neck_size = 32
    max_epoch = 50
    batch_size_train = 100
    batch_size_val = batch_size_train
    IMG_MAX = 254
    # Optimiser
    learning_rate = 1e-3
    # Scheduler
    lr_factor = .1
    patience = 5

    # Init and load data
    np.random.seed(42)
    image_file = os.path.join(PATH, "data", "train_images", "training.h5")
    raw_labels = pd.read_json(os.path.join(PATH, "data", "training_labels_df.json"))

    # Split into training/validation
    train_df, val_df = pd.DataFrame(), pd.DataFrame()
    for _, group in raw_labels.groupby(TARGET_COLS):
        val_group = group.sample(frac=val_frac)
        val_df = pd.concat([val_df, val_group])
        train_df = pd.concat([train_df, group.drop(val_group.index)])

    # Data preparation
    # See https://pytorch.org/hub/pytorch_vision_mobilenet_v2/
    img_transform = DataTransform(max_val=IMG_MAX)
    train_data = RSNAData(image_file, train_df, transform=img_transform)
    val_data = RSNAData(image_file, val_df, transform=img_transform)
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size_train,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=batch_size_val,
                                             shuffle=True)

    # Data augmentation # TODO

    # Training
    model = CustomClassifier(img_size=img_size, neck_size=neck_size, train_backbone=False)
    optimiser = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, factor=lr_factor, patience=patience, cooldown=0, verbose=True)
    RSNATrainer = Trainer(model=model,
                          loss=custom_loss,
                          train_loader=train_loader,
                          val_loader=val_loader,
                          optimiser=optimiser,
                          scheduler=scheduler,
                          max_epoch=max_epoch,
                          device="cuda")
    RSNATrainer.train(verbose=True)

    # Testing
    # model.eval()
    # TODO


if __name__ == '__main__':
    main(verbose=True)
