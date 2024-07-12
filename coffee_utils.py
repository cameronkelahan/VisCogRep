import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import re
import torch
from torch import nn
from torch.utils.data import DataLoader

# Load dataset functionality
def load_dataset(PATH):
    # Specify the directory containing the images and masks
    image_dir = PATH + '/coffee-datasets/segmentation/images/train/'
    mask_dir =  PATH + '/coffee-datasets/segmentation/annotations/train/'

    # Get lists of image and mask files
    image_files = glob.glob(os.path.join(image_dir, '*.jpg'))
    mask_files = glob.glob(os.path.join(mask_dir, '*_mask.png'))

    def extract_prefix(filename, mask=False):
        if mask:
            return re.match(r"(\d+)_mask\.png", os.path.basename(filename)).group(1)
        else:
            return re.match(r"(\d+)\.jpg", os.path.basename(filename)).group(1)

    image_dict = {extract_prefix(f): f for f in image_files}
    mask_dict = {extract_prefix(f, mask=True): f for f in mask_files}

    assert len(image_dict) == len(mask_dict), "Mismatch between number of images and masks"

    sorted_prefixes = sorted(image_dict.keys())
    paired_files = [(image_dict[prefix], mask_dict[prefix]) for prefix in sorted_prefixes]

    train_images = []
    train_masks = []

    for image_path, mask_path in paired_files:
        image = Image.open(image_path)
        mask = Image.open(mask_path)
        train_images.append(image)
        train_masks.append(mask)

    return train_images, train_masks

# Plot example image and mask
# - Used for original image
# - Used for resized image
# - Used for Histogram image?
def plot_image_and_mask(image, mask):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(image)
    axes[1].imshow(mask)

    plt.tight_layout()
    plt.show()

    return fig, axes

# # Resizing images and masks
def resize_image_and_mask(image, mask, size):
    resized_image = image.resize(size, Image.ANTIALIAS)
    resized_mask = mask.resize(size, Image.NEAREST)
    return resized_image, resized_mask

# CONSTANT: Define the mapping from RGB to class labels for the masks
MAPPING = {
    (0, 0, 0): 0, # Background
    (0, 176, 0): 1, # Leaf
    (255, 0, 0): 2 # symptom
}

# Convert masks to single channel images
def convert_image_to_single_channel(image_array):

    single_channel = np.zeros((image_array.shape[0], image_array.shape[1]), dtype=np.uint8)

    for rgb, class_index in MAPPING.items():
        mask = np.all(image_array == rgb, axis=-1)
        single_channel[mask] = class_index

    return single_channel

# Handles the processing of images to single channel
def process_images(images):
    processed_images = []
    for image_array in images:
        single_channel_image = convert_image_to_single_channel(image_array)
        processed_images.append(single_channel_image)
    return processed_images

# Histogram equalization
def histogram_equalization(image):
    img_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    return img_output

# calculate the mean and standard deviation
def calculate_mean_std(image_list, batch_size=64):
    mean = np.zeros(3)
    std = np.zeros(3)
    total_images_count = 0

    for i in range(0, len(image_list), batch_size):
        batch_images = np.stack(image_list[i:i+batch_size], axis=0)

        # Calculate mean and std for the current batch
        batch_mean = np.mean(batch_images, axis=(0, 1, 2))
        batch_std = np.std(batch_images, axis=(0, 1, 2))

        # Update the overall mean and std
        batch_image_count = batch_images.shape[0]
        total_images_count += batch_image_count

        mean = mean + batch_mean * batch_image_count
        std = std + batch_std * batch_image_count

    mean /= total_images_count
    std /= total_images_count

    return mean, std

# Create the data loaders
def make_data_loaders(full_dataset, batch_size_train=1, batch_size_val=1, batch_size_test=1):
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    validation_size = int(0.1 * total_size)
    test_size = total_size - train_size - validation_size

    generator = torch.Generator().manual_seed(42)
    train, validation, test = torch.utils.data.random_split(full_dataset, [train_size, validation_size, test_size], generator=generator)

    train_loader = DataLoader(train, batch_size=batch_size_train, shuffle=True, num_workers=2)
    validation_loader = DataLoader(validation, batch_size=batch_size_val, shuffle=False, num_workers=2)
    test_loader = DataLoader(test, batch_size=batch_size_test, shuffle=False, num_workers=2)

    return train_loader, validation_loader, test_loader

# Early stopping class
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_loss = None

    def __call__(self, val_loss, model, optimizer, epoch, save_path):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, epoch, save_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, epoch, save_path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, optimizer, epoch, save_path):
        '''Saves model when validation loss decreases.'''
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(save_path, 'checkpoint.pth'))
            print(f'Saving model with validation loss {val_loss:.4f}')

########## METRICS ##########

# Intersection over Union metric
def calculate_iou(pred, target, num_classes):
    pred = pred.view(-1)
    target = target.view(-1)
    ious = np.zeros(num_classes)
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).long().sum().item()
        union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
        if union > 0:
            ious[cls] = float(intersection) / float(max(union, 1))
    return ious

# Dice Coefficient metric
def calculate_dice(pred, target, num_classes):
    pred = pred.view(-1)
    target = target.view(-1)
    dices = np.zeros(num_classes)
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).long().sum().item()
        dice = (2. * intersection) / (pred_inds.long().sum().item() + target_inds.long().sum().item() + 1e-10)
        dices[cls] = dice
    return dices

# Pixel accuracy metric
def calculate_pixel_accuracy(pred, target, num_classes):
    pred = pred.view(-1)
    target = target.view(-1)
    accuracies = np.zeros(num_classes)
    for cls in range(num_classes):
        correct = (pred == target) & (target == cls)
        total = (target == cls).sum().item()
        if total > 0:
            accuracies[cls] = correct.float().sum().item() / total
    return accuracies

# Evaluate the loaded model
def evaluate_model(model, dataloader, device, num_classes):
    model.eval()
    total_iou = np.zeros(num_classes)
    total_dice = np.zeros(num_classes)
    total_pixel_accuracy = np.zeros(num_classes)
    total_samples = np.zeros(num_classes)
    predictions = []
    gt = []

    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
            images = images.permute(0, 3, 1, 2).float()

            model.to(device)

            outputs = model(images)

            _, preds = torch.max(outputs, 1)
            prediction = preds.to(device)
            targets = targets.to(device)

            ious = calculate_iou(preds, targets, num_classes)
            dices = calculate_dice(preds, targets, num_classes)
            accuracies = calculate_pixel_accuracy(preds, targets, num_classes)

            for cls in range(num_classes):
                mask = targets == cls
                total_samples[cls] += mask.sum().item()
                total_iou[cls] += ious[cls] * mask.sum().item()
                total_dice[cls] += dices[cls] * mask.sum().item()
                total_pixel_accuracy[cls] += accuracies[cls] * mask.sum().item()

            predictions.append(prediction.cpu().numpy())
            gt.append(targets.cpu().numpy())

    mean_iou = total_iou / total_samples
    mean_dice = total_dice / total_samples
    mean_pixel_accuracy = total_pixel_accuracy / total_samples

    return predictions, gt, mean_iou, mean_dice, mean_pixel_accuracy