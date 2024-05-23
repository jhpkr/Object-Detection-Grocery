import pandas as pd
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import torch
import torchvision
from torchvision import transforms, datasets
from torchvision.models.detection import *
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import json
import torch.optim as optim
from PIL import Image  # Import Image from PIL

from engine import train_one_epoch, evaluate
import utils
import torchvision.transforms as T

# For image augmentations
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

# As the data directory contains .xml files
from xml.etree import ElementTree as et

import warnings
warnings.filterwarnings('ignore')

class GroceryDataset(Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        with open(annotation) as f:
            self.coco = json.load(f)
        self.ids = list(sorted(self.coco['images'], key=lambda x: x['id']))
        self.ids = [img for img in self.ids if os.path.exists(os.path.join(self.root, img['file_name']))]

    def __getitem__(self, idx):
        img_id = self.ids[idx]['id']
        img_path = os.path.join(self.root, self.ids[idx]['file_name'])
        img = Image.open(img_path).convert("RGB")

        ann_ids = [ann for ann in self.coco['annotations'] if ann['image_id'] == img_id]
        boxes = [ann['bbox'] for ann in ann_ids]
        labels = [ann['category_id'] for ann in ann_ids]

        # Convert boxes from [x, y, width, height] to [x_min, y_min, x_max, y_max]
        boxes = [[b[0], b[1], b[0] + b[2], b[1] + b[3]] for b in boxes]

        # Filter out invalid boxes
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        valid_boxes = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes = boxes[valid_boxes]
        labels = torch.as_tensor(labels, dtype=torch.int64)[valid_boxes]

        image_id = torch.tensor([img_id])
        area = torch.as_tensor([ann['area'] for ann in ann_ids], dtype=torch.float32)[valid_boxes]
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': image_id,
            'area': area,
            'iscrowd': iscrowd
        }

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.ids)

transforms = T.Compose([T.ToTensor()])

def collate_fn(batch):
    return tuple(zip(*batch))

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss

def main():
    # Paths to the images and annotations
    train_image_path = 'coco/train'
    train_annotation_path = 'coco/train/train_annotations.coco.json'
    val_image_path = 'coco/valid'
    val_annotation_path = 'coco/valid/val_annotations.coco.json'
    test_image_path = 'coco/test'
    test_annotation_path = 'coco/test/test_annotations.coco.json'

    # Create datasets
    train_dataset = GroceryDataset(root=train_image_path, annotation=train_annotation_path, transforms=transforms)
    val_dataset = GroceryDataset(root=val_image_path, annotation=val_annotation_path, transforms=transforms)
    test_dataset = GroceryDataset(root=test_image_path, annotation=test_annotation_path, transforms=transforms)

    # Create data loaders
    train_data_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=1, collate_fn=collate_fn)
    val_data_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=1, collate_fn=collate_fn)
    test_data_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=1, collate_fn=collate_fn)

    # Load the pre-trained model
    model = fasterrcnn_resnet50_fpn(pretrained=True)

    # Replace the classifier with a new one, that has num_classes which is user-defined
    category_ids = [ann['category_id'] for ann in train_dataset.coco['annotations']]
    num_classes = len(set(category_ids)) + 1  # Including background
    print("Number of classes:", num_classes)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Move model to the GPU if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # Create optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Number of epochs
    num_epochs = 1

    # Initialize the early stopping object
    early_stopping = EarlyStopping(patience=5, verbose=True)

    for epoch in range(num_epochs):
        print('this is running')
        # Train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, train_data_loader, device, epoch, print_freq=10)
        # Update the learning rate
        lr_scheduler.step()
        # Evaluate on the validation dataset
        val_image_ids = []
        for _, targets in val_data_loader:
            for target in targets:
                val_image_ids.append(target['image_id'].item())
        print("Validation Data Loader Image IDs:", val_image_ids)
        eval_result = evaluate(model, val_data_loader, device)
        val_loss = eval_result.coco_eval['bbox'].stats[0]  # Assuming this is the validation loss
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # Final evaluation on the test dataset
    test_image_ids = []
    for _, targets in test_data_loader:
        for target in targets:
            test_image_ids.append(target['image_id'].item())
    print("Test Data Loader Image IDs:", test_image_ids)
    evaluate(model, test_data_loader, device)

if __name__ == '__main__':
    main()
