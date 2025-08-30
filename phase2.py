import os
import json
import time
import copy
import random
from collections import Counter

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler
from torch.quantization import QuantStub, DeQuantStub, prepare_qat, convert

import torchvision.models as models
import torchvision.transforms as T
from PIL import Image, ImageOps, ImageEnhance

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, roc_auc_score
)

from tqdm import tqdm


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


IMG_SIZE = 224  # Smaller for mobile models
BATCH_SIZE = 32  # Larger batch for QAT
NUM_EPOCHS = 30
LR_HEAD = 1e-3
LR_BACKBONE = 1e-5
WEIGHT_DECAY = 1e-4
LABEL_SMOOTH = 0.1
HEAD_WARMUP_EPOCHS = 3
USE_TTA = True
EARLY_STOP_PATIENCE = 8

USE_QAT = True
QAT_EPOCHS = 10
QAT_LR = 1e-4


USE_PRUNING = True
PRUNING_RATIO = 0.3
PRUNING_EPOCHS = 5


USE_ATTENTION = True
ATTENTION_RATIO = 16

import kagglehub


path = kagglehub.dataset_download("ninadaithal/imagesoasis")
print("Path to dataset files:", path)


EXPECTED_CLASSES = ['Non Demented', 'Very mild Dementia', 'Mild Dementia', 'Moderate Dementia']

def resolve_dataset_path(root_candidates):
    for root in root_candidates:
        if not root:
            continue
        candidates = [root, os.path.join(root, 'Data')]
        for cand in candidates:
            if os.path.isdir(cand):
                subdirs = [d for d in os.listdir(cand) if os.path.isdir(os.path.join(cand, d))]
                found = [d for d in subdirs if d in EXPECTED_CLASSES]
                if len(found) == len(EXPECTED_CLASSES):
                    return cand, found
                if len(found) > 0:
                    print(f"Found {found} in {cand}, but not all expected classes.")
                    return cand, found
    return None, []

DATASET_ROOT = os.environ.get('ALZ_DATASET_ROOT', None)

common_roots = [
    path,
    DATASET_ROOT,
    os.path.expanduser(r"~/.cache/kagglehub/datasets/ninadaithal/imagesoasis/versions/1"),
    os.path.expanduser(r"~/kaggle/input/imagesoasis"),
    r"C:\\Users\\sajib\\.cache\\kagglehub\\datasets\\ninadaithal\\imagesoasis\\versions\\1",
]

dataset_path, class_names = resolve_dataset_path(common_roots)
if not dataset_path:
    raise FileNotFoundError("Could not resolve dataset path. Set ALZ_DATASET_ROOT to the dataset root or adjust common_roots.")

print(f"Dataset path: {dataset_path}")
print(f"Classes: {class_names}")

class AlzheimerDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.class_to_idx = {
            'Non Demented': 0,
            'Very mild Dementia': 1,
            'Mild Dementia': 2,
            'Moderate Dementia': 3
        }
        if len(self.labels) > 0 and isinstance(self.labels[0], str):
            self.labels = [self.class_to_idx[lbl] for lbl in self.labels]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('L')
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        out = avg_out + max_out
        return torch.sigmoid(out).view(x.size(0), x.size(1), 1, 1)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        return torch.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, spatial_kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(spatial_kernel_size)
        
    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class MobileNetV3WithAttention(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        super(MobileNetV3WithAttention, self).__init__()

        self.backbone = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None)
        

        self.attention_modules = nn.ModuleList([
            CBAM(16, ATTENTION_RATIO),   # After first block
            CBAM(24, ATTENTION_RATIO),   # After second block
            CBAM(40, ATTENTION_RATIO),   # After third block
            CBAM(80, ATTENTION_RATIO),   # After fourth block
            CBAM(112, ATTENTION_RATIO),  # After fifth block
            CBAM(160, ATTENTION_RATIO),  # After sixth block
        ])
        
        # Dual classification heads
        original_in_features = self.backbone.classifier[3].in_features
        

        self.binary_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(original_in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2)  # Binary: Non-Dementia vs Dementia
        )
        

        self.multiclass_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(original_in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)  # 4 classes
        )
        

        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        
    def forward(self, x):
        x = self.quant(x)
        

        features = []
        x = self.backbone.features[0](x)  # First conv
        

        attention_idx = 0
        for i, layer in enumerate(self.backbone.features[1:]):
            x = layer(x)
            if i in [2, 5, 9, 13, 17, 21]: 
                if attention_idx < len(self.attention_modules):
                    x = self.attention_modules[attention_idx](x)
                    attention_idx += 1
        

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        

        binary_out = self.binary_head(x)
        multiclass_out = self.multiclass_head(x)
        
        binary_out = self.dequant(binary_out)
        multiclass_out = self.dequant(multiclass_out)
        
        return binary_out, multiclass_out


class StructuredPruner:
    def __init__(self, model, pruning_ratio=0.3):
        self.model = model
        self.pruning_ratio = pruning_ratio
        self.masks = {}
        
    def compute_magnitude(self, weight):

        return torch.norm(weight.view(weight.size(0), -1), dim=1)
    
    def create_mask(self, weight, pruning_ratio):

        magnitude = self.compute_magnitude(weight)
        threshold = torch.quantile(magnitude, pruning_ratio)
        mask = (magnitude > threshold).float()
        return mask.view(-1, 1, 1, 1)
    
    def apply_pruning(self):

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                mask = self.create_mask(module.weight, self.pruning_ratio)
                self.masks[name] = mask
                module.weight.data *= mask
                
    def remove_pruning(self):

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) and name in self.masks:
                module.weight.data /= (self.masks[name] + 1e-8)


def get_transforms(img_size=224, is_training=False):
    if is_training:
        return T.Compose([
            T.Resize(int(img_size * 1.1)),
            T.RandomCrop(img_size),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(10),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.Grayscale(num_output_channels=3),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return T.Compose([
            T.Resize(img_size),
            T.CenterCrop(img_size),
            T.Grayscale(num_output_channels=3),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def load_and_split(dataset_path, class_names, test_size=0.15, val_size=0.15, random_state=SEED):
    image_paths, labels = [], []
    for cls in class_names:
        cls_dir = os.path.join(dataset_path, cls)
        files = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for f in files:
            image_paths.append(os.path.join(cls_dir, f))
            labels.append(cls)
    print(f"Total images: {len(image_paths)}")

    X_temp, X_test, y_temp, y_test = train_test_split(
        image_paths, labels, test_size=test_size, stratify=labels, random_state=random_state
    )
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, stratify=y_temp, random_state=random_state
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def make_loaders(dataset_path, class_names, batch_size=BATCH_SIZE, img_size=IMG_SIZE, num_workers=0):
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_and_split(dataset_path, class_names)

    train_tf = get_transforms(img_size, True)
    eval_tf = get_transforms(img_size, False)

    train_ds = AlzheimerDataset(X_train, y_train, train_tf)
    val_ds = AlzheimerDataset(X_val, y_val, eval_tf)
    test_ds = AlzheimerDataset(X_test, y_test, eval_tf)


    num_classes = 4
    train_label_tensor = torch.tensor(train_ds.labels, dtype=torch.long)
    class_counts = torch.bincount(train_label_tensor, minlength=num_classes).float()

    class_weights = 1.0 / (class_counts + 1e-8)
    class_weights = class_weights / class_weights.sum() * num_classes
    print(f"Class weights: {class_weights.numpy()}")

    per_sample_w = class_weights[train_label_tensor]
    sampler = WeightedRandomSampler(
        weights=per_sample_w.double(),
        num_samples=len(per_sample_w),
        replacement=True
    )

    loaders = {
        'train': DataLoader(train_ds, batch_size=batch_size, sampler=sampler, shuffle=False,
                          num_workers=num_workers, pin_memory=True, drop_last=True),
        'val': DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True),
        'test': DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=True),
        'class_weights': class_weights
    }
    return loaders, {'class_names': class_names}


class DualLoss(nn.Module):
    def __init__(self, binary_weight=0.3, multiclass_weight=0.7):
        super(DualLoss, self).__init__()
        self.binary_weight = binary_weight
        self.multiclass_weight = multiclass_weight
        self.binary_criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
        self.multiclass_criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
        
    def forward(self, binary_out, multiclass_out, binary_target, multiclass_target):
        binary_loss = self.binary_criterion(binary_out, binary_target)
        multiclass_loss = self.multiclass_criterion(multiclass_out, multiclass_target)
        return self.binary_weight * binary_loss + self.multiclass_weight * multiclass_loss


def train_epoch(model, loader, criterion, optimizer, epoch, is_qat=False):
    model.train()
    scaler = GradScaler(enabled=torch.cuda.is_available())
    running_loss = 0.0
    correct_binary = 0
    correct_multiclass = 0
    total = 0
    pbar = tqdm(loader, desc=f"Train {epoch+1}")
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        

        binary_labels = (labels > 0).long()
        
        optimizer.zero_grad(set_to_none=True)
        
        with autocast('cuda', enabled=torch.cuda.is_available()):
            binary_out, multiclass_out = model(images)
            loss = criterion(binary_out, multiclass_out, binary_labels, labels)
        
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item() * images.size(0)
        binary_preds = binary_out.argmax(1)
        multiclass_preds = multiclass_out.argmax(1)
        correct_binary += (binary_preds == binary_labels).sum().item()
        correct_multiclass += (multiclass_preds == labels).sum().item()
        total += images.size(0)
        
        pbar.set_postfix(loss=running_loss/total, 
                        binary_acc=correct_binary/total,
                        multiclass_acc=correct_multiclass/total)
    
    return running_loss/total, correct_binary/total, correct_multiclass/total

def eval_epoch(model, loader, criterion, epoch, phase="Val"):
    model.eval()
    running_loss = 0.0
    correct_binary = 0
    correct_multiclass = 0
    total = 0
    pbar = tqdm(loader, desc=f"{phase} {epoch+1}")
    all_binary_probs, all_multiclass_probs, all_binary_labels, all_multiclass_labels = [], [], [], []
    
    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            binary_labels = (labels > 0).long()
            
            with autocast('cuda', enabled=torch.cuda.is_available()):
                binary_out, multiclass_out = model(images)
                loss = criterion(binary_out, multiclass_out, binary_labels, labels)
            
            running_loss += loss.item() * images.size(0)
            binary_probs = F.softmax(binary_out, dim=1)
            multiclass_probs = F.softmax(multiclass_out, dim=1)
            binary_preds = binary_out.argmax(1)
            multiclass_preds = multiclass_out.argmax(1)
            
            correct_binary += (binary_preds == binary_labels).sum().item()
            correct_multiclass += (multiclass_preds == labels).sum().item()
            total += images.size(0)
            
            all_binary_probs.append(binary_probs.detach().cpu())
            all_multiclass_probs.append(multiclass_probs.detach().cpu())
            all_binary_labels.append(binary_labels.detach().cpu())
            all_multiclass_labels.append(labels.detach().cpu())
            
            pbar.set_postfix(loss=running_loss/total, 
                            binary_acc=correct_binary/total,
                            multiclass_acc=correct_multiclass/total)
    
    all_binary_probs = torch.cat(all_binary_probs).numpy()
    all_multiclass_probs = torch.cat(all_multiclass_probs).numpy()
    all_binary_labels = torch.cat(all_binary_labels).numpy()
    all_multiclass_labels = torch.cat(all_multiclass_labels).numpy()
    
    return (running_loss/total, correct_binary/total, correct_multiclass/total,
            all_binary_probs, all_multiclass_probs, all_binary_labels, all_multiclass_labels)

# Main training function
def train_phase2_model(loaders, num_epochs=NUM_EPOCHS):
    print(f"\n==== Phase 2: MobileNetV3 with Attention ====")
    
    # Create model
    model = MobileNetV3WithAttention(num_classes=4, pretrained=True).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    
    # Loss and optimizer
    criterion = DualLoss(binary_weight=0.3, multiclass_weight=0.7)
    
    # Progressive unfreezing
    head_params, backbone_params = [], []
    for n, p in model.named_parameters():
        if any(k in n for k in ['binary_head', 'multiclass_head']):
            head_params.append(p)
        else:
            backbone_params.append(p)
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': LR_BACKBONE, 'weight_decay': WEIGHT_DECAY},
        {'params': head_params, 'lr': LR_HEAD, 'weight_decay': WEIGHT_DECAY}
    ])
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-7
    )
    
    history = {
        'train_loss': [], 'train_binary_acc': [], 'train_multiclass_acc': [],
        'val_loss': [], 'val_binary_acc': [], 'val_multiclass_acc': [],
        'test_loss': [], 'test_binary_acc': [], 'test_multiclass_acc': []
    }
    
    best_acc = 0.0
    best_state = None
    epochs_no_improve = 0
    
    # Phase 1: Full precision training
    print("Phase 1: Full precision training...")
    for epoch in range(num_epochs):
        if epoch < HEAD_WARMUP_EPOCHS:
            for p in backbone_params:
                p.requires_grad = False
        else:
            for p in backbone_params:
                p.requires_grad = True
        
        tr_loss, tr_binary_acc, tr_multiclass_acc = train_epoch(
            model, loaders['train'], criterion, optimizer, epoch)
        
        va_loss, va_binary_acc, va_multiclass_acc, va_binary_probs, va_multiclass_probs, va_binary_labels, va_multiclass_labels = eval_epoch(
            model, loaders['val'], criterion, epoch, phase='Val')
        
        te_loss, te_binary_acc, te_multiclass_acc, te_binary_probs, te_multiclass_probs, te_binary_labels, te_multiclass_labels = eval_epoch(
            model, loaders['test'], criterion, epoch, phase='Test')
        
        
        history['train_loss'].append(tr_loss)
        history['train_binary_acc'].append(tr_binary_acc)
        history['train_multiclass_acc'].append(tr_multiclass_acc)
        history['val_loss'].append(va_loss)
        history['val_binary_acc'].append(va_binary_acc)
        history['val_multiclass_acc'].append(va_multiclass_acc)
        history['test_loss'].append(te_loss)
        history['test_binary_acc'].append(te_binary_acc)
        history['test_multiclass_acc'].append(te_multiclass_acc)
        
        scheduler.step()
        
        print(f"Epoch {epoch+1}: binary_acc={va_binary_acc:.4f} multiclass_acc={va_multiclass_acc:.4f}")
        
        
        os.makedirs('results', exist_ok=True)
        log_row = {
            'phase': 'phase2_full_precision',
            'epoch': epoch + 1,
            'train_loss': tr_loss,
            'train_binary_acc': tr_binary_acc,
            'train_multiclass_acc': tr_multiclass_acc,
            'val_loss': va_loss,
            'val_binary_acc': va_binary_acc,
            'val_multiclass_acc': va_multiclass_acc,
            'test_loss': te_loss,
            'test_binary_acc': te_binary_acc,
            'test_multiclass_acc': te_multiclass_acc,
        }
        log_path = os.path.join('results', 'phase2_log.csv')
        pd.DataFrame([log_row]).to_csv(log_path, mode='a', header=not os.path.exists(log_path), index=False)
        
        if va_multiclass_acc > best_acc:
            best_acc = va_multiclass_acc
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    
    if USE_QAT:
        print("\nPhase 2: Quantization-Aware Training...")
        model.train()
        model = prepare_qat(model)
        
        optimizer_qat = optim.AdamW(model.parameters(), lr=QAT_LR, weight_decay=WEIGHT_DECAY)
        
        for epoch in range(QAT_EPOCHS):
            tr_loss, tr_binary_acc, tr_multiclass_acc = train_epoch(
                model, loaders['train'], criterion, optimizer_qat, epoch, is_qat=True)
            
            va_loss, va_binary_acc, va_multiclass_acc, _, _, _, _ = eval_epoch(
                model, loaders['val'], criterion, epoch, phase='Val')
            
            print(f"QAT Epoch {epoch+1}: binary_acc={va_binary_acc:.4f} multiclass_acc={va_multiclass_acc:.4f}")
    
    
    if USE_PRUNING:
        print("\nPhase 3: Structured Pruning...")
        pruner = StructuredPruner(model, PRUNING_RATIO)
        pruner.apply_pruning()
        
        optimizer_prune = optim.AdamW(model.parameters(), lr=QAT_LR, weight_decay=WEIGHT_DECAY)
        
        for epoch in range(PRUNING_EPOCHS):
            tr_loss, tr_binary_acc, tr_multiclass_acc = train_epoch(
                model, loaders['train'], criterion, optimizer_prune, epoch)
            
            va_loss, va_binary_acc, va_multiclass_acc, _, _, _, _ = eval_epoch(
                model, loaders['val'], criterion, epoch, phase='Val')
            
            print(f"Pruning Epoch {epoch+1}: binary_acc={va_binary_acc:.4f} multiclass_acc={va_multiclass_acc:.4f}")
        
        pruner.remove_pruning()
    
    
    te_loss, te_binary_acc, te_multiclass_acc, te_binary_probs, te_multiclass_probs, te_binary_labels, te_multiclass_labels = eval_epoch(
        model, loaders['test'], criterion, -1, phase='Test')
    
    print(f"\nFinal Results:")
    print(f"Binary Accuracy: {te_binary_acc:.4f}")
    print(f"Multi-class Accuracy: {te_multiclass_acc:.4f}")
    
    
    os.makedirs('models', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_name': 'mobilenetv3_attention',
        'history': history,
        'test_results': {
            'binary_acc': te_binary_acc,
            'multiclass_acc': te_multiclass_acc,
            'binary_probs': te_binary_probs,
            'multiclass_probs': te_multiclass_probs,
            'binary_labels': te_binary_labels,
            'multiclass_labels': te_multiclass_labels
        }
    }, os.path.join('models', 'phase2_mobilenetv3_attention.pth'))
    
    return {
        'model_name': 'mobilenetv3_attention',
        'history': history,
        'test_results': {
            'binary_acc': te_binary_acc,
            'multiclass_acc': te_multiclass_acc,
            'binary_probs': te_binary_probs,
            'multiclass_probs': te_multiclass_probs,
            'binary_labels': te_binary_labels,
            'multiclass_labels': te_multiclass_labels
        }
    }


if __name__ == "__main__":
    print("Starting Phase 2: Lightweight Mobile Model Training")
    
    
    loaders, meta = make_loaders(dataset_path, class_names)
    
    
    results = train_phase2_model(loaders)
    
    print("Phase 2 training completed!")
    print(f"Final Binary Accuracy: {results['test_results']['binary_acc']:.4f}")
    print(f"Final Multi-class Accuracy: {results['test_results']['multiclass_acc']:.4f}")
