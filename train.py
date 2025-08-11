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

import torchvision.models as models
import torchvision.transforms as T
from PIL import Image, ImageOps, ImageEnhance

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, roc_auc_score
)

from tqdm import tqdm

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Optimized Hyperparameters based on research
IMG_SIZE = 384  # Larger images for better feature extraction
BATCH_SIZE = 16  # Smaller batch for better generalization
NUM_EPOCHS = 50  # More epochs for convergence
LR_HEAD = 1e-3   # Bump head LR for faster convergence
LR_BACKBONE = 1e-5  # Very low backbone learning rate
WEIGHT_DECAY = 5e-5  # Reduced weight decay
LABEL_SMOOTH = 0.1   # Increased label smoothing
HEAD_WARMUP_EPOCHS = 5  # Longer warmup before unfreezing backbone
USE_MIXUP = False  # Disable MixUp for clearer metrics
MIXUP_ALPHA = 0.2
USE_TTA = True
EARLY_STOP_PATIENCE = 10  # More patience
USE_CUTMIX = False  # Disable CutMix for stability
CUTMIX_ALPHA = 1.0
USE_AUTOAUGMENT = False  # Keep medical-safe minimal aug

import kagglehub

# Download latest version
path = kagglehub.dataset_download("ninadaithal/imagesoasis")

print("Path to dataset files:", path)

# Dataset discovery from KaggleHub-style layout or explicit path
EXPECTED_CLASSES = ['Non Demented', 'Very mild Dementia', 'Mild Dementia', 'Moderate Dementia']

def resolve_dataset_path(root_candidates):
    for root in root_candidates:
        if not root:
            continue
        candidates = [os.path.join(root, 'Data'), root]
        for cand in candidates:
            if os.path.isdir(cand):
                subdirs = [d for d in os.listdir(cand) if os.path.isdir(os.path.join(cand, d))]
                found = [d for d in subdirs if d in EXPECTED_CLASSES]
                if len(found) == len(EXPECTED_CLASSES):
                    return cand, found
                if len(found) > 0:
                    return cand, found
    return None, []

DATASET_ROOT = os.environ.get('ALZ_DATASET_ROOT', None)

common_roots = [
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

# Advanced data augmentation pipeline
def get_transforms(img_size=224, is_training=False):
    if is_training:
        # Medical-safe, modest augmentation to reduce train-val gap
        return T.Compose([
            T.Resize(int(img_size * 1.10)),
            T.CenterCrop(img_size),
            T.RandomHorizontalFlip(p=0.5),
            T.Grayscale(num_output_channels=3),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return T.Compose([
            T.Resize(int(img_size * 1.10)),
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

    # Improved class balancing
    num_classes = 4
    train_label_tensor = torch.tensor(train_ds.labels, dtype=torch.long)
    class_counts = torch.bincount(train_label_tensor, minlength=num_classes).float()
    
    # More balanced class weights
    class_weights = 1.0 / (class_counts + 1e-8)
    class_weights = class_weights / class_weights.sum() * num_classes
    print(f"Class weights: {class_weights.numpy()}")

    # Balanced sampling
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

loaders, meta = make_loaders(dataset_path, class_names)

# Enhanced model creation with better heads
def create_model(name: str, num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    if name == 'vgg16':
        m = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
        # Enhanced classifier head
        m.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 7 * 7, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )
        return m
    if name == 'vgg19':
        m = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 7 * 7, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )
        return m
    if name == 'resnet50':
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        # Enhanced head with attention
        m.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(m.fc.in_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )
        return m
    if name == 'resnet101':
        m = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2 if pretrained else None)
        m.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(m.fc.in_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )
        return m
    if name == 'resnet152':
        m = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V2 if pretrained else None)
        m.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(m.fc.in_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )
        return m
    if name == 'densenet121':
        m = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(m.classifier.in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        return m
    if name == 'densenet201':
        m = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(m.classifier.in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        return m
    if name == 'mobilenetv3_large':
        m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None)
        m.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(m.classifier[3].in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        return m
    if name == 'shufflenet_v2_x1_0':
        m = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1 if pretrained else None)
        m.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(m.fc.in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        return m
    raise ValueError(f"Unknown model: {name}")

ALL_MODELS = [
    'vgg16', 'vgg19',
    'resnet50', 'resnet101', 'resnet152',
    'densenet121', 'densenet201',
    'mobilenetv3_large', 'shufflenet_v2_x1_0'
]

# Advanced augmentation techniques
def cutmix_data(x, y, alpha=CUTMIX_ALPHA):
    if alpha <= 0:
        return x, y, 1.0
    
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    y_a, y_b = y, y[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, (y_a, y_b), lam

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def mixup_data(x, y, alpha=MIXUP_ALPHA):
    if alpha <= 0:
        return x, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, (y_a, y_b), lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Advanced training with progressive unfreezing
def train_epoch(model, loader, criterion, optimizer, epoch):
    use_mixup = USE_MIXUP and epoch > HEAD_WARMUP_EPOCHS
    use_cutmix = USE_CUTMIX and epoch > HEAD_WARMUP_EPOCHS
    
    model.train()
    scaler = GradScaler(enabled=torch.cuda.is_available())
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc=f"Train {epoch+1}")
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        
        with autocast('cuda', enabled=torch.cuda.is_available()):
            if use_cutmix and random.random() > 0.5:
                images, (ya, yb), lam = cutmix_data(images, labels, CUTMIX_ALPHA)
                outputs = model(images)
                loss = mixup_criterion(criterion, outputs, ya, yb, lam)
            elif use_mixup and random.random() > 0.5:
                images, (ya, yb), lam = mixup_data(images, labels, MIXUP_ALPHA)
                outputs = model(images)
                loss = mixup_criterion(criterion, outputs, ya, yb, lam)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
        pbar.set_postfix(loss=running_loss/total, acc=correct/total)
    
    return running_loss/total, correct/total

def eval_epoch(model, loader, criterion, epoch, phase="Val"):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc=f"{phase} {epoch+1}")
    all_probs, all_labels, all_preds = [], [], []
    
    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            with autocast('cuda', enabled=torch.cuda.is_available()):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            probs = F.softmax(outputs, dim=1)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
            
            all_probs.append(probs.detach().cpu())
            all_labels.append(labels.detach().cpu())
            all_preds.append(preds.detach().cpu())
            pbar.set_postfix(loss=running_loss/total, acc=correct/total)
    
    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()
    return running_loss/total, correct/total, all_probs, all_labels, all_preds

# Advanced training with cosine annealing and progressive unfreezing
def train_model(name, loaders, num_epochs=NUM_EPOCHS, lr=LR_BACKBONE, weight_decay=WEIGHT_DECAY):
    print(f"\n==== Training {name} ====")
    model = create_model(name, num_classes=4, pretrained=True).to(device)
    class_weights = loaders['class_weights'].to(device)
    # Use sampler for imbalance; avoid double-compensation by removing class weights from loss
    base_criterion = nn.CrossEntropyLoss(weight=None, label_smoothing=LABEL_SMOOTH)

    # Progressive unfreezing setup
    head_params, backbone_params = [], []
    for n, p in model.named_parameters():
        if any(k in n for k in ['fc', 'classifier']):
            head_params.append(p)
        else:
            backbone_params.append(p)
    
    # Advanced optimizer with different learning rates
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': lr, 'weight_decay': weight_decay},
        {'params': head_params, 'lr': LR_HEAD, 'weight_decay': weight_decay}
    ])
    
    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-7
    )

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        # Progressive unfreezing
        if epoch < HEAD_WARMUP_EPOCHS:
            for p in backbone_params:
                p.requires_grad = False
        else:
            for p in backbone_params:
                p.requires_grad = True

        tr_loss, tr_acc = train_epoch(model, loaders['train'], base_criterion, optimizer, epoch)
        va_loss, va_acc, va_probs, va_labels, va_preds = eval_epoch(model, loaders['val'], base_criterion, epoch, phase='Val')
        
        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(va_loss)
        history['val_acc'].append(va_acc)
        
        scheduler.step()
        
        # Metrics
        va_precision = precision_score(va_labels, va_preds, average='macro', zero_division=0)
        va_recall = recall_score(va_labels, va_preds, average='macro', zero_division=0)
        va_f1 = f1_score(va_labels, va_preds, average='macro', zero_division=0)
        
        print(f"Epoch {epoch+1}: train_acc={tr_acc:.4f} val_acc={va_acc:.4f} val_f1={va_f1:.4f} val_prec={va_precision:.4f} val_rec={va_recall:.4f}")

        # per-epoch CSV logging
        os.makedirs('results', exist_ok=True)
        backbone_lr = optimizer.param_groups[0]['lr']
        head_lr = optimizer.param_groups[1]['lr']
        log_row = {
            'model_name': name,
            'epoch': epoch + 1,
            'train_loss': tr_loss,
            'train_acc': tr_acc,
            'val_loss': va_loss,
            'val_acc': va_acc,
            'val_f1_macro': va_f1,
            'val_precision_macro': va_precision,
            'val_recall_macro': va_recall,
            'lr_backbone': backbone_lr,
            'lr_head': head_lr,
            'img_size': IMG_SIZE,
            'batch_size': BATCH_SIZE,
            'label_smooth': LABEL_SMOOTH,
            'warmup_epochs': HEAD_WARMUP_EPOCHS,
            'mixup': USE_MIXUP,
            'cutmix': USE_CUTMIX
        }
        log_path = os.path.join('results', 'epoch_log.csv')
        pd.DataFrame([log_row]).to_csv(log_path, mode='a', header=not os.path.exists(log_path), index=False)

        if va_acc > best_acc:
            best_acc = va_acc
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch+1} (no val acc improvement for {EARLY_STOP_PATIENCE} epochs)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    
    te_loss, te_acc, te_probs, te_labels, te_preds = eval_epoch(model, loaders['test'], base_criterion, -1, phase='Test')

    if USE_TTA:
        model.eval()
        with torch.no_grad():
            tta_probs = []
            for images, _ in tqdm(loaders['test'], desc='TTA'):
                images = images.to(device)
                p1 = F.softmax(model(images), dim=1)
                p2 = F.softmax(model(torch.flip(images, dims=[3])), dim=1)
                p3 = F.softmax(model(torch.flip(images, dims=[2])), dim=1)
                tta_probs.append(((p1 + p2 + p3) / 3).cpu())
            te_probs = torch.cat(tta_probs).numpy()

    try:
        y_true_bin = F.one_hot(torch.tensor(te_labels), num_classes=4).numpy()
        auc_macro = roc_auc_score(y_true_bin, te_probs, average='macro', multi_class='ovr')
    except Exception:
        auc_macro = 0.0

    # detailed test metrics
    te_precision = precision_score(te_labels, te_preds, average='macro', zero_division=0)
    te_recall = recall_score(te_labels, te_preds, average='macro', zero_division=0)
    te_f1 = f1_score(te_labels, te_preds, average='macro', zero_division=0)

    print(f"Test: acc={te_acc:.4f} f1={te_f1:.4f} prec={te_precision:.4f} rec={te_recall:.4f} auc_macro={auc_macro:.4f}")
    print(f"Model: {name} | ImgSize: {IMG_SIZE} | Batch: {BATCH_SIZE} | Epochs: {NUM_EPOCHS} | LR(head/backbone): {LR_HEAD}/{LR_BACKBONE} | Warmup: {HEAD_WARMUP_EPOCHS}")

    return {
        'model_name': name,
        'best_val_acc': best_acc,
        'test_acc': te_acc,
        'test_f1_macro': te_f1,
        'test_precision_macro': te_precision,
        'test_recall_macro': te_recall,
        'test_auc_macro': auc_macro,
        'config': {
            'img_size': IMG_SIZE,
            'batch_size': BATCH_SIZE,
            'num_epochs': NUM_EPOCHS,
            'lr_head': LR_HEAD,
            'lr_backbone': LR_BACKBONE,
            'weight_decay': WEIGHT_DECAY,
            'label_smooth': LABEL_SMOOTH,
            'warmup_epochs': HEAD_WARMUP_EPOCHS,
            'mixup': USE_MIXUP,
            'cutmix': USE_CUTMIX
        },
        'history': history,
        'state_dict': best_state
    }

# Main training loop
results = []
os.makedirs('models', exist_ok=True)

# Start with best performing models first
priority_models = [
    'resnet50', 'resnet101', 'densenet121', 'densenet201',
    'vgg16', 'vgg19', 'resnet152',
    'mobilenetv3_large', 'shufflenet_v2_x1_0'
]

for name in priority_models:
    try:
        print(f"Training {name}")
        res = train_model(name, loaders)
        results.append(res)
        torch.save({
            'model_name': name,
            'state_dict': res['state_dict'],
            'history': res['history'],
            'meta': meta,
            'results': {k: res[k] for k in ['best_val_acc', 'test_acc', 'test_auc_macro']}
        }, os.path.join('models', f'{name}_finetuned.pth'))
    except Exception as e:
        print(f"Error training {name}: {e}")

if results:
    df = pd.DataFrame([
        {
            'model': r['model_name'],
            'val_acc': r['best_val_acc'],
            'test_acc': r['test_acc'],
            'test_f1_macro': r.get('test_f1_macro', None),
            'test_precision_macro': r.get('test_precision_macro', None),
            'test_recall_macro': r.get('test_recall_macro', None),
            'test_auc_macro': r['test_auc_macro'],
            'img_size': r['config']['img_size'],
            'batch_size': r['config']['batch_size'],
            'epochs': r['config']['num_epochs'],
            'lr_head': r['config']['lr_head'],
            'lr_backbone': r['config']['lr_backbone'],
            'weight_decay': r['config']['weight_decay'],
            'label_smooth': r['config']['label_smooth'],
            'warmup_epochs': r['config']['warmup_epochs'],
            'mixup': r['config']['mixup'],
            'cutmix': r['config']['cutmix']
        }
        for r in results
    ])
    print(df.sort_values('test_acc', ascending=False))
    df.to_csv('models/summary.csv', index=False)
else:
    print("No models trained.")
