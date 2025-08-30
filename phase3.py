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
from PIL import Image

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression

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

# Phase 3 Hyperparameters
IMG_SIZE = 224
BATCH_SIZE = 16
NUM_EPOCHS = 20
LR_HEAD = 1e-3
LR_BACKBONE = 1e-5
WEIGHT_DECAY = 1e-4
LABEL_SMOOTH = 0.1
HEAD_WARMUP_EPOCHS = 3
EARLY_STOP_PATIENCE = 8

# Ensemble Settings
N_FOLDS = 5
USE_CROSS_VALIDATION = True

import kagglehub

# Download dataset
path = kagglehub.dataset_download("ninadaithal/imagesoasis")
print("Path to dataset files:", path)

# Dataset discovery
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
]

dataset_path, class_names = resolve_dataset_path(common_roots)
if not dataset_path:
    raise FileNotFoundError("Could not resolve dataset path.")

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

# Vision Transformer (ViT)
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4., dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=4,
                 embed_dim=384, depth=12, num_heads=6, mlp_ratio=4., dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.n_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        x = self.head(x[:, 0])
        return x

# MobileNetV3 with Attention
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
            CBAM(16, 16), CBAM(24, 16), CBAM(40, 16),
            CBAM(80, 16), CBAM(112, 16), CBAM(160, 16)
        ])
        
        original_in_features = self.backbone.classifier[3].in_features
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(original_in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.backbone.features[0](x)
        
        attention_idx = 0
        for i, layer in enumerate(self.backbone.features[1:]):
            x = layer(x)
            if i in [2, 5, 9, 13, 17, 21]:
                if attention_idx < len(self.attention_modules):
                    x = self.attention_modules[attention_idx](x)
                    attention_idx += 1
        
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Ensemble Fusion Methods
class SimpleAveraging:
    def __init__(self):
        self.name = "Simple Averaging"
    
    def fit(self, predictions_list, labels):
        pass
    
    def predict(self, predictions_list):
        return np.mean(predictions_list, axis=0)

class WeightedAveraging:
    def __init__(self):
        self.name = "Weighted Averaging"
        self.weights = None
    
    def fit(self, predictions_list, labels):
        n_models = len(predictions_list)
        self.weights = np.ones(n_models) / n_models
        
        best_score = 0
        best_weights = self.weights.copy()
        
        for _ in range(100):
            weights = np.random.dirichlet(np.ones(n_models))
            weighted_pred = np.zeros_like(predictions_list[0])
            
            for i, pred in enumerate(predictions_list):
                weighted_pred += weights[i] * pred
            
            pred_labels = np.argmax(weighted_pred, axis=1)
            score = accuracy_score(labels, pred_labels)
            
            if score > best_score:
                best_score = score
                best_weights = weights
        
        self.weights = best_weights
    
    def predict(self, predictions_list):
        weighted_pred = np.zeros_like(predictions_list[0])
        for i, pred in enumerate(predictions_list):
            weighted_pred += self.weights[i] * pred
        return weighted_pred

class Stacking:
    def __init__(self, meta_learner='logistic'):
        self.name = "Stacking"
        self.meta_learner = meta_learner
        self.model = None
    
    def fit(self, predictions_list, labels):
        meta_features = np.concatenate(predictions_list, axis=1)
        self.model = LogisticRegression(random_state=SEED, max_iter=1000)
        self.model.fit(meta_features, labels)
    
    def predict(self, predictions_list):
        meta_features = np.concatenate(predictions_list, axis=1)
        return self.model.predict_proba(meta_features)

# Data transforms
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

    # Class balancing
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

# Training functions
def train_epoch(model, loader, criterion, optimizer, epoch):
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

# Model training function
def train_model(model_name, model, loaders, num_epochs=NUM_EPOCHS):
    print(f"\n==== Training {model_name} ====")
    
    class_weights = loaders['class_weights'].to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTH)
    
    # Progressive unfreezing
    head_params, backbone_params = [], []
    for n, p in model.named_parameters():
        if any(k in n for k in ['fc', 'classifier', 'head']):
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
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0.0
    best_state = None
    epochs_no_improve = 0
    
    for epoch in range(num_epochs):
        if epoch < HEAD_WARMUP_EPOCHS:
            for p in backbone_params:
                p.requires_grad = False
        else:
            for p in backbone_params:
                p.requires_grad = True
        
        tr_loss, tr_acc = train_epoch(model, loaders['train'], criterion, optimizer, epoch)
        va_loss, va_acc, va_probs, va_labels, va_preds = eval_epoch(
            model, loaders['val'], criterion, epoch, phase='Val')
        
        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(va_loss)
        history['val_acc'].append(va_acc)
        
        scheduler.step()
        
        print(f"Epoch {epoch+1}: train_acc={tr_acc:.4f} val_acc={va_acc:.4f}")
        
        if va_acc > best_acc:
            best_acc = va_acc
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    # Final evaluation
    te_loss, te_acc, te_probs, te_labels, te_preds = eval_epoch(
        model, loaders['test'], criterion, -1, phase='Test')
    
    print(f"Final Test Accuracy: {te_acc:.4f}")
    
    return {
        'model_name': model_name,
        'best_val_acc': best_acc,
        'test_acc': te_acc,
        'test_probs': te_probs,
        'test_labels': te_labels,
        'test_preds': te_preds,
        'history': history,
        'state_dict': best_state
    }

# Cross-validation training
def train_with_cv(model_name, model_class, loaders, num_epochs=NUM_EPOCHS):
    print(f"\n==== Cross-Validation Training for {model_name} ====")
    
    train_dataset = loaders['train'].dataset
    val_dataset = loaders['val'].dataset
    
    all_train_paths = train_dataset.image_paths + val_dataset.image_paths
    all_train_labels = train_dataset.labels + val_dataset.labels
    
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    cv_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(all_train_paths)):
        print(f"\nFold {fold + 1}/{N_FOLDS}")
        
        fold_train_paths = [all_train_paths[i] for i in train_idx]
        fold_train_labels = [all_train_labels[i] for i in train_idx]
        fold_val_paths = [all_train_paths[i] for i in val_idx]
        fold_val_labels = [all_train_labels[i] for i in val_idx]
        
        train_tf = get_transforms(IMG_SIZE, True)
        eval_tf = get_transforms(IMG_SIZE, False)
        
        fold_train_ds = AlzheimerDataset(fold_train_paths, fold_train_labels, train_tf)
        fold_val_ds = AlzheimerDataset(fold_val_paths, fold_val_labels, eval_tf)
        
        fold_loaders = {
            'train': DataLoader(fold_train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True),
            'val': DataLoader(fold_val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True),
            'test': loaders['test'],
            'class_weights': loaders['class_weights']
        }
        
        model = model_class().to(device)
        result = train_model(model_name, model, fold_loaders, num_epochs)
        cv_results.append(result)
        
        torch.save({
            'fold': fold,
            'model_name': model_name,
            'test_probs': result['test_probs'],
            'test_labels': result['test_labels'],
            'test_preds': result['test_preds']
        }, os.path.join('models', f'{model_name}_fold_{fold}.pth'))
    
    return cv_results

# Ensemble training and evaluation
def train_ensemble_models(loaders):
    print("==== Phase 3: Ensemble Training ====")
    
    models_to_train = {
        'mobilenetv3_attention': MobileNetV3WithAttention,
        'vit': lambda: VisionTransformer(num_classes=4, embed_dim=384, depth=12, num_heads=6),
    }
    
    all_results = {}
    
    for model_name, model_class in models_to_train.items():
        print(f"\nTraining {model_name}...")
        
        if USE_CROSS_VALIDATION:
            results = train_with_cv(model_name, model_class, loaders)
        else:
            model = model_class().to(device)
            results = [train_model(model_name, model, loaders)]
        
        all_results[model_name] = results
    
    return all_results

# Ensemble fusion and evaluation
def evaluate_ensemble(all_results, loaders):
    print("\n==== Ensemble Evaluation ====")
    
    # Load all predictions
    all_predictions = {}
    for model_name, results in all_results.items():
        if USE_CROSS_VALIDATION:
            fold_probs = []
            for result in results:
                fold_probs.append(result['test_probs'])
            all_predictions[model_name] = np.mean(fold_probs, axis=0)
        else:
            all_predictions[model_name] = results[0]['test_probs']
    
    test_labels = all_results[list(all_results.keys())[0]][0]['test_labels']
    
    # Ensemble methods
    ensemble_methods = {
        'simple_avg': SimpleAveraging(),
        'weighted_avg': WeightedAveraging(),
        'stacking': Stacking(meta_learner='logistic')
    }
    
    ensemble_results = {}
    
    for method_name, method in ensemble_methods.items():
        print(f"\nEvaluating {method_name}...")
        
        predictions_list = list(all_predictions.values())
        
        if method_name == 'stacking':
            split_idx = len(test_labels) // 2
            train_preds = [pred[:split_idx] for pred in predictions_list]
            train_labels = test_labels[:split_idx]
            test_preds = [pred[split_idx:] for pred in predictions_list]
            test_labels_eval = test_labels[split_idx:]
            
            method.fit(train_preds, train_labels)
            ensemble_probs = method.predict(test_preds)
            final_labels = test_labels_eval
        else:
            method.fit(predictions_list, test_labels)
            ensemble_probs = method.predict(predictions_list)
            final_labels = test_labels
        
        ensemble_preds = np.argmax(ensemble_probs, axis=1)
        accuracy = accuracy_score(final_labels, ensemble_preds)
        f1_macro = f1_score(final_labels, ensemble_preds, average='macro')
        precision_macro = precision_score(final_labels, ensemble_preds, average='macro')
        recall_macro = recall_score(final_labels, ensemble_preds, average='macro')
        
        ensemble_results[method_name] = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'probs': ensemble_probs,
            'preds': ensemble_preds,
            'labels': final_labels
        }
        
        print(f"{method_name}: Accuracy={accuracy:.4f}, F1={f1_macro:.4f}")
    
    return ensemble_results

# Main execution
if __name__ == "__main__":
    print("Starting Phase 3: Ensemble Training")
    
    # Load data
    loaders, meta = make_loaders(dataset_path, class_names)
    
    # Train ensemble models
    all_results = train_ensemble_models(loaders)
    
    # Evaluate ensemble
    ensemble_results = evaluate_ensemble(all_results, loaders)
    
    # Save results
    os.makedirs('results', exist_ok=True)
    
    ensemble_summary = []
    for method_name, result in ensemble_results.items():
        ensemble_summary.append({
            'method': method_name,
            'accuracy': result['accuracy'],
            'f1_macro': result['f1_macro'],
            'precision_macro': result['precision_macro'],
            'recall_macro': result['recall_macro']
        })
    
    df_ensemble = pd.DataFrame(ensemble_summary)
    df_ensemble.to_csv('results/phase3_ensemble_results.csv', index=False)
    
    print("\nPhase 3 completed!")
    print("\nEnsemble Results Summary:")
    print(df_ensemble)
    
    torch.save({
        'all_results': all_results,
        'ensemble_results': ensemble_results,
        'meta': meta
    }, os.path.join('models', 'phase3_ensemble_complete.pth'))
