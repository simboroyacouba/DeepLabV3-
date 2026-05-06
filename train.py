"""
Entraînement DeepLabV3+ pour segmentation des toitures cadastrales
Dataset: Images aériennes annotées avec CVAT (format COCO)

Modes d'entraînement:
  simple    : DeepLabV3+ standard, hyperparamètres fixes
  attention : DeepLabV3+ + CBAM injecté dans la tête de classification (après ASPP)
  optimize  : DeepLabV3+ standard + recherche bayésienne des hyperparamètres (Optuna)
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import argparse
from torch.utils.data import DataLoader, random_split
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import torchvision.transforms.functional as TF
from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask_utils
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from datetime import datetime, timedelta
import random
import yaml
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

def load_classes(yaml_path=None):
    path = yaml_path or os.getenv("CLASSES_FILE", "classes.yaml")
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)['classes']

OPTUNA_CONFIG = {
    "n_trials": 30,
    "n_epochs_per_trial": 5,
    "study_name": "deeplab_cadastral",
    "output_dir": "./optuna_output",
}

CONFIG = {
    "images_dir":       os.getenv("SEGMENTATION_DATASET_IMAGES_DIR"),
    "annotations_file": os.getenv("SEGMENTATION_DATASET_ANNOTATIONS_FILE"),
    "classes_file":     os.getenv("CLASSES_FILE", "classes.yaml"),
    "output_dir":       "./output",
    "classes":          load_classes(),
    "backbone":         "resnet50",   # resnet50 ou resnet101
    "pretrained":       True,
    "num_epochs":       25,
    "batch_size":       2,
    "learning_rate":    0.005,
    "momentum":         0.9,
    "weight_decay":     0.0005,
    "lr_step_size":     8,
    "lr_gamma":         0.1,
    "image_size":       512,
    "train_split":      0.85,
    "num_workers":      0,
    "save_every":       5,
    # Paramètres CBAM par défaut (mode attention uniquement)
    "cbam_reduction":   16,
    "cbam_kernel_size": 7,
}


# =============================================================================
# UTILITAIRES TEMPS
# =============================================================================

def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    else:
        return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m {int(seconds % 60)}s"


class TrainingTimer:
    def __init__(self, num_epochs):
        self.num_epochs  = num_epochs
        self.start_time  = None
        self.epoch_times = []
        self.epoch_start = None

    def start_training(self):
        self.start_time = time.time()
        self.training_start_datetime = datetime.now()

    def start_epoch(self):
        self.epoch_start = time.time()

    def end_epoch(self, epoch):
        epoch_time = time.time() - self.epoch_start
        self.epoch_times.append(epoch_time)
        total_elapsed     = time.time() - self.start_time
        avg_epoch_time    = np.mean(self.epoch_times)
        estimated_remaining = avg_epoch_time * (self.num_epochs - (epoch + 1))
        return {
            'epoch_time':           epoch_time,
            'total_elapsed':        total_elapsed,
            'avg_epoch_time':       avg_epoch_time,
            'estimated_remaining':  estimated_remaining,
            'eta':                  datetime.now() + timedelta(seconds=estimated_remaining),
            'progress_percent':     ((epoch + 1) / self.num_epochs) * 100,
        }

    def get_final_stats(self):
        total_time = time.time() - self.start_time
        return {
            'total_time':                  total_time,
            'total_time_formatted':        format_time(total_time),
            'avg_epoch_time':              np.mean(self.epoch_times),
            'avg_epoch_time_formatted':    format_time(np.mean(self.epoch_times)),
            'min_epoch_time':              np.min(self.epoch_times),
            'min_epoch_time_formatted':    format_time(np.min(self.epoch_times)),
            'max_epoch_time':              np.max(self.epoch_times),
            'max_epoch_time_formatted':    format_time(np.max(self.epoch_times)),
            'std_epoch_time':              np.std(self.epoch_times),
            'epoch_times':                 self.epoch_times,
            'start_datetime':              self.training_start_datetime.strftime("%Y-%m-%d %H:%M:%S"),
            'end_datetime':                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }


# =============================================================================
# AUGMENTATION PAR CLASSE
# =============================================================================

def load_aug_coefficients(classes):
    """
    Charge les coefficients d'augmentation depuis les variables d'environnement.
    Format index  : CLASS_AUG_1=2  (1 = première classe hors __background__)
    Format nom    : CLASS_AUG_TOITURE_TOLE_BAC=2
    Valeur par défaut: 1 (aucune augmentation).
    """
    real_classes = [c for c in classes if c != '__background__']
    coeffs = {}
    for i, cls in enumerate(real_classes, 1):
        env_idx  = f"CLASS_AUG_{i}"
        env_name = "CLASS_AUG_" + cls.upper().replace(' ', '_').replace('-', '_')
        raw = os.getenv(env_name) or os.getenv(env_idx, "1")
        coeffs[cls] = max(1, int(raw))
    return coeffs


def _count_class_stats(coco, image_ids, cat_id_to_name):
    stats = {name: {'images': set(), 'annotations': 0} for name in cat_id_to_name.values()}
    for img_id in image_ids:
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=[img_id])):
            if ann.get('iscrowd', 0):
                continue
            cid = ann['category_id']
            if cid in cat_id_to_name:
                name = cat_id_to_name[cid]
                stats[name]['images'].add(img_id)
                stats[name]['annotations'] += 1
    return {name: {'images': len(s['images']), 'annotations': s['annotations']}
            for name, s in stats.items()}


def oversample_by_class(image_ids, coco, cat_id_to_name, aug_coeffs):
    """Duplique les IDs d'images selon le coefficient maximal de leurs classes."""
    img_max_coeff = {}
    for img_id in image_ids:
        max_c = 1
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=[img_id])):
            if ann.get('iscrowd', 0):
                continue
            cid = ann['category_id']
            if cid in cat_id_to_name:
                max_c = max(max_c, aug_coeffs.get(cat_id_to_name[cid], 1))
        img_max_coeff[img_id] = max_c
    augmented = []
    for img_id in image_ids:
        augmented.extend([img_id] * img_max_coeff[img_id])
    return augmented, img_max_coeff


def print_augmentation_report(coco, train_ids_before, train_ids_after,
                               cat_id_to_name, aug_coeffs):
    before = _count_class_stats(coco, train_ids_before, cat_id_to_name)
    after  = _count_class_stats(coco, train_ids_after,  cat_id_to_name)
    print(f"\n{'='*70}")
    print(f"   RAPPORT D'AUGMENTATION DES DONNEES D'ENTRAINEMENT")
    print(f"{'='*70}")
    print(f"\n   AVANT AUGMENTATION  ({len(set(train_ids_before))} images uniques en train)")
    print(f"   {'─'*65}")
    print(f"   {'Classe':<38} {'Coeff':>5}  {'Images':>7}  {'Annot.':>7}")
    print(f"   {'─'*65}")
    total_ann_b = 0
    for cls_name, s in before.items():
        coeff  = aug_coeffs.get(cls_name, 1)
        marker = "  *" if coeff > 1 else ""
        print(f"   {cls_name:<38} x{coeff:>4}  {s['images']:>7}  {s['annotations']:>7}{marker}")
        total_ann_b += s['annotations']
    print(f"   {'─'*65}")
    print(f"   {'TOTAL':<38}       {len(set(train_ids_before)):>7}  {total_ann_b:>7}")
    print(f"\n   APRES AUGMENTATION  ({len(train_ids_after)} samples d'entrainement)")
    print(f"   {'─'*65}")
    print(f"   {'Classe':<38} {'Delta':>5}  {'Images':>7}  {'Annot.':>7}")
    print(f"   {'─'*65}")
    total_ann_a = 0
    for cls_name, sa in after.items():
        sb    = before[cls_name]
        delta = f"+{sa['images'] - sb['images']}" if sa['images'] != sb['images'] else "  ="
        print(f"   {cls_name:<38} {delta:>5}  {sa['images']:>7}  {sa['annotations']:>7}")
        total_ann_a += sa['annotations']
    print(f"   {'─'*65}")
    print(f"   {'TOTAL (avec duplicats)':<38}       {len(train_ids_after):>7}  {total_ann_a:>7}")
    ratio = len(train_ids_after) / max(len(train_ids_before), 1)
    print(f"\n   Ratio d'augmentation global: x{ratio:.2f} samples")
    print(f"{'='*70}\n")


# =============================================================================
# DATASET
# =============================================================================

class DeepLabDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, annotations_file, image_size=512, transforms=None,
                 image_ids=None):
        self.images_dir = images_dir
        self.image_size = image_size
        self.transforms = transforms
        self.coco       = COCO(annotations_file)
        self.image_ids  = list(image_ids) if image_ids is not None else list(self.coco.imgs.keys())
        self.cat_ids    = self.coco.getCatIds()
        self.cat_mapping = {cat_id: idx + 1 for idx, cat_id in enumerate(self.cat_ids)}
        print(f"Dataset chargé: {len(self.image_ids)} images")
        print(f"Catégories: {[self.coco.cats[c]['name'] for c in self.cat_ids]}")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id   = self.image_ids[idx]
        img_info = self.coco.imgs[img_id]
        image    = Image.open(os.path.join(self.images_dir, img_info['file_name'])).convert("RGB")
        mask     = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
        for ann in self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id)):
            if ann.get('iscrowd', 0):
                continue
            class_id = self.cat_mapping[ann['category_id']]
            if 'segmentation' in ann:
                if isinstance(ann['segmentation'], list):
                    rles = coco_mask_utils.frPyObjects(
                        ann['segmentation'], img_info['height'], img_info['width']
                    )
                    instance_mask = coco_mask_utils.decode(coco_mask_utils.merge(rles))
                else:
                    instance_mask = coco_mask_utils.decode(ann['segmentation'])
                mask[instance_mask > 0] = class_id
        image    = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask_pil = Image.fromarray(mask).resize((self.image_size, self.image_size), Image.NEAREST)
        if self.transforms is not None:
            image, mask_pil = self.transforms(image, mask_pil)
        image_t = TF.normalize(TF.to_tensor(image),
                               mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        mask_t  = torch.as_tensor(np.array(mask_pil), dtype=torch.long)
        return image_t, mask_t


# =============================================================================
# TRANSFORMATIONS
# =============================================================================

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, mask):
        if random.random() < self.prob:
            image = TF.hflip(image)
            mask  = TF.hflip(mask)
        return image, mask


class RandomVerticalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, mask):
        if random.random() < self.prob:
            image = TF.vflip(image)
            mask  = TF.vflip(mask)
        return image, mask


def get_transforms(train=True):
    transforms = []
    if train:
        transforms.append(RandomHorizontalFlip(0.5))
        transforms.append(RandomVerticalFlip(0.5))
    return Compose(transforms) if transforms else None


# =============================================================================
# MÉCANISME D'ATTENTION (CBAM) — utilisé uniquement en mode "attention"
# =============================================================================

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(),
            nn.Linear(mid, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c = x.shape[:2]
        avg   = self.fc(self.avg_pool(x).view(b, c))
        mx    = self.fc(self.max_pool(x).view(b, c))
        return x * self.sigmoid(avg + mx).view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv    = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg   = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        return self.spatial_att(self.channel_att(x))


# =============================================================================
# MODÈLE
# =============================================================================

def _build_base_model(backbone="resnet50", pretrained=True):
    if backbone == "resnet50":
        model      = deeplabv3_resnet50(weights="DEFAULT" if pretrained else None)
        in_channels = 2048
    elif backbone == "resnet101":
        model      = deeplabv3_resnet101(weights="DEFAULT" if pretrained else None)
        in_channels = 2048
    else:
        raise ValueError(f"Backbone inconnu: {backbone}")
    return model, in_channels


def _replace_heads(model, in_channels, num_classes):
    model.classifier = DeepLabHead(in_channels, num_classes)
    if model.aux_classifier is not None:
        model.aux_classifier = nn.Sequential(
            nn.Conv2d(1024, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1),
        )


def get_model_simple(num_classes, backbone="resnet50", pretrained=True):
    """DeepLabV3+ standard."""
    model, in_channels = _build_base_model(backbone, pretrained)
    _replace_heads(model, in_channels, num_classes)
    return model


def get_model_attention(num_classes, backbone="resnet50", pretrained=True,
                        cbam_reduction=16, cbam_kernel_size=7):
    """
    DeepLabV3+ avec CBAM(256) injecté dans la tête principale,
    entre le bloc Conv+BN+ReLU et la conv de classification finale.

    Structure DeepLabHead: ASPP → Conv → BN → ReLU → [CBAM] → Conv(num_classes)
    """
    model, in_channels = _build_base_model(backbone, pretrained)
    _replace_heads(model, in_channels, num_classes)

    # Décomposer la tête et insérer CBAM avant la dernière conv
    layers = list(model.classifier.children())
    # layers[0]=ASPP, [1]=Conv2d(256,256), [2]=BN, [3]=ReLU, [4]=Conv2d(256,num_classes)
    model.classifier = nn.Sequential(
        *layers[:4],
        CBAM(256, cbam_reduction, cbam_kernel_size),
        layers[4],
    )
    return model


# =============================================================================
# OPTIMISATION BAYÉSIENNE (OPTUNA) — mode "optimize" uniquement
# =============================================================================

def _run_optimization(device, train_loader, val_loader, num_classes):
    """Recherche bayésienne des hyperparamètres via Optuna (sans CBAM)."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        lr           = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay",  1e-5, 1e-3, log=True)
        momentum     = trial.suggest_float("momentum",      0.80, 0.99)
        lr_step_size = trial.suggest_int  ("lr_step_size",  3,    15)

        model     = get_model_simple(num_classes, CONFIG["backbone"], CONFIG["pretrained"])
        model.to(device)
        params    = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=0.1)

        best_val = float('inf')
        for epoch in range(OPTUNA_CONFIG["n_epochs_per_trial"]):
            train_one_epoch(model, optimizer, train_loader, device, epoch)
            val_loss = _evaluate(model, val_loader, device)
            scheduler.step()
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            best_val = min(best_val, val_loss)
        return best_val

    os.makedirs(OPTUNA_CONFIG["output_dir"], exist_ok=True)
    study = optuna.create_study(
        direction="minimize",
        study_name=OPTUNA_CONFIG["study_name"],
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
    )

    print(f"\n{'=' * 70}")
    print(f"   OPTIMISATION BAYESIENNE — {OPTUNA_CONFIG['n_trials']} essais")
    print(f"   {OPTUNA_CONFIG['n_epochs_per_trial']} epochs/essai | sampler: TPE | pruner: Median")
    print(f"{'=' * 70}\n")

    study.optimize(objective, n_trials=OPTUNA_CONFIG["n_trials"], show_progress_bar=True)

    best = study.best_trial
    print(f"\n{'=' * 70}")
    print(f"   MEILLEUR ESSAI #{best.number}  —  val_loss: {best.value:.4f}")
    print(f"{'=' * 70}")
    for k, v in best.params.items():
        print(f"   {k}: {v}")

    report = {
        "best_trial": best.number, "best_val_loss": best.value, "best_params": best.params,
        "all_trials": [
            {"number": t.number, "value": t.value, "params": t.params, "state": str(t.state)}
            for t in study.trials
        ],
    }
    with open(os.path.join(OPTUNA_CONFIG["output_dir"], "optuna_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        values = [t.value for t in study.trials if t.value is not None]
        axes[0].plot(values, marker='o', linewidth=1.5)
        axes[0].set_xlabel("Essai"); axes[0].set_ylabel("Val Loss")
        axes[0].set_title("Historique Optuna"); axes[0].grid(True, alpha=0.3)
        importances = optuna.importance.get_param_importances(study)
        axes[1].barh(list(importances.keys()), list(importances.values()))
        axes[1].set_xlabel("Importance"); axes[1].set_title("Importance des hyperparametres")
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OPTUNA_CONFIG["output_dir"], "optuna_results.png"), dpi=150)
        plt.close()
    except Exception:
        pass

    return best.params


# =============================================================================
# ENTRAÎNEMENT
# =============================================================================

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    criterion  = nn.CrossEntropyLoss(ignore_index=255)
    total_loss = 0
    pbar       = tqdm(data_loader, desc=f"Epoch {epoch+1}")
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)
        outputs       = model(images)
        loss          = criterion(outputs['out'], masks)
        if 'aux' in outputs:
            loss = loss + 0.4 * criterion(outputs['aux'], masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    return {'total': total_loss / len(data_loader)}


@torch.no_grad()
def _evaluate(model, data_loader, device):
    model.eval()
    criterion  = nn.CrossEntropyLoss(ignore_index=255)
    total_loss = 0
    for images, masks in data_loader:
        images, masks = images.to(device), masks.to(device)
        outputs       = model(images)
        total_loss   += criterion(outputs['out'], masks).item()
    return total_loss / len(data_loader)


def save_checkpoint(model, optimizer, epoch, loss, path, time_stats=None, model_config=None):
    ckpt = {'epoch': epoch, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), 'loss': loss}
    if time_stats:   ckpt['time_stats']   = time_stats
    if model_config: ckpt['model_config'] = model_config
    torch.save(ckpt, path)


def run_training(model, device, train_loader, val_loader, model_config):
    params       = [p for p in model.parameters() if p.requires_grad]
    optimizer    = torch.optim.SGD(params, lr=CONFIG["learning_rate"],
                                   momentum=CONFIG["momentum"],
                                   weight_decay=CONFIG["weight_decay"])
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=CONFIG["lr_step_size"], gamma=CONFIG["lr_gamma"]
    )

    history = {'train_loss': [], 'val_loss': [], 'lr': [],
               'epoch_times': [], 'cumulative_times': []}
    best_val_loss = float('inf')
    timer = TrainingTimer(CONFIG["num_epochs"])

    print("\n" + "=" * 70)
    print("   DEBUT DE L'ENTRAINEMENT")
    print(f"   Demarre le: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Epochs: {CONFIG['num_epochs']} | Batch: {CONFIG['batch_size']}")
    print("=" * 70)

    timer.start_training()

    for epoch in range(CONFIG["num_epochs"]):
        timer.start_epoch()
        train_losses = train_one_epoch(model, optimizer, train_loader, device, epoch)
        val_loss     = _evaluate(model, val_loader, device)
        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        time_stats = timer.end_epoch(epoch)

        history['train_loss'].append(train_losses['total'])
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)
        history['epoch_times'].append(time_stats['epoch_time'])
        history['cumulative_times'].append(time_stats['total_elapsed'])

        print(f"\n{'─' * 70}")
        print(f"Epoch {epoch+1}/{CONFIG['num_epochs']} | {time_stats['progress_percent']:.1f}%")
        print(f"   Train Loss: {train_losses['total']:.4f}")
        print(f"   Val Loss:   {val_loss:.4f}")
        print(f"   LR:         {current_lr:.6f}")
        print(f"   Epoch: {format_time(time_stats['epoch_time'])}  |  "
              f"Total: {format_time(time_stats['total_elapsed'])}  |  "
              f"Restant: {format_time(time_stats['estimated_remaining'])}  |  "
              f"ETA: {time_stats['eta'].strftime('%H:%M:%S')}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss,
                            os.path.join(CONFIG["output_dir"], "best_model.pth"),
                            time_stats={'epoch_time': time_stats['epoch_time'],
                                        'total_elapsed': time_stats['total_elapsed']},
                            model_config=model_config)
            print(f"   Meilleur modele sauvegarde!")

        if (epoch + 1) % CONFIG["save_every"] == 0:
            save_checkpoint(model, optimizer, epoch, val_loss,
                            os.path.join(CONFIG["output_dir"], f"checkpoint_epoch_{epoch+1}.pth"),
                            time_stats={'epoch_time': time_stats['epoch_time'],
                                        'total_elapsed': time_stats['total_elapsed']},
                            model_config=model_config)
            print(f"   Checkpoint epoch {epoch+1} sauvegarde")

    fts = timer.get_final_stats()
    history['time_stats'] = fts

    save_checkpoint(model, optimizer, CONFIG["num_epochs"] - 1, val_loss,
                    os.path.join(CONFIG["output_dir"], "final_model.pth"),
                    time_stats=fts, model_config=model_config)

    with open(os.path.join(CONFIG["output_dir"], "history.json"), 'w') as f:
        json.dump(history, f, indent=2)

    # Courbes
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history['train_loss'], label='Train', linewidth=2)
    axes[0].plot(history['val_loss'],   label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title('Courbes de perte - DeepLabV3+ Cadastral')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].bar(range(1, len(history['epoch_times']) + 1), history['epoch_times'],
                color='steelblue', alpha=0.7)
    axes[1].axhline(y=fts['avg_epoch_time'], color='red', linestyle='--', linewidth=2,
                    label=f"Moyenne: {fts['avg_epoch_time_formatted']}")
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Temps (s)')
    axes[1].set_title('Temps par epoch'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["output_dir"], "training_curves.png"), dpi=150)
    plt.close()

    print("\n" + "=" * 70)
    print("   ENTRAINEMENT TERMINE")
    print("=" * 70)
    print(f"   Meilleure Val Loss: {best_val_loss:.4f}")
    print(f"   Train Loss finale:  {history['train_loss'][-1]:.4f}")
    print(f"   Val Loss finale:    {history['val_loss'][-1]:.4f}")
    print(f"   Temps total:        {fts['total_time_formatted']}")
    print(f"   Fichiers:           {CONFIG['output_dir']}/")
    print("=" * 70)

    report_path = os.path.join(CONFIG["output_dir"], "training_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("RAPPORT D'ENTRAINEMENT - DeepLabV3+ CADASTRAL\n" + "=" * 70 + "\n\n")
        f.write("CONFIGURATION\n" + "-" * 50 + "\n")
        for k, v in CONFIG.items():
            f.write(f"   {k}: {v}\n")
        f.write(f"\nMODE: {model_config.get('mode','?')}\n")
        f.write(f"\nPERFORMANCES\n   Meilleure Val Loss: {best_val_loss:.4f}\n")
        f.write(f"   Train Loss finale:  {history['train_loss'][-1]:.4f}\n")
        f.write(f"   Val Loss finale:    {history['val_loss'][-1]:.4f}\n")
        f.write(f"\nTEMPS\n   Debut:  {fts['start_datetime']}\n   Fin:    {fts['end_datetime']}\n")
        f.write(f"   Total:  {fts['total_time_formatted']}\n")
        f.write(f"   Moy/ep: {fts['avg_epoch_time_formatted']}\n")
        f.write("\nTEMPS PAR EPOCH\n" + "-" * 50 + "\n")
        for i, t in enumerate(fts['epoch_times']):
            f.write(f"   Epoch {i+1:3d}: {format_time(t)}\n")
    print(f"   Rapport sauvegarde: {report_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="DeepLabV3+ - Toitures Cadastrales",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes disponibles:
  simple    DeepLabV3+ standard, hyperparamètres fixes
  attention DeepLabV3+ + CBAM(256) dans la tête de classification (après ASPP)
  optimize  DeepLabV3+ standard + recherche bayésienne des hyperparamètres (Optuna)
        """
    )
    parser.add_argument("--mode", choices=["simple", "attention", "optimize"],
                        default="simple", help="Mode d'entraînement (défaut: simple)")
    parser.add_argument("--n-trials",       type=int, default=OPTUNA_CONFIG["n_trials"])
    parser.add_argument("--n-epochs-trial", type=int, default=OPTUNA_CONFIG["n_epochs_per_trial"])
    parser.add_argument("--cbam-reduction", type=int, default=CONFIG["cbam_reduction"])
    parser.add_argument("--cbam-kernel-size", type=int, default=CONFIG["cbam_kernel_size"],
                        choices=[3, 5, 7])
    parser.add_argument("--backbone", default=CONFIG["backbone"],
                        choices=["resnet50", "resnet101"])
    args = parser.parse_args()

    OPTUNA_CONFIG["n_trials"]           = args.n_trials
    OPTUNA_CONFIG["n_epochs_per_trial"] = args.n_epochs_trial
    CONFIG["backbone"]                  = args.backbone

    print("=" * 70)
    print("   DeepLabV3+ - Segmentation des Toitures Cadastrales")
    print(f"   Mode: {args.mode.upper()}")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"   GPU:  {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    # ── Coefficients d'augmentation (depuis ENV vars) ────────────────────────
    aug_coeffs = load_aug_coefficients(CONFIG["classes"])

    print("\nChargement du dataset...")
    _coco_ref = COCO(CONFIG["annotations_file"])
    all_ids   = list(_coco_ref.imgs.keys())
    np.random.seed(42)
    np.random.shuffle(all_ids)
    split_idx     = int(CONFIG["train_split"] * len(all_ids))
    train_raw_ids = all_ids[:split_idx]
    val_ids       = all_ids[split_idx:]

    # ── Augmentation par classe ──────────────────────────────────────────────
    cat_id_to_name  = {cid: _coco_ref.cats[cid]['name'] for cid in _coco_ref.getCatIds()}
    train_aug_ids, _ = oversample_by_class(
        train_raw_ids, _coco_ref, cat_id_to_name, aug_coeffs
    )
    print_augmentation_report(
        _coco_ref, train_raw_ids, train_aug_ids, cat_id_to_name, aug_coeffs
    )

    train_dataset = DeepLabDataset(
        CONFIG["images_dir"], CONFIG["annotations_file"],
        image_size=CONFIG["image_size"],
        transforms=get_transforms(train=True),
        image_ids=train_aug_ids,
    )
    val_dataset = DeepLabDataset(
        CONFIG["images_dir"], CONFIG["annotations_file"],
        image_size=CONFIG["image_size"],
        transforms=None,
        image_ids=val_ids,
    )
    print(f"   Train: {len(train_dataset)} samples  ({len(set(train_aug_ids))} images uniques)")
    print(f"   Val:   {len(val_dataset)} images")

    pin = device.type == 'cuda'

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True,
                              num_workers=CONFIG["num_workers"], pin_memory=pin)
    val_loader   = DataLoader(val_dataset,   batch_size=CONFIG["batch_size"], shuffle=False,
                              num_workers=CONFIG["num_workers"], pin_memory=pin)

    num_classes = len(CONFIG["classes"])

    if args.mode == "simple":
        print(f"\nArchitecture: DeepLabV3+ ({args.backbone}, standard)")
        model        = get_model_simple(num_classes, args.backbone, CONFIG["pretrained"])
        model_config = {"mode": "simple", "backbone": args.backbone}

    elif args.mode == "attention":
        cbam_r = args.cbam_reduction
        cbam_k = args.cbam_kernel_size
        print(f"\nArchitecture: DeepLabV3+ ({args.backbone}) + CBAM(256)")
        print(f"   cbam_reduction={cbam_r}, cbam_kernel_size={cbam_k}")
        model        = get_model_attention(num_classes, args.backbone, CONFIG["pretrained"],
                                           cbam_r, cbam_k)
        model_config = {"mode": "attention", "backbone": args.backbone,
                        "cbam_reduction": cbam_r, "cbam_kernel_size": cbam_k}

    else:  # optimize
        print(f"\nArchitecture: DeepLabV3+ ({args.backbone}, standard)")
        print("Lancement de l'optimisation bayesienne des hyperparametres...")
        best_params = _run_optimization(device, train_loader, val_loader, num_classes)
        for key in ("learning_rate", "weight_decay", "momentum", "lr_step_size"):
            if key in best_params:
                CONFIG[key] = best_params[key]
        print(f"\nHyperparametres optimises appliques a l'entrainement.")
        model        = get_model_simple(num_classes, args.backbone, CONFIG["pretrained"])
        model_config = {"mode": "optimize", "backbone": args.backbone, "best_params": best_params}

    model.to(device)
    print(f"   Classes: {CONFIG['classes']}")

    run_training(model, device, train_loader, val_loader, model_config)


if __name__ == "__main__":
    main()
