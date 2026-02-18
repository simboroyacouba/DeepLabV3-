"""
Entraînement DeepLabV3+ pour segmentation des toitures cadastrales
Dataset: Images aériennes annotées avec CVAT (format COCO)
Classes: toiture_tole_ondulee, toiture_tole_bac, toiture_tuile, toiture_dalle

Structure identique à Mask R-CNN pour comparaison équitable
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
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
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION (identique à Mask R-CNN)
# =============================================================================

CONFIG = {
    # Chemins (à adapter)
    "images_dir": os.getenv("SEGMENTATION_DATASET_IMAGES_DIR"),
    "annotations_file": os.getenv("SEGMENTATION_DATASET_ANNOTATIONS_FILE"),
    "output_dir": "./output",
    
    # Classes (dans l'ordre de CVAT) - IDENTIQUE à Mask R-CNN
    "classes": [
        "__background__",        # 0 - toujours en premier
        "toiture_tole_ondulee",  # 1
        "toiture_tole_bac",      # 2
        "toiture_tuile",         # 3
        "toiture_dalle"          # 4
    ],
    
    # Modèle
    "backbone": "resnet50",  # resnet50 ou resnet101
    "pretrained": True,
    
    # Hyperparamètres - IDENTIQUES à Mask R-CNN
    "num_epochs": 25,
    "batch_size": 2,
    "learning_rate": 0.005,
    "momentum": 0.9,
    "weight_decay": 0.0005,
    "lr_step_size": 8,
    "lr_gamma": 0.1,
    
    # Images
    "image_size": 512,
    
    # Dataset - IDENTIQUE à Mask R-CNN
    "train_split": 0.85,
    "num_workers": 0,
    
    # Sauvegarde
    "save_every": 5,
}


# =============================================================================
# UTILITAIRES TEMPS (identique à Mask R-CNN)
# =============================================================================

def format_time(seconds):
    """Formater les secondes en format lisible HH:MM:SS"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h {minutes}m {secs}s"


class TrainingTimer:
    """Classe pour gérer le suivi du temps d'entraînement"""
    
    def __init__(self, num_epochs):
        self.num_epochs = num_epochs
        self.start_time = None
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
        
        total_elapsed = time.time() - self.start_time
        avg_epoch_time = np.mean(self.epoch_times)
        remaining_epochs = self.num_epochs - (epoch + 1)
        estimated_remaining = avg_epoch_time * remaining_epochs
        eta = datetime.now() + timedelta(seconds=estimated_remaining)
        
        return {
            'epoch_time': epoch_time,
            'total_elapsed': total_elapsed,
            'avg_epoch_time': avg_epoch_time,
            'estimated_remaining': estimated_remaining,
            'eta': eta,
            'progress_percent': ((epoch + 1) / self.num_epochs) * 100
        }
    
    def get_final_stats(self):
        total_time = time.time() - self.start_time
        return {
            'total_time': total_time,
            'total_time_formatted': format_time(total_time),
            'avg_epoch_time': np.mean(self.epoch_times),
            'avg_epoch_time_formatted': format_time(np.mean(self.epoch_times)),
            'min_epoch_time': np.min(self.epoch_times),
            'min_epoch_time_formatted': format_time(np.min(self.epoch_times)),
            'max_epoch_time': np.max(self.epoch_times),
            'max_epoch_time_formatted': format_time(np.max(self.epoch_times)),
            'std_epoch_time': np.std(self.epoch_times),
            'epoch_times': self.epoch_times,
            'start_datetime': self.training_start_datetime.strftime("%Y-%m-%d %H:%M:%S"),
            'end_datetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }


# =============================================================================
# DATASET (adapté pour segmentation sémantique)
# =============================================================================

class DeepLabDataset(torch.utils.data.Dataset):
    """
    Dataset pour segmentation sémantique avec DeepLab
    Convertit les annotations COCO (instances) en masques sémantiques
    """
    
    def __init__(self, images_dir, annotations_file, image_size=512, transforms=None):
        self.images_dir = images_dir
        self.image_size = image_size
        self.transforms = transforms
        
        # Charger annotations COCO
        self.coco = COCO(annotations_file)
        self.image_ids = list(self.coco.imgs.keys())
        
        # Mapping catégories COCO -> indices locaux (identique à Mask R-CNN)
        self.cat_ids = self.coco.getCatIds()
        self.cat_mapping = {cat_id: idx + 1 for idx, cat_id in enumerate(self.cat_ids)}
        
        print(f"Dataset chargé: {len(self.image_ids)} images")
        print(f"Catégories: {[self.coco.cats[c]['name'] for c in self.cat_ids]}")
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.imgs[img_id]
        
        # Charger l'image
        img_path = os.path.join(self.images_dir, img_info['file_name'])
        image = Image.open(img_path).convert("RGB")
        
        # Créer le masque sémantique (tous les pixels = 0 par défaut = background)
        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
        
        # Récupérer les annotations et remplir le masque
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        for ann in anns:
            if ann.get('iscrowd', 0):
                continue
            
            class_id = self.cat_mapping[ann['category_id']]
            
            if 'segmentation' in ann:
                if isinstance(ann['segmentation'], list):
                    rles = coco_mask_utils.frPyObjects(
                        ann['segmentation'],
                        img_info['height'],
                        img_info['width']
                    )
                    rle = coco_mask_utils.merge(rles)
                    instance_mask = coco_mask_utils.decode(rle)
                else:
                    instance_mask = coco_mask_utils.decode(ann['segmentation'])
                
                mask[instance_mask > 0] = class_id
        
        # Convertir en PIL pour redimensionnement
        mask_pil = Image.fromarray(mask)
        
        # Redimensionner
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask_pil = mask_pil.resize((self.image_size, self.image_size), Image.NEAREST)
        
        # Appliquer les transformations (augmentation)
        if self.transforms is not None:
            image, mask_pil = self.transforms(image, mask_pil)
        
        # Convertir en tenseurs
        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        mask_tensor = torch.as_tensor(np.array(mask_pil), dtype=torch.long)
        
        return image_tensor, mask_tensor


# =============================================================================
# TRANSFORMATIONS (identique à Mask R-CNN)
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
            mask = TF.hflip(mask)
        return image, mask


class RandomVerticalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob
    
    def __call__(self, image, mask):
        if random.random() < self.prob:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        return image, mask


def get_transforms(train=True):
    transforms = []
    if train:
        transforms.append(RandomHorizontalFlip(0.5))
        transforms.append(RandomVerticalFlip(0.5))
    return Compose(transforms) if transforms else None


# =============================================================================
# MODÈLE
# =============================================================================

def get_model(num_classes, backbone="resnet50", pretrained=True):
    """Créer un modèle DeepLabV3+ fine-tuné pour N classes"""
    
    if backbone == "resnet50":
        if pretrained:
            model = deeplabv3_resnet50(weights="DEFAULT")
        else:
            model = deeplabv3_resnet50(weights=None)
        in_channels = 2048
    elif backbone == "resnet101":
        if pretrained:
            model = deeplabv3_resnet101(weights="DEFAULT")
        else:
            model = deeplabv3_resnet101(weights=None)
        in_channels = 2048
    else:
        raise ValueError(f"Backbone inconnu: {backbone}")
    
    # Remplacer la tête de classification
    model.classifier = DeepLabHead(in_channels, num_classes)
    
    # Remplacer l'auxiliary classifier
    if model.aux_classifier is not None:
        model.aux_classifier = nn.Sequential(
            nn.Conv2d(1024, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1)
        )
    
    return model


# =============================================================================
# ENTRAÎNEMENT
# =============================================================================

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """Entraîner une epoch"""
    model.train()
    
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    
    total_loss = 0
    
    pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}")
    
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward
        outputs = model(images)
        pred = outputs['out']
        
        # Loss principale
        loss = criterion(pred, masks)
        
        # Loss auxiliaire
        if 'aux' in outputs:
            aux_loss = criterion(outputs['aux'], masks)
            loss = loss + 0.4 * aux_loss
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    return {'total': total_loss / len(data_loader)}


@torch.no_grad()
def evaluate(model, data_loader, device):
    """Évaluer sur le set de validation"""
    model.eval()
    
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    total_loss = 0
    
    for images, masks in data_loader:
        images = images.to(device)
        masks = masks.to(device)
        
        outputs = model(images)
        pred = outputs['out']
        
        loss = criterion(pred, masks)
        total_loss += loss.item()
    
    return total_loss / len(data_loader)


def save_checkpoint(model, optimizer, epoch, loss, path, time_stats=None):
    """Sauvegarder un checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    if time_stats:
        checkpoint['time_stats'] = time_stats
    torch.save(checkpoint, path)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("   DeepLabV3+ - Segmentation des Toitures Cadastrales")
    print("=" * 70)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📱 Device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Créer le dossier de sortie
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    
    # Dataset
    print("\n📂 Chargement du dataset...")
    full_dataset = DeepLabDataset(
        CONFIG["images_dir"],
        CONFIG["annotations_file"],
        image_size=CONFIG["image_size"],
        transforms=None
    )
    
    # Split train/val (identique à Mask R-CNN)
    train_size = int(CONFIG["train_split"] * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"   Train: {len(train_dataset)} images")
    print(f"   Val: {len(val_dataset)} images")
    
    # Appliquer les transformations au dataset d'entraînement
    full_dataset.transforms = get_transforms(train=True)
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Modèle
    print(f"\n🧠 Création du modèle DeepLabV3+ ({CONFIG['backbone']})...")
    num_classes = len(CONFIG["classes"])
    model = get_model(num_classes, CONFIG["backbone"], CONFIG["pretrained"])
    model.to(device)
    print(f"   Classes: {CONFIG['classes']}")
    
    # Optimiseur (identique à Mask R-CNN)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=CONFIG["learning_rate"],
        momentum=CONFIG["momentum"],
        weight_decay=CONFIG["weight_decay"]
    )
    
    # Scheduler (identique à Mask R-CNN)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=CONFIG["lr_step_size"],
        gamma=CONFIG["lr_gamma"]
    )
    
    # Historique des pertes et temps
    history = {
        'train_loss': [],
        'val_loss': [],
        'lr': [],
        'epoch_times': [],
        'cumulative_times': []
    }
    
    best_val_loss = float('inf')
    
    # Timer
    timer = TrainingTimer(CONFIG["num_epochs"])
    
    # Entraînement
    print("\n" + "=" * 70)
    print("   🚀 DÉBUT DE L'ENTRAÎNEMENT")
    print(f"   📅 Démarré le: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   📊 Epochs: {CONFIG['num_epochs']} | Batch size: {CONFIG['batch_size']}")
    print("=" * 70)
    
    timer.start_training()
    
    for epoch in range(CONFIG["num_epochs"]):
        timer.start_epoch()
        
        # Train
        train_losses = train_one_epoch(model, optimizer, train_loader, device, epoch)
        
        # Validation
        val_loss = evaluate(model, val_loader, device)
        
        # Scheduler
        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Stats de temps
        time_stats = timer.end_epoch(epoch)
        
        # Historique
        history['train_loss'].append(train_losses['total'])
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)
        history['epoch_times'].append(time_stats['epoch_time'])
        history['cumulative_times'].append(time_stats['total_elapsed'])
        
        # Affichage détaillé (identique à Mask R-CNN)
        print(f"\n{'─' * 70}")
        print(f"📈 Epoch {epoch+1}/{CONFIG['num_epochs']} | Progression: {time_stats['progress_percent']:.1f}%")
        print(f"{'─' * 70}")
        print(f"   📉 Train Loss: {train_losses['total']:.4f}")
        print(f"   📊 Val Loss:   {val_loss:.4f}")
        print(f"   📐 LR:         {current_lr:.6f}")
        print(f"{'─' * 70}")
        print(f"   ⏱️  Temps epoch:       {format_time(time_stats['epoch_time'])}")
        print(f"   ⏱️  Temps total:       {format_time(time_stats['total_elapsed'])}")
        print(f"   ⏱️  Temps moyen/epoch: {format_time(time_stats['avg_epoch_time'])}")
        print(f"   ⏳ Temps restant:      {format_time(time_stats['estimated_remaining'])}")
        print(f"   🏁 ETA:                {time_stats['eta'].strftime('%H:%M:%S')}")
        
        # Sauvegarder le meilleur modèle
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                os.path.join(CONFIG["output_dir"], "best_model.pth"),
                time_stats={'epoch_time': time_stats['epoch_time'], 'total_elapsed': time_stats['total_elapsed']}
            )
            print(f"   ✅ Meilleur modèle sauvegardé!")
        
        # Sauvegardes périodiques
        if (epoch + 1) % CONFIG["save_every"] == 0:
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                os.path.join(CONFIG["output_dir"], f"checkpoint_epoch_{epoch+1}.pth"),
                time_stats={'epoch_time': time_stats['epoch_time'], 'total_elapsed': time_stats['total_elapsed']}
            )
            print(f"   💾 Checkpoint epoch {epoch+1} sauvegardé")
    
    # Stats finales
    final_time_stats = timer.get_final_stats()
    history['time_stats'] = final_time_stats
    
    # Sauvegarder le modèle final
    save_checkpoint(
        model, optimizer, CONFIG["num_epochs"]-1, val_loss,
        os.path.join(CONFIG["output_dir"], "final_model.pth"),
        time_stats=final_time_stats
    )
    
    # Sauvegarder l'historique
    with open(os.path.join(CONFIG["output_dir"], "history.json"), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot des courbes (identique à Mask R-CNN)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(history['train_loss'], label='Train', linewidth=2)
    axes[0].plot(history['val_loss'], label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Courbes de perte - DeepLabV3+ Cadastral')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].bar(range(1, len(history['epoch_times']) + 1), history['epoch_times'],
                color='steelblue', alpha=0.7, label='Temps par epoch')
    axes[1].axhline(y=final_time_stats['avg_epoch_time'], color='red',
                    linestyle='--', linewidth=2, label=f"Moyenne: {final_time_stats['avg_epoch_time_formatted']}")
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Temps (secondes)')
    axes[1].set_title('Temps par epoch')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["output_dir"], "training_curves.png"), dpi=150)
    plt.close()
    
    # Rapport final (identique à Mask R-CNN)
    print("\n" + "=" * 70)
    print("   🎉 ENTRAÎNEMENT TERMINÉ")
    print("=" * 70)
    print(f"\n📊 RÉSUMÉ DES PERFORMANCES")
    print(f"   {'─' * 50}")
    print(f"   Meilleure Val Loss: {best_val_loss:.4f}")
    print(f"   Train Loss finale:  {history['train_loss'][-1]:.4f}")
    print(f"   Val Loss finale:    {history['val_loss'][-1]:.4f}")
    
    print(f"\n⏱️  RAPPORT DE TEMPS")
    print(f"   {'─' * 50}")
    print(f"   Début:              {final_time_stats['start_datetime']}")
    print(f"   Fin:                {final_time_stats['end_datetime']}")
    print(f"   {'─' * 50}")
    print(f"   ⏱️  Temps total:       {final_time_stats['total_time_formatted']}")
    print(f"   ⏱️  Temps moyen/epoch: {final_time_stats['avg_epoch_time_formatted']}")
    print(f"   ⏱️  Epoch la + rapide: {final_time_stats['min_epoch_time_formatted']}")
    print(f"   ⏱️  Epoch la + lente:  {final_time_stats['max_epoch_time_formatted']}")
    print(f"   📈 Écart-type:         {final_time_stats['std_epoch_time']:.2f}s")
    
    print(f"\n💾 FICHIERS SAUVEGARDÉS")
    print(f"   {'─' * 50}")
    print(f"   📁 Dossier: {CONFIG['output_dir']}")
    print(f"   ├── best_model.pth")
    print(f"   ├── final_model.pth")
    print(f"   ├── checkpoint_epoch_*.pth")
    print(f"   ├── history.json")
    print(f"   └── training_curves.png")
    print("=" * 70)
    
    # Sauvegarder le rapport
    report_path = os.path.join(CONFIG["output_dir"], "training_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("   RAPPORT D'ENTRAÎNEMENT - DeepLabV3+ CADASTRAL\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("CONFIGURATION\n")
        f.write("-" * 50 + "\n")
        for key, value in CONFIG.items():
            f.write(f"   {key}: {value}\n")
        
        f.write("\nPERFORMANCES\n")
        f.write("-" * 50 + "\n")
        f.write(f"   Meilleure Val Loss: {best_val_loss:.4f}\n")
        f.write(f"   Train Loss finale:  {history['train_loss'][-1]:.4f}\n")
        f.write(f"   Val Loss finale:    {history['val_loss'][-1]:.4f}\n")
        
        f.write("\nTEMPS D'ENTRAÎNEMENT\n")
        f.write("-" * 50 + "\n")
        f.write(f"   Début:               {final_time_stats['start_datetime']}\n")
        f.write(f"   Fin:                 {final_time_stats['end_datetime']}\n")
        f.write(f"   Temps total:         {final_time_stats['total_time_formatted']}\n")
        f.write(f"   Temps moyen/epoch:   {final_time_stats['avg_epoch_time_formatted']}\n")
        f.write(f"   Epoch la + rapide:   {final_time_stats['min_epoch_time_formatted']}\n")
        f.write(f"   Epoch la + lente:    {final_time_stats['max_epoch_time_formatted']}\n")
        f.write(f"   Écart-type:          {final_time_stats['std_epoch_time']:.2f}s\n")
        
        f.write("\nTEMPS PAR EPOCH\n")
        f.write("-" * 50 + "\n")
        for i, t in enumerate(final_time_stats['epoch_times']):
            f.write(f"   Epoch {i+1:3d}: {format_time(t)}\n")
    
    print(f"\n📄 Rapport sauvegardé: {report_path}")


if __name__ == "__main__":
    main()
