"""
Évaluation complète du modèle DeepLabV3+
Métriques IDENTIQUES à Mask R-CNN pour comparaison équitable:
- mAP (adapté pour segmentation sémantique)
- mAP@50 (IoU threshold = 0.5)
- mAP@50:95 (IoU thresholds de 0.5 à 0.95)
- Precision, Recall, F1-Score
- IoU moyen
- Matrice de confusion
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import torchvision.transforms.functional as TF
from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask_utils
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime
import warnings
import yaml
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION (identique à Mask R-CNN)
# =============================================================================


def load_classes(yaml_path=None):
    path = yaml_path or os.getenv("CLASSES_FILE", "classes.yaml")
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data['classes'] 


CONFIG = {
    # Chemins
    "images_dir":  os.getenv("SEGMENTATION_DATASET_IMAGES_DIR"),
    "annotations_file": os.getenv("SEGMENTATION_DATASET_ANNOTATIONS_FILE"),
    "classes_file": os.getenv("CLASSES_FILE", "classes.yaml"),
    "model_path": "./output/best_model.pth",
    "output_dir": "./evaluation",
    
    # Classes (identique à Mask R-CNN)
    
    "classes": load_classes(), 
    
    # Paramètres d'évaluation (identique à Mask R-CNN)
    "score_threshold": 0.5,
    "iou_thresholds": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
    "batch_size": 1,
    "num_workers": 0,
    "image_size": 512,
    "backbone": "resnet50",
}


# =============================================================================
# DATASET
# =============================================================================

class EvalDataset(torch.utils.data.Dataset):
    """Dataset pour l'évaluation"""
    
    def __init__(self, images_dir, annotations_file, image_size=512):
        self.images_dir = images_dir
        self.image_size = image_size
        
        self.coco = COCO(annotations_file)
        self.image_ids = list(self.coco.imgs.keys())
        
        self.cat_ids = self.coco.getCatIds()
        self.cat_mapping = {cat_id: idx + 1 for idx, cat_id in enumerate(self.cat_ids)}
        self.reverse_cat_mapping = {v: k for k, v in self.cat_mapping.items()}
        
        print(f"Dataset d'évaluation: {len(self.image_ids)} images")
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.imgs[img_id]
        
        # Charger l'image
        img_path = os.path.join(self.images_dir, img_info['file_name'])
        image = Image.open(img_path).convert("RGB")
        original_size = (img_info['height'], img_info['width'])
        
        # Créer le masque sémantique ground truth
        mask_gt = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
        
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # Stocker aussi les masques individuels pour calcul IoU par instance
        instance_masks = []
        instance_labels = []
        instance_boxes = []
        
        for ann in anns:
            if ann.get('iscrowd', 0):
                continue
            
            class_id = self.cat_mapping[ann['category_id']]
            
            x, y, w, h = ann['bbox']
            if w <= 0 or h <= 0:
                continue
            
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
                
                mask_gt[instance_mask > 0] = class_id
                instance_masks.append(instance_mask)
                instance_labels.append(class_id)
                instance_boxes.append([x, y, x + w, y + h])
        
        # Redimensionner
        image_resized = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask_gt_resized = Image.fromarray(mask_gt).resize((self.image_size, self.image_size), Image.NEAREST)
        
        # Convertir en tenseurs
        image_tensor = TF.to_tensor(image_resized)
        image_tensor = TF.normalize(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        mask_tensor = torch.as_tensor(np.array(mask_gt_resized), dtype=torch.long)
        
        # Target avec les instances (pour calcul identique à Mask R-CNN)
        target = {
            'masks': instance_masks,
            'labels': instance_labels,
            'boxes': instance_boxes,
            'semantic_mask': mask_gt,
            'image_id': img_id,
            'original_size': original_size
        }
        
        return image_tensor, mask_tensor, target


# =============================================================================
# MÉCANISME D'ATTENTION (CBAM) — doit correspondre à train.py
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

def _build_base_model(num_classes, backbone="resnet50"):
    if backbone == "resnet50":
        model = deeplabv3_resnet50(weights=None)
        in_channels = 2048
    elif backbone == "resnet101":
        model = deeplabv3_resnet101(weights=None)
        in_channels = 2048
    else:
        raise ValueError(f"Backbone inconnu: {backbone}")
    model.classifier = DeepLabHead(in_channels, num_classes)
    if model.aux_classifier is not None:
        model.aux_classifier = nn.Sequential(
            nn.Conv2d(1024, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1)
        )
    return model


def _build_attention_model(num_classes, backbone="resnet50",
                           cbam_reduction=16, cbam_kernel_size=7):
    model = _build_base_model(num_classes, backbone)
    layers = list(model.classifier.children())
    model.classifier = nn.Sequential(
        *layers[:4],
        CBAM(256, cbam_reduction, cbam_kernel_size),
        layers[4],
    )
    return model


def load_model(model_path, num_classes, backbone, device):
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model_state_dict']

    # Détecter l'architecture (simple vs attention) selon les clés présentes
    has_cbam = 'classifier.5.weight' in state_dict
    final_key = 'classifier.5.weight' if has_cbam else 'classifier.4.weight'

    if final_key not in state_dict:
        raise KeyError(f"Clé finale de classification introuvable dans le checkpoint. "
                       f"Clés classifier.*: {[k for k in state_dict if k.startswith('classifier')]}")

    detected_classes = state_dict[final_key].shape[0]
    if detected_classes != num_classes:
        print(f"⚠️ num_classes corrigé: {num_classes} → {detected_classes}")
        num_classes = detected_classes

    has_aux = any(k.startswith('aux_classifier') for k in state_dict.keys())

    if has_cbam:
        # Récupérer les hyperparamètres CBAM depuis le checkpoint si disponibles
        model_config     = checkpoint.get('model_config', {})
        cbam_reduction   = model_config.get('cbam_reduction',   16)
        cbam_kernel_size = model_config.get('cbam_kernel_size',  7)
        # Fallback: déduire depuis le state_dict
        if 'classifier.4.channel_att.fc.0.weight' in state_dict:
            mid = state_dict['classifier.4.channel_att.fc.0.weight'].shape[0]
            cbam_reduction = max(1, 256 // mid)
        if 'classifier.4.spatial_att.conv.weight' in state_dict:
            cbam_kernel_size = state_dict['classifier.4.spatial_att.conv.weight'].shape[2]
        model = _build_attention_model(num_classes, backbone, cbam_reduction, cbam_kernel_size)
        print(f"   Architecture: DeepLabV3+ + CBAM(reduction={cbam_reduction}, kernel={cbam_kernel_size})")
    else:
        model = _build_base_model(num_classes, backbone)
        print(f"   Architecture: DeepLabV3+ standard")

    if has_aux and model.aux_classifier is None:
        model.aux_classifier = nn.Sequential(
            nn.Conv2d(1024, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1)
        )

    model.load_state_dict(state_dict)
    model.aux_classifier = None  # inutile à l'évaluation
    model.to(device)
    model.eval()

    print(f"✅ Modèle chargé: {model_path} (epoch={checkpoint.get('epoch', 'N/A')})")
    return model


# =============================================================================
# CALCUL DES MÉTRIQUES (identique à Mask R-CNN)
# =============================================================================

def calculate_iou_masks(mask1, mask2):
    """Calculer IoU entre deux masques binaires"""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0


def extract_connected_components(semantic_mask, class_id):
    """
    Extraire les composantes connexes d'une classe dans un masque sémantique
    Pour simuler la détection d'instances à partir de la segmentation sémantique
    """
    from scipy import ndimage
    
    binary_mask = (semantic_mask == class_id).astype(np.uint8)
    labeled_array, num_features = ndimage.label(binary_mask)
    
    instances = []
    for i in range(1, num_features + 1):
        instance_mask = (labeled_array == i).astype(np.uint8)
        if instance_mask.sum() > 100:  # Ignorer les très petites régions (bruit)
            # Calculer la bounding box
            rows = np.any(instance_mask, axis=1)
            cols = np.any(instance_mask, axis=0)
            if rows.any() and cols.any():
                y1, y2 = np.where(rows)[0][[0, -1]]
                x1, x2 = np.where(cols)[0][[0, -1]]
                instances.append({
                    'mask': instance_mask,
                    'box': [x1, y1, x2, y2],
                    'area': instance_mask.sum()
                })
    
    return instances


def calculate_ap(recalls, precisions):
    """Calculer AP = aire sous la courbe Precision-Recall (interpolation continue)."""
    recalls = np.concatenate([[0.0], recalls, [1.0]])
    precisions = np.concatenate([[0.0], precisions, [0.0]])
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    indices = np.where(recalls[1:] != recalls[:-1])[0] + 1
    ap = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])
    return float(ap)


class MetricsCalculator:
    """
    Calcule le vrai mAP (mean Average Precision) = moyenne de l'AP
    (aire sous la courbe Precision-Recall) par classe et par seuil IoU.
    Pour DeepLab, l'aire de la composante connexe sert de proxy de score de confiance.
    """

    def __init__(self, num_classes, class_names, iou_thresholds):
        self.num_classes = num_classes
        self.class_names = class_names
        self.iou_thresholds = iou_thresholds
        self.reset()

    def reset(self):
        # class_id -> liste de {'score', 'ious': np.array(n_gt,), 'img_idx'}
        self.detections = defaultdict(list)
        # class_id -> nombre total d'instances GT
        self.n_gts = defaultdict(int)
        self._img_idx = 0
        self.mask_ious = []
        # Matrice de confusion pour les métriques sémantiques
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update_semantic(self, pred_mask, gt_mask):
        """Mettre à jour la matrice de confusion pour segmentation sémantique."""
        pred_flat = pred_mask.flatten()
        gt_flat = gt_mask.flatten()
        for p, g in zip(pred_flat, gt_flat):
            if g < self.num_classes and p < self.num_classes:
                self.confusion_matrix[g, p] += 1

    def add_image(self, pred_semantic, gt_semantic, gt_instances, gt_labels):
        """Ajouter une image pour évaluation (segmentation sémantique → instances)."""
        self.update_semantic(pred_semantic, gt_semantic)

        for class_id in range(1, self.num_classes):
            pred_instances = extract_connected_components(pred_semantic, class_id)

            gt_class_masks = []
            for i, label in enumerate(gt_labels):
                if label == class_id:
                    gt_mask = gt_instances[i]
                    if gt_mask.shape != pred_semantic.shape:
                        gt_mask_pil = Image.fromarray(gt_mask.astype(np.uint8))
                        gt_mask_pil = gt_mask_pil.resize(
                            (pred_semantic.shape[1], pred_semantic.shape[0]),
                            Image.NEAREST
                        )
                        gt_mask = np.array(gt_mask_pil)
                    gt_class_masks.append(gt_mask)

            n_pred = len(pred_instances)
            n_gt = len(gt_class_masks)
            self.n_gts[class_id] += n_gt

            if n_pred == 0:
                continue

            # Matrice IoU (n_pred x n_gt) — calculée une seule fois par image/classe
            iou_matrix = np.zeros((n_pred, n_gt))
            for i, pred_inst in enumerate(pred_instances):
                for j, gt_mask in enumerate(gt_class_masks):
                    iou_val = calculate_iou_masks(pred_inst['mask'], gt_mask)
                    iou_matrix[i, j] = iou_val

            # IoU stats: meilleure correspondance GT par prédiction
            if n_gt > 0:
                for i in range(n_pred):
                    best_j = int(np.argmax(iou_matrix[i]))
                    self.mask_ious.append(iou_matrix[i, best_j])

            # L'aire de la composante connexe sert de proxy pour le score de confiance
            for i, pred_inst in enumerate(pred_instances):
                self.detections[class_id].append({
                    'score': float(pred_inst['area']),
                    'ious': iou_matrix[i].copy(),
                    'img_idx': self._img_idx
                })

        self._img_idx += 1

    def _compute_ap(self, class_id, iou_thresh):
        """AP = aire sous la courbe PR pour une classe à un seuil IoU donné."""
        n_gt = self.n_gts[class_id]
        dets = self.detections[class_id]
        if n_gt == 0 or not dets:
            return 0.0

        dets_sorted = sorted(dets, key=lambda d: d['score'], reverse=True)
        matched = defaultdict(set)  # img_idx -> ensemble d'indices GT matchés
        tp_list, fp_list = [], []

        for d in dets_sorted:
            ious = d['ious']
            img_idx = d['img_idx']
            best_iou, best_j = 0.0, -1

            for j, v in enumerate(ious):
                if j not in matched[img_idx] and v > best_iou:
                    best_iou, best_j = v, j

            if best_iou >= iou_thresh:
                tp_list.append(1); fp_list.append(0)
                matched[img_idx].add(best_j)
            else:
                tp_list.append(0); fp_list.append(1)

        tp_cum = np.cumsum(tp_list, dtype=float)
        fp_cum = np.cumsum(fp_list, dtype=float)
        recalls = tp_cum / n_gt
        precisions = tp_cum / (tp_cum + fp_cum)
        return calculate_ap(recalls, precisions)

    def _compute_prf(self, class_id, iou_thresh, score_thresh=0.0):
        """TP/FP/FN et Precision/Recall/F1 (score_thresh=0 : toutes les composantes incluses)."""
        n_gt = self.n_gts[class_id]
        dets = [d for d in self.detections[class_id] if d['score'] >= score_thresh]
        n_pred = len(dets)

        if n_gt == 0 and n_pred == 0:
            return {'TP': 0, 'FP': 0, 'FN': 0, 'Precision': 0.0, 'Recall': 0.0, 'F1': 0.0}
        if n_gt == 0:
            return {'TP': 0, 'FP': n_pred, 'FN': 0, 'Precision': 0.0, 'Recall': 0.0, 'F1': 0.0}
        if n_pred == 0:
            return {'TP': 0, 'FP': 0, 'FN': n_gt, 'Precision': 0.0, 'Recall': 0.0, 'F1': 0.0}

        dets_sorted = sorted(dets, key=lambda d: d['score'], reverse=True)
        matched = defaultdict(set)
        tp = fp = 0

        for d in dets_sorted:
            ious = d['ious']
            img_idx = d['img_idx']
            best_iou, best_j = 0.0, -1

            for j, v in enumerate(ious):
                if j not in matched[img_idx] and v > best_iou:
                    best_iou, best_j = v, j

            if best_iou >= iou_thresh:
                tp += 1; matched[img_idx].add(best_j)
            else:
                fp += 1

        fn = n_gt - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return {'TP': tp, 'FP': fp, 'FN': fn,
                'Precision': precision, 'Recall': recall, 'F1': f1}

    def compute_metrics(self):
        """Calculer toutes les métriques finales."""
        results = {'per_class': {}, 'overall': {}, 'iou_stats': {}, 'semantic': {}}

        # --- AP et PRF par classe ---
        for class_id in range(1, self.num_classes):
            class_name = self.class_names[class_id]
            results['per_class'][class_name] = {}
            for iou_thresh in self.iou_thresholds:
                prf = self._compute_prf(class_id, iou_thresh, score_thresh=0.0)
                prf['AP'] = self._compute_ap(class_id, iou_thresh)
                results['per_class'][class_name][f'iou_{iou_thresh}'] = prf

        # --- Métriques globales micro-moyennées ---
        for iou_thresh in self.iou_thresholds:
            total_tp = sum(results['per_class'][self.class_names[c]][f'iou_{iou_thresh}']['TP']
                           for c in range(1, self.num_classes))
            total_fp = sum(results['per_class'][self.class_names[c]][f'iou_{iou_thresh}']['FP']
                           for c in range(1, self.num_classes))
            total_fn = sum(results['per_class'][self.class_names[c]][f'iou_{iou_thresh}']['FN']
                           for c in range(1, self.num_classes))
            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            results['overall'][f'iou_{iou_thresh}'] = {
                'TP': total_tp, 'FP': total_fp, 'FN': total_fn,
                'Precision': precision, 'Recall': recall, 'F1': f1
            }

        # --- mAP@50 = moyenne de AP@50 sur toutes les classes ---
        results['mAP50'] = float(np.mean([
            results['per_class'][self.class_names[c]]['iou_0.5']['AP']
            for c in range(1, self.num_classes)
        ]))

        # --- mAP@50:95 = moyenne de AP sur toutes les classes ET tous les seuils ---
        results['mAP50_95'] = float(np.mean([
            results['per_class'][self.class_names[c]][f'iou_{t}']['AP']
            for c in range(1, self.num_classes)
            for t in self.iou_thresholds
        ]))

        # --- AP par classe ---
        results['mAP_per_class'] = {}
        for class_id in range(1, self.num_classes):
            class_name = self.class_names[class_id]
            results['mAP_per_class'][class_name] = {
                'AP50': results['per_class'][class_name]['iou_0.5']['AP'],
                'AP50_95': float(np.mean([
                    results['per_class'][class_name][f'iou_{t}']['AP']
                    for t in self.iou_thresholds
                ]))
            }

        # --- Stats IoU ---
        if self.mask_ious:
            results['iou_stats']['mask_iou_mean'] = float(np.mean(self.mask_ious))
            results['iou_stats']['mask_iou_std'] = float(np.std(self.mask_ious))
            results['iou_stats']['mask_iou_median'] = float(np.median(self.mask_ious))

        # --- Métriques sémantiques ---
        cm = self.confusion_matrix
        intersection = np.diag(cm)
        union = cm.sum(axis=1) + cm.sum(axis=0) - intersection
        iou_per_class = intersection / (union + 1e-10)
        valid_classes = cm.sum(axis=1) > 0
        results['semantic']['mIoU'] = float(np.mean(iou_per_class[valid_classes]))
        results['semantic']['pixel_accuracy'] = float(np.diag(cm).sum() / (cm.sum() + 1e-10))
        results['semantic']['iou_per_class'] = {
            self.class_names[i]: float(iou_per_class[i]) for i in range(self.num_classes)
        }

        return results


# =============================================================================
# VISUALISATION (identique à Mask R-CNN)
# =============================================================================

def plot_metrics(results, output_dir):
    """Créer les graphiques des métriques"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Graphique AP par classe
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    class_names = list(results['mAP_per_class'].keys())
    ap50_values = [results['mAP_per_class'][c]['AP50'] for c in class_names]
    ap50_95_values = [results['mAP_per_class'][c]['AP50_95'] for c in class_names]
    
    x = np.arange(len(class_names))
    width = 0.35
    
    axes[0].bar(x - width/2, ap50_values, width, label='AP@50', color='steelblue')
    axes[0].bar(x + width/2, ap50_95_values, width, label='AP@50:95', color='coral')
    axes[0].set_xlabel('Classes')
    axes[0].set_ylabel('Average Precision')
    axes[0].set_title('AP par classe')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(class_names, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1)
    
    # 2. Precision, Recall, F1 par classe
    precisions = [results['per_class'][c]['iou_0.5']['Precision'] for c in class_names]
    recalls = [results['per_class'][c]['iou_0.5']['Recall'] for c in class_names]
    f1s = [results['per_class'][c]['iou_0.5']['F1'] for c in class_names]
    
    width = 0.25
    axes[1].bar(x - width, precisions, width, label='Precision', color='green')
    axes[1].bar(x, recalls, width, label='Recall', color='blue')
    axes[1].bar(x + width, f1s, width, label='F1-Score', color='red')
    axes[1].set_xlabel('Classes')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Precision / Recall / F1 par classe (IoU=0.5)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(class_names, rotation=45, ha='right')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_per_class.png'), dpi=150)
    plt.close()
    
    # 3. Métriques vs seuil IoU
    fig, ax = plt.subplots(figsize=(10, 6))
    
    iou_thresholds = CONFIG['iou_thresholds']
    global_precisions = [results['overall'][f'iou_{t}']['Precision'] for t in iou_thresholds]
    global_recalls = [results['overall'][f'iou_{t}']['Recall'] for t in iou_thresholds]
    global_f1s = [results['overall'][f'iou_{t}']['F1'] for t in iou_thresholds]
    
    ax.plot(iou_thresholds, global_precisions, 'o-', label='Precision', linewidth=2, markersize=8)
    ax.plot(iou_thresholds, global_recalls, 's-', label='Recall', linewidth=2, markersize=8)
    ax.plot(iou_thresholds, global_f1s, '^-', label='F1-Score', linewidth=2, markersize=8)
    
    ax.set_xlabel('Seuil IoU')
    ax.set_ylabel('Score')
    ax.set_title('Métriques globales vs Seuil IoU')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    ax.set_xlim(0.45, 1.0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_vs_iou.png'), dpi=150)
    plt.close()
    
    print(f"📊 Graphiques sauvegardés dans: {output_dir}")


def generate_report(results, output_dir):
    """Générer un rapport complet (identique à Mask R-CNN)"""
    
    report_path = os.path.join(output_dir, 'evaluation_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("   RAPPORT D'ÉVALUATION - DeepLabV3+ CADASTRAL\n")
        f.write("=" * 70 + "\n")
        f.write(f"   Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        
        # Résumé principal (identique à Mask R-CNN)
        f.write("📊 RÉSUMÉ DES MÉTRIQUES PRINCIPALES\n")
        f.write("-" * 50 + "\n")
        f.write(f"   mAP@50:        {results['mAP50']:.4f} ({results['mAP50']*100:.2f}%)\n")
        f.write(f"   mAP@50:95:     {results['mAP50_95']:.4f} ({results['mAP50_95']*100:.2f}%)\n")
        f.write(f"\n   Precision@50:  {results['overall']['iou_0.5']['Precision']:.4f}\n")
        f.write(f"   Recall@50:     {results['overall']['iou_0.5']['Recall']:.4f}\n")
        f.write(f"   F1-Score@50:   {results['overall']['iou_0.5']['F1']:.4f}\n")
        
        if results.get('iou_stats'):
            f.write(f"\n   IoU moyen (masques): {results['iou_stats'].get('mask_iou_mean', 0):.4f}\n")
        
        # Métriques sémantiques
        f.write("\n\n📊 MÉTRIQUES SÉMANTIQUES (BONUS)\n")
        f.write("-" * 50 + "\n")
        f.write(f"   mIoU:            {results['semantic']['mIoU']:.4f} ({results['semantic']['mIoU']*100:.2f}%)\n")
        f.write(f"   Pixel Accuracy:  {results['semantic']['pixel_accuracy']:.4f} ({results['semantic']['pixel_accuracy']*100:.2f}%)\n")
        
        # Métriques par classe
        f.write("\n\n📋 MÉTRIQUES PAR CLASSE (IoU=0.5)\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Classe':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AP50':>10}\n")
        f.write("-" * 65 + "\n")
        
        for class_name in results['per_class']:
            metrics = results['per_class'][class_name]['iou_0.5']
            ap50 = results['mAP_per_class'][class_name]['AP50']
            f.write(f"{class_name:<25} {metrics['Precision']:>10.4f} {metrics['Recall']:>10.4f} "
                   f"{metrics['F1']:>10.4f} {ap50:>10.4f}\n")
        
        # Détails TP/FP/FN
        f.write("\n\n📈 DÉTAILS TP/FP/FN PAR CLASSE (IoU=0.5)\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Classe':<25} {'TP':>8} {'FP':>8} {'FN':>8}\n")
        f.write("-" * 50 + "\n")
        
        for class_name in results['per_class']:
            metrics = results['per_class'][class_name]['iou_0.5']
            f.write(f"{class_name:<25} {metrics['TP']:>8} {metrics['FP']:>8} {metrics['FN']:>8}\n")
        
        f.write("\n" + "=" * 70 + "\n")
    
    print(f"📄 Rapport sauvegardé: {report_path}")
    return report_path


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("   ÉVALUATION DeepLabV3+ - Segmentation des Toitures")
    print("   (Métriques identiques à Mask R-CNN pour comparaison)")
    print("=" * 70)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📱 Device: {device}")
    
    # Créer le dossier de sortie
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    
    # Charger le dataset
    print("\n📂 Chargement du dataset...")
    dataset = EvalDataset(
        CONFIG["images_dir"],
        CONFIG["annotations_file"],
        CONFIG["image_size"]
    )
    
    # Charger le modèle
    print("\n🧠 Chargement du modèle...")
    num_classes = len(CONFIG["classes"])
    model = load_model(CONFIG["model_path"], num_classes, CONFIG["backbone"], device)
    
    # Initialiser le calculateur de métriques
    metrics_calc = MetricsCalculator(
        num_classes=num_classes,
        class_names=CONFIG["classes"],
        iou_thresholds=CONFIG["iou_thresholds"]
    )
    
    # Évaluation
    print("\n📊 Calcul des métriques...")
    model.eval()
    
    for idx in tqdm(range(len(dataset)), desc="Évaluation"):
        image_tensor, mask_gt_tensor, target = dataset[idx]
        
        with torch.no_grad():
            output = model(image_tensor.unsqueeze(0).to(device))
            pred = torch.argmax(output['out'], dim=1).squeeze().cpu().numpy()
        
        mask_gt = mask_gt_tensor.numpy()
        
        # Ajouter pour calcul des métriques
        metrics_calc.add_image(
            pred_semantic=pred,
            gt_semantic=mask_gt,
            gt_instances=target['masks'],
            gt_labels=target['labels']
        )
    
    # Calculer les métriques finales
    results = metrics_calc.compute_metrics()
    
    # Affichage (identique à Mask R-CNN)
    print("\n" + "=" * 70)
    print("   📊 RÉSULTATS DE L'ÉVALUATION")
    print("=" * 70)
    
    print(f"\n🎯 MÉTRIQUES PRINCIPALES")
    print(f"   {'─' * 40}")
    print(f"   mAP@50:        {results['mAP50']:.4f} ({results['mAP50']*100:.2f}%)")
    print(f"   mAP@50:95:     {results['mAP50_95']:.4f} ({results['mAP50_95']*100:.2f}%)")
    print(f"\n   Precision@50:  {results['overall']['iou_0.5']['Precision']:.4f}")
    print(f"   Recall@50:     {results['overall']['iou_0.5']['Recall']:.4f}")
    print(f"   F1-Score@50:   {results['overall']['iou_0.5']['F1']:.4f}")
    
    if results.get('iou_stats'):
        print(f"\n   IoU moyen (masques): {results['iou_stats'].get('mask_iou_mean', 0):.4f}")
    
    print(f"\n🎯 MÉTRIQUES SÉMANTIQUES")
    print(f"   {'─' * 40}")
    print(f"   mIoU:           {results['semantic']['mIoU']:.4f} ({results['semantic']['mIoU']*100:.2f}%)")
    print(f"   Pixel Accuracy: {results['semantic']['pixel_accuracy']:.4f}")
    
    print(f"\n📋 PAR CLASSE (IoU=0.5)")
    print(f"   {'─' * 40}")
    for class_name in results['per_class']:
        metrics = results['per_class'][class_name]['iou_0.5']
        print(f"   {class_name}:")
        print(f"      Precision: {metrics['Precision']:.4f} | Recall: {metrics['Recall']:.4f} | F1: {metrics['F1']:.4f}")
    
    # Sauvegarder les résultats
    results_path = os.path.join(CONFIG["output_dir"], "metrics.json")
    
    def convert_to_serializable(obj):
        if isinstance(obj, defaultdict):
            return dict(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    results_serializable = json.loads(
        json.dumps(results, default=convert_to_serializable)
    )
    
    with open(results_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    print(f"\n💾 Métriques sauvegardées: {results_path}")
    
    # Générer les graphiques
    plot_metrics(results, CONFIG["output_dir"])
    
    # Générer le rapport
    generate_report(results, CONFIG["output_dir"])
    
    print("\n" + "=" * 70)
    print("   ✅ ÉVALUATION TERMINÉE")
    print("=" * 70)


if __name__ == "__main__":
    main()
