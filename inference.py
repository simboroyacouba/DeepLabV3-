"""
Inférence DeepLabV3+ - Prédiction sur nouvelles images
Structure identique à Mask R-CNN pour comparaison
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from tqdm import tqdm
from scipy import ndimage
import argparse
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION (identique à Mask R-CNN)
# =============================================================================

CLASSES = [
    "__background__",
    "toiture_tole_ondulee",
    "toiture_tole_bac",
    "toiture_tuile",
    "toiture_dalle"
]

COLORS = {
    "toiture_tole_ondulee": (255, 0, 0),
    "toiture_tole_bac": (0, 255, 0),
    "toiture_tuile": (0, 0, 255),
    "toiture_dalle": (255, 165, 0),
}


# =============================================================================
# MODÈLE
# =============================================================================

def get_model(num_classes, backbone="resnet50"):
    """Créer le modèle DeepLabV3+"""
    
    if backbone == "resnet50":
        model = deeplabv3_resnet50(weights=None)
        in_channels = 2048
    elif backbone == "resnet101":
        model = deeplabv3_resnet101(weights=None)
        in_channels = 2048
    
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


def load_model(checkpoint_path, device, backbone="resnet50"):
    """Charger le modèle depuis un checkpoint"""
    model = get_model(len(CLASSES), backbone)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Modèle chargé depuis: {checkpoint_path}")
    print(f"Epoch: {checkpoint.get('epoch', 'N/A')}, Loss: {checkpoint.get('loss', 'N/A'):.4f}")
    
    return model


# =============================================================================
# INFÉRENCE
# =============================================================================

def predict(model, image_path, device, image_size=512):
    """Prédire sur une image"""
    
    # Charger l'image
    image = Image.open(image_path).convert("RGB")
    original_size = image.size  # (width, height)
    
    # Redimensionner
    image_resized = image.resize((image_size, image_size), Image.BILINEAR)
    
    # Convertir en tensor
    image_tensor = TF.to_tensor(image_resized)
    image_tensor = TF.normalize(image_tensor, 
                                mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
    
    # Inférence
    with torch.no_grad():
        output = model(image_tensor.unsqueeze(0).to(device))
        pred = output['out']
        
        # Probabilités
        probs = torch.softmax(pred, dim=1)
        confidence = probs.max(dim=1)[0].squeeze().cpu().numpy()
        
        # Classe prédite
        pred_mask = torch.argmax(pred, dim=1).squeeze().cpu().numpy()
    
    # Redimensionner à la taille originale
    pred_pil = Image.fromarray(pred_mask.astype(np.uint8))
    pred_pil = pred_pil.resize(original_size, Image.NEAREST)
    pred_mask = np.array(pred_pil)
    
    conf_pil = Image.fromarray((confidence * 255).astype(np.uint8))
    conf_pil = conf_pil.resize(original_size, Image.BILINEAR)
    confidence = np.array(conf_pil) / 255.0
    
    return image, pred_mask, confidence


def extract_instances(pred_mask):
    """
    Extraire les instances (composantes connexes) du masque sémantique
    Pour avoir un comportement similaire à Mask R-CNN
    """
    
    instances = []
    
    for class_id in range(1, len(CLASSES)):  # Ignorer background
        binary_mask = (pred_mask == class_id).astype(np.uint8)
        labeled_array, num_features = ndimage.label(binary_mask)
        
        for i in range(1, num_features + 1):
            instance_mask = (labeled_array == i).astype(np.uint8)
            
            if instance_mask.sum() > 100:  # Ignorer les très petits objets
                # Calculer la bounding box
                rows = np.any(instance_mask, axis=1)
                cols = np.any(instance_mask, axis=0)
                
                if rows.any() and cols.any():
                    y1, y2 = np.where(rows)[0][[0, -1]]
                    x1, x2 = np.where(cols)[0][[0, -1]]
                    
                    instances.append({
                        'mask': instance_mask,
                        'box': [x1, y1, x2 + 1, y2 + 1],
                        'label': class_id,
                        'class_name': CLASSES[class_id],
                        'surface_px': int(instance_mask.sum())
                    })
    
    return instances


def calculate_surface(mask, pixel_size_m2=None):
    """Calculer la surface d'un masque"""
    surface_pixels = np.sum(mask > 0)
    
    if pixel_size_m2 is not None:
        return surface_pixels * pixel_size_m2
    return surface_pixels


# =============================================================================
# VISUALISATION (identique à Mask R-CNN)
# =============================================================================

def visualize_predictions(image, pred_mask, instances, output_path=None, show=True):
    """Visualiser les prédictions avec masques et boîtes (comme Mask R-CNN)"""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Image originale
    axes[0].imshow(image)
    axes[0].set_title("Image originale")
    axes[0].axis('off')
    
    # Image avec prédictions
    axes[1].imshow(image)
    
    # Overlay des masques
    overlay = np.zeros((*np.array(image).shape[:2], 4))
    
    for inst in instances:
        class_name = inst['class_name']
        color = COLORS.get(class_name, (128, 128, 128))
        color_normalized = [c/255 for c in color]
        
        mask = inst['mask']
        box = inst['box']
        
        # Masque
        overlay[mask > 0] = [*color_normalized, 0.5]
        
        # Boîte
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=2,
            edgecolor=color_normalized,
            facecolor='none'
        )
        axes[1].add_patch(rect)
        
        # Label
        surface = inst['surface_px']
        label_text = f"{class_name}\n{surface:,} px"
        axes[1].text(
            x1, y1-10,
            label_text,
            fontsize=8,
            color='white',
            bbox=dict(boxstyle='round', facecolor=color_normalized, alpha=0.8)
        )
    
    axes[1].imshow(overlay)
    axes[1].set_title(f"Prédictions ({len(instances)} objets détectés)")
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Résultat sauvegardé: {output_path}")
    
    if show:
        plt.show()
    
    plt.close()


def generate_report(instances, image_name):
    """Générer un rapport des surfaces détectées (identique à Mask R-CNN)"""
    
    report = {
        'image': image_name,
        'total_objects': len(instances),
        'surfaces_by_class': {},
        'details': []
    }
    
    for class_name in CLASSES[1:]:
        report['surfaces_by_class'][class_name] = {
            'count': 0,
            'total_surface_px': 0
        }
    
    for i, inst in enumerate(instances):
        class_name = inst['class_name']
        surface = inst['surface_px']
        
        report['surfaces_by_class'][class_name]['count'] += 1
        report['surfaces_by_class'][class_name]['total_surface_px'] += surface
        
        report['details'].append({
            'id': i,
            'class': class_name,
            'surface_px': surface,
            'bbox': inst['box']
        })
    
    return report


def print_report(report):
    """Afficher le rapport (identique à Mask R-CNN)"""
    print("\n" + "=" * 50)
    print(f"RAPPORT DE SEGMENTATION - {report['image']}")
    print("=" * 50)
    print(f"Total objets détectés: {report['total_objects']}")
    print("\nSurfaces par classe:")
    print("-" * 50)
    
    for class_name, data in report['surfaces_by_class'].items():
        if data['count'] > 0:
            print(f"  {class_name}:")
            print(f"    - Nombre: {data['count']}")
            print(f"    - Surface totale: {data['total_surface_px']:,} pixels")
    
    print("\nDétails:")
    print("-" * 50)
    for obj in report['details']:
        print(f"  [{obj['id']}] {obj['class']} - {obj['surface_px']:,} px")
    
    print("=" * 50)


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def process_directory(model, input_dir, output_dir, device, image_size=512):
    """Traiter toutes les images d'un répertoire"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
    image_paths = [
        p for p in Path(input_dir).iterdir()
        if p.suffix.lower() in image_extensions
    ]
    
    print(f"\nTraitement de {len(image_paths)} images...")
    
    all_reports = []
    
    for img_path in tqdm(image_paths, desc="Inférence"):
        # Prédiction
        image, pred_mask, confidence = predict(model, str(img_path), device, image_size)
        
        # Extraire les instances
        instances = extract_instances(pred_mask)
        
        # Visualisation
        output_path = os.path.join(output_dir, f"{img_path.stem}_pred.png")
        visualize_predictions(image, pred_mask, instances, output_path, show=False)
        
        # Rapport
        report = generate_report(instances, img_path.name)
        all_reports.append(report)
        print_report(report)
    
    # Sauvegarder tous les rapports
    reports_path = os.path.join(output_dir, "reports.json")
    with open(reports_path, 'w') as f:
        json.dump(all_reports, f, indent=2)
    
    print(f"\nRapports sauvegardés: {reports_path}")
    
    return all_reports


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Inférence DeepLabV3+ Cadastral")
    parser.add_argument("--model", type=str, required=True, help="Chemin vers le checkpoint")
    parser.add_argument("--input", type=str, required=True, help="Image ou dossier d'images")
    parser.add_argument("--output", type=str, default="./predictions", help="Dossier de sortie")
    parser.add_argument("--backbone", type=str, default="resnet50", help="Backbone: resnet50, resnet101")
    parser.add_argument("--image-size", type=int, default=512, help="Taille des images")
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Charger le modèle
    model = load_model(args.model, device, args.backbone)
    
    # Traitement
    input_path = Path(args.input)
    
    if input_path.is_dir():
        process_directory(model, str(input_path), args.output, device, args.image_size)
    else:
        os.makedirs(args.output, exist_ok=True)
        
        image, pred_mask, confidence = predict(model, str(input_path), device, args.image_size)
        instances = extract_instances(pred_mask)
        
        output_path = os.path.join(args.output, f"{input_path.stem}_pred.png")
        visualize_predictions(image, pred_mask, instances, output_path)
        
        report = generate_report(instances, input_path.name)
        print_report(report)


if __name__ == "__main__":
    main()
