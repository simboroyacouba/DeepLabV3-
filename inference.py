"""
Inférence DeepLabV3+ - Prédiction sur nouvelles images
Segmentation des toitures cadastrales

Fonctionnalités:
- Temps d'inférence par image
- Résumé global pour les dossiers
- Export des masques
- Rapports JSON détaillés
"""

import os
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms.functional as TF
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from scipy import ndimage
from pathlib import Path
from datetime import datetime
import time
import json


# =============================================================================
# CONFIGURATION
# =============================================================================

CLASSES = [
    "__background__",
    "toiture_tole_ondulee",
    "toiture_tole_bac",
    "toiture_tuile",
    "toiture_dalle"
]

COLORS = {
    "__background__": (0, 0, 0),
    "toiture_tole_ondulee": (255, 0, 0),
    "toiture_tole_bac": (0, 255, 0),
    "toiture_tuile": (0, 0, 255),
    "toiture_dalle": (255, 165, 0),
}

CONFIG = {
    "model_path": os.getenv("SEGMENTATION_MODEL_PATH", "./output/best_model.pth"),
    "input_dir": os.getenv("SEGMENTATION_TEST_IMAGES_DIR", "./test_images"),
    "output_dir": os.getenv("SEGMENTATION_OUTPUT_DIR", "./predictions"),
    "backbone": os.getenv("SEGMENTATION_BACKBONE", "resnet50"),
    "image_size": 512,
    "export_masks": False,
    "show_display": False,
}


# =============================================================================
# UTILITAIRES
# =============================================================================

def format_time(seconds):
    if seconds < 1:
        return f"{seconds*1000:.1f} ms"
    elif seconds < 60:
        return f"{seconds:.2f} s"
    else:
        return f"{int(seconds//60)}m {seconds%60:.1f}s"


# =============================================================================
# MODÈLE
# =============================================================================

def get_model(num_classes, backbone="resnet50"):
    if backbone == "resnet50":
        model = deeplabv3_resnet50(weights=None)
    else:
        model = deeplabv3_resnet101(weights=None)
    
    model.classifier = DeepLabHead(2048, num_classes)
    if model.aux_classifier is not None:
        model.aux_classifier = nn.Sequential(
            nn.Conv2d(1024, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1)
        )
    return model


def load_model(checkpoint_path, device, backbone="resnet50"):
    model = get_model(len(CLASSES), backbone)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"✅ Modèle chargé: {checkpoint_path}")
    return model


# =============================================================================
# INFÉRENCE
# =============================================================================

def predict(model, image_path, device, image_size=512):
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    
    image_resized = image.resize((image_size, image_size), Image.BILINEAR)
    image_tensor = TF.to_tensor(image_resized)
    image_tensor = TF.normalize(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    start_time = time.time()
    with torch.no_grad():
        output = model(image_tensor.unsqueeze(0).to(device))
        pred_mask = torch.argmax(output['out'], dim=1).squeeze().cpu().numpy()
    inference_time = time.time() - start_time
    
    pred_pil = Image.fromarray(pred_mask.astype(np.uint8))
    pred_pil = pred_pil.resize(original_size, Image.NEAREST)
    pred_mask = np.array(pred_pil)
    
    return image, pred_mask, inference_time


def extract_instances(pred_mask):
    instances = []
    for class_id in range(1, len(CLASSES)):
        binary_mask = (pred_mask == class_id).astype(np.uint8)
        labeled_array, num_features = ndimage.label(binary_mask)
        
        for i in range(1, num_features + 1):
            instance_mask = (labeled_array == i).astype(np.uint8)
            if instance_mask.sum() > 100:
                rows = np.any(instance_mask, axis=1)
                cols = np.any(instance_mask, axis=0)
                if rows.any() and cols.any():
                    y1, y2 = np.where(rows)[0][[0, -1]]
                    x1, x2 = np.where(cols)[0][[0, -1]]
                    instances.append({
                        'mask': instance_mask, 'box': [x1, y1, x2+1, y2+1],
                        'class_id': class_id, 'class_name': CLASSES[class_id],
                        'surface_px': int(instance_mask.sum())
                    })
    return instances


# =============================================================================
# VISUALISATION
# =============================================================================

def visualize_predictions(image, pred_mask, instances, inference_time, output_path=None, show=True):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(image)
    axes[0].set_title("Image originale")
    axes[0].axis('off')
    
    colored_mask = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
    for class_id, class_name in enumerate(CLASSES):
        color = COLORS.get(class_name, (128, 128, 128))
        colored_mask[pred_mask == class_id] = color
    axes[1].imshow(colored_mask)
    axes[1].set_title("Segmentation sémantique")
    axes[1].axis('off')
    
    axes[2].imshow(image)
    overlay = np.zeros((*np.array(image).shape[:2], 4))
    
    for inst in instances:
        color = COLORS.get(inst['class_name'], (128, 128, 128))
        color_norm = [c/255 for c in color]
        mask = inst['mask']
        if mask.shape[:2] != overlay.shape[:2]:
            mask = np.array(Image.fromarray(mask.astype(np.uint8)).resize(
                (overlay.shape[1], overlay.shape[0]), Image.NEAREST))
        overlay[mask > 0] = [*color_norm, 0.5]
        
        x1, y1, x2, y2 = inst['box']
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2,
                                  edgecolor=color_norm, facecolor='none')
        axes[2].add_patch(rect)
        axes[2].text(x1, y1-5, f"{inst['class_name']}\n{inst['surface_px']:,} px",
                     fontsize=8, color='white',
                     bbox=dict(boxstyle='round', facecolor=color_norm, alpha=0.8))
    
    axes[2].imshow(overlay)
    axes[2].set_title(f"Instances ({len(instances)} objets) | ⏱️ {format_time(inference_time)}")
    axes[2].axis('off')
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def export_masks(pred_mask, instances, output_dir, image_name):
    os.makedirs(output_dir, exist_ok=True)
    Image.fromarray(pred_mask.astype(np.uint8)).save(os.path.join(output_dir, "semantic_mask.png"))
    for i, inst in enumerate(instances):
        mask = (inst['mask'] > 0).astype(np.uint8) * 255
        Image.fromarray(mask).save(os.path.join(output_dir, f"{i:02d}_{inst['class_name']}.png"))


def generate_report(instances, image_name, inference_time):
    report = {
        'image': image_name,
        'timestamp': datetime.now().isoformat(),
        'inference_time_ms': inference_time * 1000,
        'total_objects': len(instances),
        'surfaces_by_class': {c: {'count': 0, 'total_surface_px': 0} for c in CLASSES[1:]},
        'details': []
    }
    
    for i, inst in enumerate(instances):
        report['surfaces_by_class'][inst['class_name']]['count'] += 1
        report['surfaces_by_class'][inst['class_name']]['total_surface_px'] += inst['surface_px']
        report['details'].append({
            'id': i, 'class': inst['class_name'],
            'surface_px': inst['surface_px'], 'bbox': inst['box']
        })
    return report


# =============================================================================
# RÉSUMÉ GLOBAL
# =============================================================================

def generate_summary(all_reports, output_dir, total_processing_time):
    summary = {
        'timestamp': datetime.now().isoformat(),
        'model': 'DeepLabV3+',
        'total_images': len(all_reports),
        'total_processing_time_s': total_processing_time,
        'avg_inference_time_ms': 0,
        'total_objects': 0,
        'objects_by_class': {c: 0 for c in CLASSES[1:]},
        'surfaces_by_class': {c: 0 for c in CLASSES[1:]},
        'per_image_stats': []
    }
    
    total_inference_time = 0
    for report in all_reports:
        total_inference_time += report['inference_time_ms']
        summary['total_objects'] += report['total_objects']
        for class_name, data in report['surfaces_by_class'].items():
            summary['objects_by_class'][class_name] += data['count']
            summary['surfaces_by_class'][class_name] += data['total_surface_px']
        summary['per_image_stats'].append({
            'image': report['image'],
            'objects': report['total_objects'],
            'inference_time_ms': report['inference_time_ms']
        })
    
    summary['avg_inference_time_ms'] = total_inference_time / len(all_reports) if all_reports else 0
    
    with open(os.path.join(output_dir, "summary.json"), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    total_surface = sum(summary['surfaces_by_class'].values())
    with open(os.path.join(output_dir, "summary.txt"), 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("   RÉSUMÉ D'INFÉRENCE - DEEPLABV3+ CADASTRAL\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"📅 Date: {summary['timestamp']}\n")
        f.write(f"🖼️  Images traitées: {summary['total_images']}\n")
        f.write(f"⏱️  Temps total: {format_time(summary['total_processing_time_s'])}\n")
        f.write(f"⏱️  Temps moyen/image: {summary['avg_inference_time_ms']:.1f} ms\n")
        f.write(f"🎯 Total objets: {summary['total_objects']}\n\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Classe':<25} {'Objets':>10} {'Surface (px)':>15} {'%':>10}\n")
        f.write("-" * 70 + "\n")
        for class_name in CLASSES[1:]:
            count = summary['objects_by_class'][class_name]
            surface = summary['surfaces_by_class'][class_name]
            pct = (surface / total_surface * 100) if total_surface > 0 else 0
            f.write(f"{class_name:<25} {count:>10} {surface:>15,} {pct:>9.1f}%\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'TOTAL':<25} {summary['total_objects']:>10} {total_surface:>15,} {'100.0%':>10}\n")
        f.write("\n" + "-" * 70 + "\n")
        f.write("DÉTAILS PAR IMAGE\n" + "-" * 70 + "\n")
        f.write(f"{'Image':<40} {'Objets':>10} {'Temps (ms)':>15}\n")
        f.write("-" * 70 + "\n")
        for stat in summary['per_image_stats']:
            img_name = stat['image'][:38] + '..' if len(stat['image']) > 40 else stat['image']
            f.write(f"{img_name:<40} {stat['objects']:>10} {stat['inference_time_ms']:>15.1f}\n")
        f.write("=" * 70 + "\n")
    
    return summary


def print_summary(summary):
    print("\n" + "=" * 70)
    print("   📊 RÉSUMÉ GLOBAL - DEEPLABV3+")
    print("=" * 70)
    print(f"\n   🖼️  Images traitées:     {summary['total_images']}")
    print(f"   ⏱️  Temps total:          {format_time(summary['total_processing_time_s'])}")
    print(f"   ⏱️  Temps moyen/image:    {summary['avg_inference_time_ms']:.1f} ms")
    print(f"   🎯 Total objets:         {summary['total_objects']}")
    
    total_surface = sum(summary['surfaces_by_class'].values())
    print(f"\n   📋 Par classe:")
    for class_name in CLASSES[1:]:
        count = summary['objects_by_class'][class_name]
        surface = summary['surfaces_by_class'][class_name]
        pct = (surface / total_surface * 100) if total_surface > 0 else 0
        if count > 0:
            print(f"      • {class_name}: {count} objets | {surface:,} px ({pct:.1f}%)")
    print("\n" + "=" * 70)


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def process_directory(model, input_dir, output_dir, device, image_size=512, export_masks_flag=False, show_display=False):
    os.makedirs(output_dir, exist_ok=True)
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
    image_paths = sorted([p for p in Path(input_dir).iterdir() if p.suffix.lower() in image_extensions])
    
    if not image_paths:
        print(f"❌ Aucune image trouvée dans {input_dir}")
        return []
    
    print(f"\n🖼️  {len(image_paths)} images à traiter\n")
    
    all_reports = []
    start_total = time.time()
    
    for idx, img_path in enumerate(image_paths, 1):
        print(f"[{idx}/{len(image_paths)}] 🔍 {img_path.name}")
        
        image, pred_mask, inference_time = predict(model, str(img_path), device, image_size)
        instances = extract_instances(pred_mask)
        
        output_path = os.path.join(output_dir, f"{img_path.stem}_pred.png")
        visualize_predictions(image, pred_mask, instances, inference_time, output_path, show=show_display)
        
        if export_masks_flag:
            export_masks(pred_mask, instances, os.path.join(output_dir, "masks", img_path.stem), img_path.stem)
        
        report = generate_report(instances, img_path.name, inference_time)
        all_reports.append(report)
        print(f"   ✅ {report['total_objects']} objets | ⏱️ {report['inference_time_ms']:.1f} ms")
    
    total_processing_time = time.time() - start_total
    
    with open(os.path.join(output_dir, "reports.json"), 'w', encoding='utf-8') as f:
        json.dump(all_reports, f, indent=2, ensure_ascii=False)
    
    summary = generate_summary(all_reports, output_dir, total_processing_time)
    print_summary(summary)
    
    print(f"\n📁 Résultats sauvegardés dans: {output_dir}")
    return all_reports


# =============================================================================
# MAIN
# =============================================================================

def main():
    # Configuration depuis variables d'environnement
    model_path = CONFIG["model_path"]
    input_dir = CONFIG["input_dir"]
    output_dir = CONFIG["output_dir"]
    backbone = CONFIG["backbone"]
    image_size = CONFIG["image_size"]
    export_masks_flag = CONFIG["export_masks"]
    show_display = CONFIG["show_display"]
    
    # Vérifications
    if not os.path.exists(model_path):
        print(f"❌ Modèle non trouvé: {model_path}")
        print(f"   Définissez SEGMENTATION_MODEL_PATH")
        return
    
    if not os.path.exists(input_dir):
        print(f"❌ Dossier d'images non trouvé: {input_dir}")
        print(f"   Définissez SEGMENTATION_TEST_IMAGES_DIR")
        return
    
    print("=" * 70)
    print("   🚀 INFÉRENCE DEEPLABV3+ CADASTRAL")
    print("=" * 70)
    print(f"\n📂 Configuration:")
    print(f"   • Modèle:      {model_path}")
    print(f"   • Backbone:    {backbone}")
    print(f"   • Images:      {input_dir}")
    print(f"   • Sortie:      {output_dir}")
    print(f"   • Image size:  {image_size}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   • Device:      {device}")
    
    model = load_model(model_path, device, backbone)
    
    input_path = Path(input_dir)
    
    if input_path.is_dir():
        process_directory(model, str(input_path), output_dir, device, image_size, export_masks_flag, show_display)
    else:
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n🔍 Traitement: {input_path.name}")
        
        image, pred_mask, inference_time = predict(model, str(input_path), device, image_size)
        instances = extract_instances(pred_mask)
        
        output_path = os.path.join(output_dir, f"{input_path.stem}_pred.png")
        visualize_predictions(image, pred_mask, instances, inference_time, output_path, show=show_display)
        
        if export_masks_flag:
            export_masks(pred_mask, instances, os.path.join(output_dir, "masks"), input_path.stem)
        
        report = generate_report(instances, input_path.name, inference_time)
        print(f"\n{'='*60}")
        print(f"📊 RAPPORT - {report['image']}")
        print(f"{'='*60}")
        print(f"   ⏱️  Temps d'inférence: {report['inference_time_ms']:.1f} ms")
        print(f"   🎯 Objets détectés: {report['total_objects']}")
        for class_name, data in report['surfaces_by_class'].items():
            if data['count'] > 0:
                print(f"      • {class_name}: {data['count']} objets, {data['total_surface_px']:,} px")
        print(f"{'='*60}")
        
        with open(os.path.join(output_dir, f"{input_path.stem}_report.json"), 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()