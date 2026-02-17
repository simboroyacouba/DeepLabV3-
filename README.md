# DeepLabV3+ - Segmentation des Toitures Cadastrales

Projet de segmentation pour la classification automatique des types de toitures.
**Structure identique à Mask R-CNN pour comparaison équitable.**

## Structure des deux projets

```
maskrcnn_cadastral/          deeplab_cadastral/
├── train.py                 ├── train.py
├── evaluate.py              ├── evaluate.py
├── inference.py             ├── inference.py
├── verify_dataset.py        ├── verify_dataset.py
├── requirements.txt         ├── requirements.txt
└── README.md                └── README.md
```

## Métriques identiques pour comparaison

Les deux modèles sont évalués avec **exactement les mêmes métriques** :

| Métrique | Description |
|----------|-------------|
| mAP@50 | Mean Average Precision à IoU=0.5 |
| mAP@50:95 | Moyenne des AP de 0.5 à 0.95 |
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| F1-Score | 2 × (P × R) / (P + R) |
| IoU moyen | Intersection over Union |

## Configuration identique

Les deux projets utilisent :
- **Mêmes classes** : background, toiture_tole_ondulee, toiture_tole_bac, toiture_tuile, toiture_dalle
- **Même split** : 85% train / 15% validation
- **Mêmes hyperparamètres** : 25 epochs, batch_size=2, lr=0.005, SGD optimizer
- **Même scheduler** : StepLR avec step=8, gamma=0.1
- **Même seed** : 42 pour reproductibilité

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

### 1. Vérifier le dataset
```bash
python verify_dataset.py --images chemin/images --annotations chemin/annotations.json
```

### 2. Entraîner
```bash
python train.py
```

### 3. Évaluer
```bash
python evaluate.py
```

### 4. Inférence
```bash
python inference.py --model output/best_model.pth --input image.jpg
```

## Comparaison des modèles

### Différences architecturales

| Aspect | Mask R-CNN | DeepLabV3+ |
|--------|------------|------------|
| Type | Instance segmentation | Semantic segmentation |
| Backbone | ResNet50 + FPN | ResNet50 + ASPP |
| Sortie | Masques par instance | Masque sémantique |
| Détection | Oui (boîtes) | Non |

### Pour la comparaison

DeepLabV3+ génère un masque sémantique, donc pour calculer les mêmes métriques :
1. Le masque sémantique est converti en **composantes connexes**
2. Chaque composante = une "instance" détectée
3. Les métriques TP/FP/FN sont calculées comme pour Mask R-CNN

### Tableau de comparaison attendu

Après entraînement des deux modèles, tu pourras comparer :

```
┌─────────────────┬────────────┬─────────────┐
│ Métrique        │ Mask R-CNN │ DeepLabV3+  │
├─────────────────┼────────────┼─────────────┤
│ mAP@50          │   0.XXXX   │   0.XXXX    │
│ mAP@50:95       │   0.XXXX   │   0.XXXX    │
│ Precision@50    │   0.XXXX   │   0.XXXX    │
│ Recall@50       │   0.XXXX   │   0.XXXX    │
│ F1-Score@50     │   0.XXXX   │   0.XXXX    │
│ IoU moyen       │   0.XXXX   │   0.XXXX    │
│ Temps/epoch     │   XX min   │   XX min    │
│ Temps total     │   XX h     │   XX h      │
└─────────────────┴────────────┴─────────────┘
```

## Fichiers générés

### Entraînement (output/)
```
output/
├── best_model.pth
├── final_model.pth
├── checkpoint_epoch_*.pth
├── history.json
├── training_curves.png
└── training_report.txt
```

### Évaluation (evaluation/)
```
evaluation/
├── metrics.json
├── evaluation_report.txt
├── metrics_per_class.png
└── metrics_vs_iou.png
```

## Auteur

Projet de thèse - Exploitation de l'IA pour l'évaluation cadastrale automatisée
Burkina Faso - SYCAD/DGI
