# ============================================================
# MobileNetV2 + SELU + CrossEntropy Comparative Study (PyTorch)
# RTX 4500 Ada GPU Version - Cross-Dataset Testing Comparison
# ============================================================

import os
import time
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import MobileNet_V2_Weights
from torch.amp import autocast, GradScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve,
    matthews_corrcoef, cohen_kappa_score
)

import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- CONFIG ----------------
IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3
NUM_WORKERS = 4 

DATASETS = {
    "140K": r"C:\Users\HP\Desktop\Results\140k_split_dataset",
    "200K": r"C:\Users\HP\Desktop\Results\200k_split_dataset"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- MODEL ----------------
def build_model():
    model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 512),
        nn.SELU(), 
        nn.Linear(512, 2)
    )
    return model.to(device)

# ---------------- DATA ----------------
def get_loaders(train_val_root, test_root):
    """
    Separates the training/validation paths from the testing path 
    to enable clean cross-dataset evaluation.
    """
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = datasets.ImageFolder(os.path.join(train_val_root, "train"), transform)
    val_ds = datasets.ImageFolder(os.path.join(train_val_root, "val"), transform)
    test_ds = datasets.ImageFolder(os.path.join(test_root, "test"), transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, 
                             num_workers=NUM_WORKERS, pin_memory=True)

    return train_loader, val_loader, test_loader

# ---------------- TRAIN ----------------
def train_dataset(name, train_val_root, test_root, cross_name):
    os.makedirs(f"Results/{name}", exist_ok=True)
    
    # Pass the cross-dataset path to get the correct testing loader
    train_loader, val_loader, test_loader = get_loaders(train_val_root, test_root)

    model = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = GradScaler()

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc = 0
    best_model = None
    start_train = time.time()

    for epoch in range(EPOCHS):
        print(f"\n{name} (Testing on {cross_name}) Epoch {epoch+1}/{EPOCHS}")
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            preds = outputs.argmax(1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        train_acc = correct / total
        train_loss = running_loss / len(train_loader)

        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                with autocast(device_type="cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item()
                preds = outputs.argmax(1)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()

        val_acc = val_correct / val_total
        val_loss /= len(val_loader)

        print(f"Train Loss: {train_loss:.4f} | Train Acc : {train_acc:.4f}")
        print(f"Val Loss  : {val_loss:.4f} | Val Acc   : {val_acc:.4f}")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            best_model = copy.deepcopy(model.state_dict())

    training_time = time.time() - start_train
    torch.save(best_model, f"Results/{name}/best_model.pth")
    model.load_state_dict(best_model)

    # EVALUATION ON CROSS TEST SET
    print(f"Evaluating Model {name} on {cross_name} Test Set...")
    y_true, y_pred, y_prob = [], [], []
    start_inf = time.time()

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            with autocast(device_type="cuda"):
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                preds = outputs.argmax(1).cpu().numpy()

            y_prob.extend(probs)
            y_pred.extend(preds)
            y_true.extend(labels.numpy())

    inference_time = time.time() - start_inf
    fps = len(y_true) / inference_time

    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)
    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

    metrics = {
        "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1, "AUC": auc,
        "Specificity": specificity, "Sensitivity": sensitivity, "MCC": mcc, "Kappa": kappa,
        "Training_Time": training_time, "Inference_Time": inference_time, "FPS": fps
    }

    pd.DataFrame([metrics]).to_csv(f"Results/{name}/metrics.csv", index=False)
    
    return metrics, history, {"y_true": y_true, "y_prob": y_prob, "cm": cm}


# ---------------- RUN PIPELINE ----------------
if __name__ == '__main__':
    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    all_metrics = {}
    all_histories = {}
    evaluation_raw = {}

    # Define explicitly which dataset uses which for testing
    cross_mapping = {
        "140K": "200K",
        "200K": "140K"
    }

    for ds_name, ds_path in DATASETS.items():
        cross_name = cross_mapping[ds_name]
        cross_path = DATASETS[cross_name]
        
        # Train/Val on ds_path, but Test on cross_path
        metrics, history, raw_eval = train_dataset(ds_name, ds_path, cross_path, cross_name)
        
        all_metrics[ds_name] = metrics
        all_histories[ds_name] = history
        evaluation_raw[ds_name] = raw_eval

    # Custom column names to indicate the cross testing in the final files
    comparison = pd.DataFrame({
        "Metric": list(all_metrics["140K"].keys()),
        "140K Train (Tested on 200K)": list(all_metrics["140K"].values()),
        "200K Train (Tested on 140K)": list(all_metrics["200K"].values())
    })

    os.makedirs("Results", exist_ok=True)
    comparison.to_csv("Results/comparison.csv", index=False)
    comparison.to_excel("Results/comparison.xlsx", index=False)

    print("\n--- Final Cross-Evaluation Summary Table ---")
    print(comparison)


    # ---------------- VISUALIZATION BLOCK ----------------
    print("\nGenerating Performance Graphs...")
    epochs_range = np.arange(1, EPOCHS + 1)
    sns.set_theme(style="whitegrid")

    # Graph 1: Loss curves
    plt.figure(figsize=(9, 5.5))
    plt.plot(epochs_range, all_histories["140K"]["train_loss"], color='royalblue', marker='o', label='140K Train Loss')
    plt.plot(epochs_range, all_histories["140K"]["val_loss"], color='royalblue', linestyle='--', marker='s', label='140K Val Loss')
    plt.plot(epochs_range, all_histories["200K"]["train_loss"], color='darkorange', marker='o', label='200K Train Loss')
    plt.plot(epochs_range, all_histories["200K"]["val_loss"], color='darkorange', linestyle='--', marker='s', label='200K Val Loss')
    plt.title('Training and Validation Loss Comparison (SELU)', fontsize=13, fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.xticks(epochs_range)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig('Results/loss_comparison_curve.png', dpi=300)
    plt.close()

    # Graph 2: Accuracy curves
    plt.figure(figsize=(9, 5.5))
    plt.plot(epochs_range, all_histories["140K"]["train_acc"], color='royalblue', marker='o', label='140K Train Acc')
    plt.plot(epochs_range, all_histories["140K"]["val_acc"], color='royalblue', linestyle='--', marker='s', label='140K Val Acc')
    plt.plot(epochs_range, all_histories["200K"]["train_acc"], color='darkorange', marker='o', label='200K Train Acc')
    plt.plot(epochs_range, all_histories["200K"]["val_acc"], color='darkorange', linestyle='--', marker='s', label='200K Val Acc')
    plt.title('Training and Validation Accuracy Comparison (SELU)', fontsize=13, fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy Score')
    plt.xticks(epochs_range)
    plt.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    plt.savefig('Results/accuracy_comparison_curve.png', dpi=300)
    # Graph 3: Grouped Bar Chart for Core Final Metrics
    core_metrics = ["Accuracy", "Precision", "Recall", "F1", "AUC", "Specificity", "Sensitivity"]
    filtered_comp = comparison[comparison['Metric'].isin(core_metrics)]

    plt.figure(figsize=(10, 6))
    x_idx = np.arange(len(filtered_comp))
    width = 0.35

    plt.bar(x_idx - width/2, filtered_comp['140K Train (Tested on 200K)'], width, label='140K Model (Tested on 200K)', color='royalblue')
    plt.bar(x_idx + width/2, filtered_comp['200K Train (Tested on 140K)'], width, label='200K Model (Tested on 140K)', color='darkorange')

    plt.title('Cross-Dataset Final Evaluation Benchmarks', fontsize=13, fontweight='bold', pad=15)
    plt.ylabel('Score Output')
    plt.xticks(x_idx, filtered_comp['Metric'], rotation=30, ha='right')
    plt.ylim(0.0, 1.02)
    plt.legend(loc='lower left')
    plt.grid(True, axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('Results/final_metrics_bar_chart.png', dpi=300)
    plt.close()

    # Graph 4: True ROC Curve
    plt.figure(figsize=(7, 6))
    colors = {"140K": "royalblue", "200K": "darkorange"}

    for ds_name in DATASETS.keys():
        true_labels = evaluation_raw[ds_name]["y_true"]
        pred_probs = evaluation_raw[ds_name]["y_prob"]
        
        fpr, tpr, _ = roc_curve(true_labels, pred_probs)
        auc_score = all_metrics[ds_name]["AUC"]
        
        cross_lbl = cross_mapping[ds_name]
        plt.plot(fpr, tpr, color=colors[ds_name], lw=2, 
                 label=f'{ds_name} Model on {cross_lbl} (AUC = {auc_score:.4f})')

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Cross-Dataset Comparison', fontsize=13, fontweight='bold')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('Results/roc_comparison.png', dpi=300)
    plt.close()

    # Graph 5: True Confusion Matrix Heatmap
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for i, ds_name in enumerate(DATASETS.keys()):
        real_cm = evaluation_raw[ds_name]["cm"]
        cross_lbl = cross_mapping[ds_name]
        
        sns.heatmap(real_cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
        axes[i].set_title(f'Model {ds_name} Evaluated on {cross_lbl} Test')
        axes[i].set_ylabel('True Label')
        axes[i].set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig('Results/confusion_matrices.png', dpi=300)
    plt.close()

    print("All cross-evaluation figures successfully rendered and saved under 'Results/'.")
