import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, precision_recall_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import os
from sklearn.metrics import roc_curve, auc
from itertools import cycle
from sklearn.metrics import precision_recall_curve, average_precision_score

def train_model(model, train_loader, val_loader, num_epochs=20, lr=1e-4, device='cuda', optimizer=None):
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

    best_acc = 0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        print(f"Epoch {epoch + 1}/{num_epochs} - Val Acc: {acc:.2f}%")
        scheduler.step(acc)

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'best_model.pth')

    return model


def evaluate_model(model, test_loader, classes, device='cuda', save_dir='results', return_acc_only=False):
    model.eval()
    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            y_true.extend(labels.numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    # ===================== Quick Accuracy Calculation =====================
    correct = (y_pred == y_true).sum()
    total = len(y_true)
    accuracy = 100 * correct / total

    if return_acc_only:
        return None, accuracy   # Return None, accuracy for hyperparameter tuning

    # ===================== Full Evaluation (Only if return_acc_only=False) =====================

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # ===================== Confusion Matrix =====================
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    # Save the figure
    cm_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close figure to free memory

    print(f"✅ Confusion Matrix saved to: {cm_path}")

    # ===================== Classification Report =====================
    report = classification_report(y_true, y_pred, target_names=classes)
    print("\nClassification Report:\n")
    print(report)

    # Save classification report as text file
    with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)

    print(f"✅ Classification Report saved to: {save_dir}/classification_report.txt")

    # ===================== ROC Curves (One-vs-Rest) ====================

    n_classes = len(classes)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    plt.figure(figsize=(10, 8))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green',
                    'blue', 'yellow', 'purple', 'pink', 'brown'])

    for i, color in zip(range(n_classes), colors):
        fpr[i], tpr[i], _ = roc_curve(y_true == i, y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{classes[i]} (AUC = {roc_auc[i]:0.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC Curves (One-vs-Rest)')
    plt.legend(loc="lower right")
    roc_path = os.path.join(save_dir, 'roc_curves.png')
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ ROC Curves saved to: {roc_path}")

    # ===================== Precision-Recall Curves =====================

    plt.figure(figsize=(10, 8))
    for i, color in zip(range(n_classes), colors):
        precision, recall, _ = precision_recall_curve(y_true == i, y_prob[:, i])
        avg_precision = average_precision_score(y_true == i, y_prob[:, i])
        plt.plot(recall, precision, color=color, lw=2,
                 label=f'{classes[i]} (AP = {avg_precision:0.3f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Multi-class Precision-Recall Curves')
    plt.legend(loc="lower left")
    pr_path = os.path.join(save_dir, 'precision_recall_curves.png')
    plt.savefig(pr_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Precision-Recall Curves saved to: {pr_path}")

    return y_true, y_pred, y_prob