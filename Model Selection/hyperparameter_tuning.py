import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import itertools
import joblib
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from dataset import EuroSatDataset
from models import create_model
from train import train_model, evaluate_model

def hyperparameter_search(
        full_dataset,
        saved_model_path='models/AlexNet.pth',
        pca_path='models/pca_8components.pkl',
        num_trials=None,  # if None → search all combinations
        device=None
):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load PCA
    pca = joblib.load(pca_path)
    full_dataset.pca = pca

    # Train/Val/Test Split
    train_idx, temp_idx = train_test_split(range(len(full_dataset)),
                                           test_size=0.40, stratify=full_dataset.labels, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50,
                                         stratify=[full_dataset.labels[i] for i in temp_idx], random_state=42)

    train_ds = Subset(full_dataset, train_idx)
    val_ds = Subset(full_dataset, val_idx)

    # ==================== Expanded Hyperparameter Grid ====================
    param_grid = {
        'lr': [1e-4, 5e-5, 1e-5],
        'batch_size': [32, 64, 128],
        'epochs': [8, 12, 15],
        'optimizer': ['Adam', 'AdamW', 'SGD', 'RMSprop', 'Adagrad'],
        'weight_decay': [0, 1e-5, 1e-4],
        'momentum': [0.9]  # used only with SGD
    }

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    all_combinations = list(itertools.product(*values))

    if num_trials is not None:
        all_combinations = all_combinations[:num_trials]

    print(f"🔍 Starting Hyperparameter Search: {len(all_combinations)} combinations\n")

    best_acc = 0.0
    best_params = None
    best_model = None
    results = []

    for i, combination in enumerate(all_combinations):
        params = dict(zip(keys, combination))
        print(f"\nTrial {i + 1}/{len(all_combinations)} → {params}")

        # Load saved model
        model = create_model(num_classes=10, in_channels=8, model_name='alexnet', pretrained=False)
        model.load_state_dict(torch.load(saved_model_path, map_location=device, weights_only=True))
        model = model.to(device)

        # DataLoader with current batch size
        train_loader = DataLoader(train_ds, batch_size=params['batch_size'], shuffle=True,
                                  num_workers=8, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=params['batch_size'], shuffle=False,
                                num_workers=8, pin_memory=True)

        # ===================== Optimizer Selection =====================
        opt_name = params['optimizer']

        if opt_name == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=params['lr'],
                                   weight_decay=params['weight_decay'])
        elif opt_name == 'AdamW':
            optimizer = optim.AdamW(model.parameters(), lr=params['lr'],
                                    weight_decay=params['weight_decay'])
        elif opt_name == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=params['lr'],
                                  momentum=params.get('momentum', 0.9),
                                  weight_decay=params['weight_decay'])
        elif opt_name == 'RMSprop':
            optimizer = optim.RMSprop(model.parameters(), lr=params['lr'],
                                      weight_decay=params['weight_decay'])
        elif opt_name == 'Adagrad':
            optimizer = optim.Adagrad(model.parameters(), lr=params['lr'],
                                      weight_decay=params['weight_decay'])
        else:
            optimizer = optim.Adam(model.parameters(), lr=params['lr'])

        # Training
        model = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=params['epochs'],
            lr=params['lr'],
            device=device,
            optimizer=optimizer
        )

        # Evaluate
        _, val_acc = evaluate_model(model, val_loader, full_dataset.classes,
                                    device=device, return_acc_only=True)

        results.append((val_acc, params))

        if val_acc > best_acc:
            best_acc = val_acc
            best_params = params
            best_model = model
            print(f"🎉 New Best! Accuracy: {best_acc:.2f}% with {params}")
            torch.save(model.state_dict(), 'models/Best_AlexNet.pth')
            torch.save(best_params, 'models/best_hyperparams.pth')

    # ===================== Final Results =====================
    print("\n" + "=" * 70)
    print(f"🏆 BEST MODEL FOUND:")
    print(f"   Accuracy : {best_acc:.2f}%")
    print(f"   Params   : {best_params}")
    print("=" * 70)

    return best_model, best_params, results


full_dataset = EuroSatDataset("Balanced data")

best_model, best_params, results = hyperparameter_search(
    full_dataset=full_dataset,
    saved_model_path='models/AlexNet.pth',
    num_trials=20
)