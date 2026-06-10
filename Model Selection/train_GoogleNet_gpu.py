from dataset import *
from PCA import *
from models import *
from train import *
import joblib
import os
from sklearn.model_selection import train_test_split

# ========================= CONFIG =========================
ROOT_DIR = "Balanced data"
BATCH_SIZE = 64
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
NUM_COMPONENTS = 8

# Load dataset
full_dataset = EuroSatDataset(ROOT_DIR)
# =======================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {device} - {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# ====================== PCA ======================
pca_path = 'models/pca_8components.pkl'

if os.path.exists(pca_path):
    print(f"📂 Loading saved PCA from {pca_path}")
    pca = joblib.load(pca_path)
else:
    print("🔄 Saved PCA not found. Fitting new PCA...")
    pca = fit_pca(full_dataset, n_components=NUM_COMPONENTS, sample_size=40000)
    os.makedirs('models', exist_ok=True)
    joblib.dump(pca, pca_path)
    print(f"💾 PCA saved to {pca_path}")

full_dataset.pca = pca
# =================================================
# Train/Val/Test Split
print("Splitting the dataset...")
train_idx, temp_idx = train_test_split(range(len(full_dataset)), test_size=0.40, stratify=full_dataset.labels, random_state=42)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, stratify=[full_dataset.labels[i] for i in temp_idx], random_state=42)

train_ds = Subset(full_dataset, train_idx)
val_ds   = Subset(full_dataset, val_idx)
test_ds  = Subset(full_dataset, test_idx)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=8, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE*2, shuffle=False, num_workers=8, pin_memory=True)

# Model
print("Creating model...")
model = create_model(num_classes=10, in_channels=NUM_COMPONENTS, model_name='googlenet', pretrained=True)

print("Training the model...")
model = train_model(model, train_loader, val_loader, num_epochs=NUM_EPOCHS, lr=LEARNING_RATE, device=device)

# Final Test
print("Testing the model...")
evaluate_model(model, test_loader, full_dataset.classes, device=device)

# Save final model
torch.save(model.state_dict(), 'models/GoogleNet.pth')
print("✅ Training completed and model saved!")