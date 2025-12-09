#%% packages
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.models import resnet50
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

#%% data augmentation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    # transforms.RandomHorizontalFlip(p=0.5),
    # transforms.RandomRotation(degrees=20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#%% dataset and dataloader
data = "data/PetImages/"
full_dataset = datasets.ImageFolder(data, transform=transform)

train_size = int(0.7 * len(full_dataset))
val_size   = int(0.15 * len(full_dataset))
test_size  = len(full_dataset) - train_size - val_size

train_ds, val_ds, test_ds = random_split(full_dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=15)
val_loader   = DataLoader(val_ds, batch_size=32, num_workers=15)
test_loader  = DataLoader(test_ds, batch_size=32, num_workers=15)

#%% model
class PetsClassification(pl.LightningModule):
    def __init__(self, num_classes=2, freeze_backbone=True):
        super().__init__()
        self.model = resnet50(weights="IMAGENET1K_V2") 
        in_features = self.model.fc.in_features 
        self.model.fc = nn.Linear(in_features, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.fc.parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        acc = (preds.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        acc = (preds.argmax(dim=1) == y).float().mean()
        self.log("test_loss", loss)
        self.log("test_acc", acc)

    def configure_optimizers(self):
        learning_rate = 1e-4
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        return optimizer
    
#%% training
model = PetsClassification(num_classes=2)

early_stop_callback = EarlyStopping(
    monitor="val_loss", 
    patience=10, 
    mode="min"
)

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    mode="min",
    filename="best-pets-model",
    verbose=True
)

trainer = pl.Trainer(accelerator='auto', 
                     devices=1, 
                     max_epochs=200, 
                     log_every_n_steps=2, 
                     callbacks=[early_stop_callback, checkpoint_callback],
                     enable_checkpointing=True)

trainer.fit(model, train_loader, val_loader)

#%% plot loss curves
metrics_file = "lightning_logs/version_0/metrics.csv"
df = pd.read_csv(metrics_file)

train_loss = df[df["train_loss"].notna()]
val_loss   = df[df["val_loss"].notna()]

plt.figure(figsize=(8,5))
plt.plot(train_loss["step"], train_loss["train_loss"], label="Train Loss")
plt.plot(val_loss["step"], val_loss["val_loss"], label="Val Loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_curve.png")

#%% test
def test_model(model, test_loader, ckpt_path="lightning_logs/version_0/checkpoints/best-pets-model.ckpt"):
    model = model.load_from_checkpoint(ckpt_path, num_classes=2)
    model.eval()
    device = "gpu" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            preds = model(x)
            preds = preds.argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(y.tolist())

    cm = confusion_matrix(all_labels, all_preds)
    labels = test_loader.dataset.dataset.classes 
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=labels))

test_model(PetsClassification, test_loader)
