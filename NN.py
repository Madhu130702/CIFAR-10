# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

#Transforms
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

transform_eval = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load & Split Dataset
full_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
train_size = int(0.9 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
val_dataset.dataset.transform = transform_eval

test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_eval)

# DataLoaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

#  Intermediate Block with BatchNorm + ReLU
class IntermediateBlock(nn.Module):
    def __init__(self, in_channels, conv_layer_configs):
        super().__init__()
        self.L = len(conv_layer_configs)
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, config['out_channels'], config['kernel_size'], padding=config.get('padding', 0)),
                nn.BatchNorm2d(config['out_channels']),
                nn.ReLU()
            )
            for config in conv_layer_configs
        ])
        self.fc = nn.Linear(in_channels, self.L)

    def forward(self, x):
        conv_outputs = torch.stack([conv(x) for conv in self.convs], dim=0)
        m = x.mean(dim=[2, 3])
        a = self.fc(m)
        a = F.softmax(a, dim=1)
        a = a.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        a = a.permute(1, 0, 2, 3, 4)
        out = (a * conv_outputs).sum(dim=0)
        return out

#  Output Block with Dropout
class OutputBlock(nn.Module):
    def __init__(self, in_channels, hidden_dims=[256], num_classes=10):
        super().__init__()
        layers = []
        input_dim = in_channels
        for h in hidden_dims:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.3))
            input_dim = h
        layers.append(nn.Linear(input_dim, num_classes))
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        m = x.mean(dim=[2, 3])
        return self.fc(m)

# Full Model with 3 Blocks + 4 Conv Layers Each
class CIFARClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(
            IntermediateBlock(3, [
                {'out_channels': 64, 'kernel_size': 1, 'padding': 0},
                {'out_channels': 64, 'kernel_size': 3, 'padding': 1},
                {'out_channels': 64, 'kernel_size': 5, 'padding': 2},
                {'out_channels': 64, 'kernel_size': 7, 'padding': 3},
            ]),
            IntermediateBlock(64, [
                {'out_channels': 128, 'kernel_size': 1, 'padding': 0},
                {'out_channels': 128, 'kernel_size': 3, 'padding': 1},
                {'out_channels': 128, 'kernel_size': 5, 'padding': 2},
                {'out_channels': 128, 'kernel_size': 7, 'padding': 3},
            ]),
            IntermediateBlock(128, [
                {'out_channels': 256, 'kernel_size': 1, 'padding': 0},
                {'out_channels': 256, 'kernel_size': 3, 'padding': 1},
                {'out_channels': 256, 'kernel_size': 5, 'padding': 2},
                {'out_channels': 256, 'kernel_size': 7, 'padding': 3},
            ])
        )
        self.output = OutputBlock(in_channels=256, hidden_dims=[256])

    def forward(self, x):
        x = self.blocks(x)
        return self.output(x)

# Training Setup
model = CIFARClassifier().to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)


# Train & Eval Functions
batch_losses = []
train_accuracies = []
val_accuracies = []
best_acc = 0.0
best_model_state = None

def train(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), correct / total

def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    return total_loss / len(loader), correct / total

# Train Loop
epochs = 50
for epoch in range(epochs):
    train_loss, train_acc = train(model, train_loader, optimizer, criterion)
    val_loss, val_acc = evaluate(model, val_loader, criterion)
    scheduler.step()

    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

    if val_acc > best_acc:
        best_acc = val_acc
        best_model_state = model.state_dict()

    print(f"Epoch {epoch+1}/{epochs}")
    print(f"  Train Loss: {train_loss:.4f}, Accuracy: {train_acc*100:.2f}%")
    print(f"  Val   Loss: {val_loss:.4f}, Accuracy: {val_acc*100:.2f}%\n")

# Save best model
torch.save(best_model_state, "best_model.pth")
print(f"✅ Best Validation Accuracy: {best_acc*100:.2f}%")

# Plotting
plt.plot(train_accuracies, label="Train Accuracy")
plt.plot(val_accuracies, label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy per Epoch")
plt.legend()
plt.show()

plt.plot(batch_losses)
plt.xlabel("Batch")
plt.ylabel("Training Loss")
plt.title("Loss per Batch")
plt.show()

# Final Test Evaluation
model.load_state_dict(torch.load("best_model.pth"))
test_loss, test_acc = evaluate(model, test_loader, criterion)
print(f" Final Test Accuracy: {test_acc*100:.2f}%")
