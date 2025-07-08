import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split

# 1. Load and split MNIST dataset
full_train_data = torchvision.datasets.MNIST(
    './data', train=True, download=True, transform=transforms.ToTensor())

train_data, val_data = random_split(full_train_data, [54000, 6000])

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=1000, shuffle=False)

# 2. Define neural network
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # [B, 16, 28, 28]
        x = self.pool(x)           # [B, 16, 14, 14]
        x = F.relu(self.conv2(x))  # [B, 32, 14, 14]
        x = self.pool(x)           # [B, 32, 7, 7]
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


model = CNN()

if __name__ == "__main__":
    # 3. Loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # 4. Early stopping setup
    best_val_acc = 0.0
    epochs_no_improve = 0
    early_stop_patience = 2
    best_model_state = None

    # 5. Training loop with validation and early stopping
    for epoch in range(20):
        model.train()
        for images, labels in train_loader:
            output = model(images)
            loss = loss_fn(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                output = model(images)
                _, predicted = torch.max(output, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_accuracy = 100 * val_correct / val_total
        print(f"✅ Epoch {epoch+1} | Validation Accuracy: {val_accuracy:.2f}%")

        # Early stopping check
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(f"⏹️ Early stopping triggered after {epoch+1} epochs.")
                break

    # 6. Save best model
    if best_model_state:
        torch.save(best_model_state, "mnist-model.pth")
        print(f"✅ Best model saved with Validation Accuracy: {best_val_acc:.2f}%")
