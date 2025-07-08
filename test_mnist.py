import torch
import torchvision
import torchvision.transforms as transforms
from train_mnist import CNN  # Import the same model class

# Load the MNIST test dataset
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor()),
    batch_size=1000, shuffle=False)

# Initialize the model and load trained weights
model = CNN()
model.load_state_dict(torch.load("mnist-model.pth"))
model.eval()

# Evaluate accuracy on the test set
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"✅ Final Test Accuracy: {accuracy:.2f}%")
