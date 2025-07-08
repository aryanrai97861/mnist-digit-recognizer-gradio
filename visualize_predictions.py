import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from train_mnist import CNN  # Uses your saved model class

# Load test data
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor()),
    batch_size=10, shuffle=True)

# Load model
model = CNN()
model.load_state_dict(torch.load("mnist-model.pth"))
model.eval()

# Get 1 batch of 10 images
images, labels = next(iter(test_loader))

# Make predictions
with torch.no_grad():
    outputs = model(images)
    _, preds = torch.max(outputs, 1)

# Plot images with predictions
plt.figure(figsize=(12, 6))
for i in range(10):
    image = images[i].squeeze()  # 1x28x28 → 28x28
    plt.subplot(2, 5, i+1)
    plt.imshow(image, cmap='gray')
    color = 'green' if preds[i] == labels[i] else 'red'
    plt.title(f"Pred: {preds[i].item()}\nTrue: {labels[i].item()}", color=color)
    plt.axis('off')

plt.tight_layout()
plt.show()
