import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import gradio as gr
from train_mnist import CNN  # Make sure your CNN class is correctly imported

# Load trained model
model = CNN()
model.load_state_dict(torch.load("mnist-model.pth", map_location=torch.device('cpu')))
model.eval()

# Prediction function
def predict_digit(image):
    if isinstance(image, dict):
        if "layers" in image:
            image_np = np.array(image["layers"][-1])  # Use last drawing layer
        else:
            raise ValueError("Unexpected input format from Sketchpad.")
    else:
        image_np = image  # Already a numpy array

    # Convert to PIL, grayscale, resize to 28x28
    image = Image.fromarray(image_np).convert("L").resize((28, 28))
    image = np.array(image) / 255.0  # Normalize pixel values
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, 28, 28]

    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)[0]

    return {str(i): float(probabilities[i]) for i in range(10)}

# Gradio interface
interface = gr.Interface(
    fn=predict_digit,
    inputs=gr.Sketchpad(label="✍️ Draw a digit (0–9)", type="numpy"),
    outputs=gr.Label(num_top_classes=3),
    title="🧠 Handwritten Digit Recognizer",
    description="Draw a digit and the CNN model will try to predict it."
)

interface.launch()
