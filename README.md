# 🧠 MNIST Digit Recognizer gradio

This project uses a Convolutional Neural Network (CNN) to classify handwritten digits from the [MNIST dataset](http://yann.lecun.com/exdb/mnist/). It includes a Gradio-powered web interface that allows users to draw a digit and see the model's prediction in real-time.

---

## 📦 Features

- 📊 Trains a CNN model on the MNIST dataset  
- ✅ Includes validation with early stopping  
- 🧪 Evaluates final test accuracy  
- 🖼️ Provides a Gradio UI to draw and recognize digits  
- 💾 Saves the best performing model as `mnist-model.pth`

---

## 🧰 Tech Stack

- Python  
- PyTorch  
- Gradio  
- NumPy  
- PIL

---

## 🚀 Getting Started

### 1. Clone the repository

bash
git clone https://github.com/your-username/mnist-digit-recognizer.git
cd mnist-digit-recognizer


### 2. Create a virtual environment

bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


### 3. Install dependencies

bash
pip install -r requirements.txt


### 4. Train the model (optional)

bash
python train_mnist.py


### 5. Test the model (optional)

bash
python test_mnist.py


### 6. Launch Gradio app

bash
python gradio_app.py


Then open your browser and go to: [http://127.0.0.1:7860](http://127.0.0.1:7860)

---

## 📂 Project Structure

bash
.
├── gradio_app.py          # Gradio interface to draw digits
├── train_mnist.py         # Training script with early stopping
├── test_mnist.py          # Evaluate test accuracy
├── mnist-model.pth        # Saved model
├── requirements.txt       # Required Python packages
├── visualize_predictions.py # (Optional) For visualizing predictions
├── .gitignore             # Files/folders to ignore in version control
└── README.md              # You're reading it!


---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first.

---


