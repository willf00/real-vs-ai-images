import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
import os
from torchvision import transforms
import model
"""
gui.py: Provides an interactive GUI for a user to input an image to get classified
Authors: Will Fete & Jason Albanus
Date: 12/7/2025
Notice: To use this, you must have a model saved to ./cnn_model.pth
"""

# --- CONFIGURATION ---
CNN_MODEL_PATH = './cnn_model.pth'
CLASS_NAMES = ['Fake', 'Real']

# Load Model globally
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading model on {device}...")

# Initialize architecture
cnn = model.SimpleCNN().to(device) # Use simple cnn model for now

# Load weights
# map_location ensures it works even if you switch between GPU/CPU machines
cnn.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=device))

# Set to eval mode to avoid training and dropout
cnn.eval()

# Define the Transform (Same transform done to dataset)
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def select_file():
    """
       Args:
       None

       Purpose: Executes when the button has been pressed. Opens up the file explorer for user to choose an input image.
                Calls the selected model in inference mode to get a prediction and confidence rating.
    """
    filetypes = (('jpg files', '*.jpg'),) # Only accepts jpg
    filename = filedialog.askopenfilename(title='Open a file', initialdir=os.getcwd(), filetypes=filetypes)

    if filename:
        img_pil = Image.open(filename).convert('RGB')

        display_img = img_pil.resize((300, 300), Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(display_img)
        img_label.config(image=tk_img)
        img_label.image = tk_img

        # Apply transforms -> becomes tensor (3, 224, 224)
        input_tensor = inference_transform(img_pil)

        # Add batch dimension -> becomes (1, 3, 224, 224)
        input_batch = input_tensor.unsqueeze(0).to(device)

        with torch.no_grad():  # Disable gradient calculation for speed
            output = cnn(input_batch)

            # Get the predicted class index
            _, predicted_idx = torch.max(output, 1)
            prediction = CLASS_NAMES[predicted_idx.item()]

            # Get confidence
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            confidence = probabilities[predicted_idx].item() * 100

        # Update GUI
        result_label.config(text=f"Prediction: {prediction} ({confidence:.1f}%)", fg="blue", font=("Arial", 16, "bold"))


# --- GUI SETUP ---
root = tk.Tk()
root.title('Real vs AI Detector')
root.geometry('400x500')

open_button = tk.Button(root, text='Upload Image', command=select_file, font=("Arial", 12))
open_button.pack(pady=20)

img_label = tk.Label(root)
img_label.pack()

# Label to show the text result
result_label = tk.Label(root, text="Upload an image to test", font=("Arial", 14))
result_label.pack(pady=20)

root.mainloop()