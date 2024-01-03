import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageOps
try:
    from PIL import ImageTk
except ImportError:
    from PIL import Image as ImageTk
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Specify the backend (TkAgg is commonly used for Tkinter-based GUIs)


# Load the MNIST dataset
mnist = fetch_openml('mnist_784')

# Preprocess data for PCA
X = mnist.data.astype('float64')
y = mnist.target.astype('int')

# Use PCA to reduce dimensions
pca = PCA(n_components=50)
X_pca = pca.fit_transform(X)

# Train a classifier
clf = LogisticRegression(max_iter=10000)
clf.fit(X_pca, y)

# Function to classify the image
def classify_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path).convert("L")  # Open image and convert to grayscale
        img = ImageOps.invert(img)  # Invert colors (background to black, number to white)
        img = img.resize((28, 28))  # Resize to MNIST image size
        img_array = np.array(img)
        img_array = img_array / 255.0  # Normalize pixel values

        # Display the processed image
        processed_img = Image.fromarray((img_array * 255).astype(np.uint8))
        processed_img.show()

        # Reshape for prediction
        img_array = img_array.reshape(1, -1)

        img_pca = pca.transform(img_array)  # PCA transform

         # Function to classify the image
def classify_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img = img.convert('L')  # Convert to grayscale
        img = img.resize((28, 28))  # Resize to MNIST image size
        img = np.array(img)
        img = img.flatten()  # Flatten to match MNIST data format
        img = img.reshape(1, -1)  # Reshape for prediction
        
        # Display the processed image
        plt.imshow(img.reshape(28, 28), cmap='gray')
        plt.show()

        img_pca = pca.transform(img)  # PCA transform
        prediction = clf.predict(img_pca)
        if prediction[0] == 5:
            result_label.config(text="Recognized: Number 5")
        else:
            result_label.config(text="Recognized: Not Number 5")

# ... (remaining code)

        prediction = clf.predict(img_pca)
        if prediction[0] == 5:
            result_label.config(text="Recognized: Number 5")
        else:
            result_label.config(text="Not Recognized: Not Number 5")

# Create GUI
root = tk.Tk()
root.title("Number 5 Recognizer")

canvas = tk.Canvas(root, width=300, height=300)
canvas.pack()

browse_button = tk.Button(root, text="Browse Image", command=classify_image)
browse_button.pack()

result_label = tk.Label(root, text="")
result_label.pack()

root.mainloop()
