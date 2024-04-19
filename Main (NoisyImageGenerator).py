import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import ImageTk, Image
import numpy as np
import cv2

def generate_gaussian_noise(shape, mean=0, stddev=1):
    noise = np.random.normal(mean, stddev, shape)
    return noise

def generate_gaussian_noise(shape, mean=0, stddev=1):
    noise = np.random.normal(mean, stddev, shape)
    return noise
def generate_noisy_image(img_path, noise_scale, intensity, grayscale=False, pixelate=False):
    # Read the image
    img = cv2.imread(img_path)
    
    # Convert image to grayscale if specified
    if grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # Convert back to BGR for consistent shape

    img = img[..., ::-1] / 255.0  # Convert to float and normalize to [0, 1]

    # Generate Gaussian noise with specified scale
    noise = generate_gaussian_noise(img.shape, stddev=noise_scale)

    # Increase intensity of noise
    noisy_noise = noise * intensity

    # Add noise to the image
    noisy_img = np.clip((img + noisy_noise), 0, 1)

    # Apply pixelation if specified
    if pixelate:
        noisy_img = cv2.resize(noisy_img, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_NEAREST)
        noisy_img = cv2.resize(noisy_img, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    return noisy_img


# Function to preprocess image
def preprocess_image(img):
    # Convert image to numpy array and normalize
    img_array = np.array(img)
    img_array = (img_array.astype('float32') - img_array.min()) / (img_array.max() - img_array.min())
    return img_array

# Function to open image file
def open_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img = img.resize((300, 300), Image.ANTIALIAS)  # Resize image for display
        img = ImageTk.PhotoImage(img)
        original_label.config(image=img)
        original_label.image = img
        apply_button.config(state=tk.NORMAL)
        global selected_image_path
        selected_image_path = file_path

# Function to apply noise
def apply_noise():
    noise_scale = noise_scale_slider.get()
    intensity = intensity_slider.get()
    grayscale = grayscale_var.get()
    pixelate = pixelate_var.get()
    
    noisy_img = generate_noisy_image(selected_image_path, noise_scale, intensity, grayscale, pixelate)
    
    if noisy_img is None:
        return
    
    noisy_img = preprocess_image(noisy_img)
    noisy_img = Image.fromarray((noisy_img * 255).astype(np.uint8))
    noisy_img = noisy_img.resize((300, 300), Image.ANTIALIAS)  # Resize image for display
    noisy_img = ImageTk.PhotoImage(noisy_img)
    noisy_label.config(image=noisy_img)
    noisy_label.image = noisy_img

# Create main window
root = tk.Tk()
root.title("NoisyImageGenerator")

# Create original image label
original_label = tk.Label(root, text="Original Image")
original_label.pack()

# Create noisy image label
noisy_label = tk.Label(root, text="Noisy Image")
noisy_label.pack()

# Create buttons
open_button = tk.Button(root, text="Open Image", command=open_file)
open_button.pack()

apply_button = tk.Button(root, text="Apply Noise", command=apply_noise, state=tk.DISABLED)
apply_button.pack()

# Create noise scale slider
noise_scale_label = tk.Label(root, text="Noise Scale:")
noise_scale_label.pack()

noise_scale_slider = tk.Scale(root, from_=0.01, to=0.1, resolution=0.01, orient=tk.HORIZONTAL)
noise_scale_slider.pack()

# Create intensity slider
intensity_label = tk.Label(root, text="Intensity:")
intensity_label.pack()

intensity_slider = tk.Scale(root, from_=0.1, to=2.0, resolution=0.1, orient=tk.HORIZONTAL)
intensity_slider.pack()

# Create checkboxes for grayscale and pixelation
grayscale_var = tk.BooleanVar()
grayscale_check = tk.Checkbutton(root, text="Grayscale", variable=grayscale_var)
grayscale_check.pack()

pixelate_var = tk.BooleanVar()
pixelate_check = tk.Checkbutton(root, text="Pixelate", variable=pixelate_var)
pixelate_check.pack()

# Initialize selected image path
selected_image_path = None

root.mainloop()