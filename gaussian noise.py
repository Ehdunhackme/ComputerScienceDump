import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import ImageTk, Image
import numpy as np
import cv2

def generate_gaussian_noise(shape, mean=0, stddev=1):
    noise = np.random.normal(mean, stddev, shape)
    return noise

def add_noise(image, noise):
    noisy_image = np.clip((image + noise), 0, 255)
    return noisy_image.astype(np.uint8)

def generate_noisy_image(image, noise_scale, grayscale=False, pixelate=False):
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    image = image.astype(np.float32) / 255.0
    noise = generate_gaussian_noise(image.shape, stddev=noise_scale)
    noisy_image = add_noise(image, noise)

    if pixelate:
        noisy_image = cv2.resize(noisy_image, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_NEAREST)
        noisy_image = cv2.resize(noisy_image, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    return noisy_image

def preprocess_image(image):
    return (image.astype('float32') - image.min()) / (image.max() - image.min())

def open_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = img.resize((300, 300), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        original_label.config(image=img)
        original_label.image = img
        apply_button.config(state=tk.NORMAL)
        global selected_image
        selected_image = cv2.imread(file_path)

def apply_noise():
    noise_scale = noise_scale_slider.get()
    grayscale = grayscale_var.get()
    pixelate = pixelate_var.get()

    noisy_img = generate_noisy_image(selected_image, noise_scale, grayscale, pixelate)
    noisy_img = preprocess_image(noisy_img)
    noisy_img = Image.fromarray((noisy_img * 255).astype(np.uint8))
    noisy_img = noisy_img.resize((300, 300), Image.ANTIALIAS)
    noisy_img = ImageTk.PhotoImage(noisy_img)
    noisy_label.config(image=noisy_img)
    noisy_label.image = noisy_img

# Create main window
root = tk.Tk()
root.title("Image Noising")

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

# Create grayscale checkbox
grayscale_var = tk.BooleanVar()
grayscale_checkbox = tk.Checkbutton(root, text="Grayscale", variable=grayscale_var)
grayscale_checkbox.pack()

# Create pixelate checkbox
pixelate_var = tk.BooleanVar()
pixelate_checkbox = tk.Checkbutton(root, text="Pixelate", variable=pixelate_var)
pixelate_checkbox.pack()

# Initialize selected image
selected_image = None

# Run the main event loop
root.mainloop()