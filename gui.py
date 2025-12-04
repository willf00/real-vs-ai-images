import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os

def select_file():
    filetypes = (
        ('jpg files', '*.jpg'),
    )

    filename = filedialog.askopenfilename(
        title='Open a file',
        initialdir=os.getcwd(),
        filetypes=filetypes
    )

    if filename:
        img_path = filename
        img = Image.open(img_path)

        # Resize the image to fit the window
        img = img.resize((300, 300), Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(img)
        img_label.config(image=tk_img)
        img_label.image = tk_img

        #Call model here

# Create the main window
root = tk.Tk()
root.title('Tkinter File Upload')
root.geometry('400x400')

open_button = tk.Button(
    root,
    text='Upload File',
    command=select_file
)
open_button.pack(pady=20)

# Create the label
img_label = tk.Label(root)
img_label.pack()

root.mainloop()