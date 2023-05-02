import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
from PIL import ImageGrab
import tkinter.messagebox as tkMessageBox


# Load the pre-trained CNN model
model = keras.models.load_model('mnist_cnn_model1.h5')
# model = load_model('mnist_cnn_model.h5')

# Create a canvas to draw on
canvas_width = 280
canvas_height = 280
canvas_color = 'white'

class DrawingCanvas(tk.Canvas):
    def __init__(self, master, **kwargs):
        super().__init__(master, width=canvas_width, height=canvas_height, bg=canvas_color, **kwargs)
        self.old_coords = None
        self.bind('<B1-Motion>', self.draw)
        
    def draw(self, event):
        if self.old_coords:
            x1, y1 = self.old_coords
            x2, y2 = event.x, event.y
            self.create_line(x1, y1, x2, y2, width=15, capstyle='round', smooth=True)
        self.old_coords = event.x, event.y
        
    def clear(self):
        self.delete('all')
        self.old_coords = None

# Create a GUI window with a drawing canvas and a button to recognize the digit
class DrawingApp(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.pack()
        self.create_widgets()
        
    def create_widgets(self):
        self.canvas = DrawingCanvas(self)
        self.canvas.pack(side='top', pady=5)
        
        self.recognize_button = tk.Button(self, text='Recognize', command=self.recognize_digit)
        self.recognize_button.pack(side='top')
        
        self.clear_button = tk.Button(self, text='Clear', command=self.clear_canvas)
        self.clear_button.pack(side='top', pady=5)
        
        self.quit_button = tk.Button(self, text='Quit', command=self.quit)
        self.quit_button.pack(side='bottom', pady=10)
        
    def recognize_digit(self):
        # Capture the image from the canvas
        x0 = self.canvas.winfo_rootx() + self.canvas.winfo_x()
        y0 = self.canvas.winfo_rooty() + self.canvas.winfo_y()
        x1 = x0 + self.canvas.winfo_width()
        y1 = y0 + self.canvas.winfo_height()
        image = ImageGrab.grab((x0, y0, x1, y1)).convert('L')
        image = image.resize((28, 28))
        
        # Preprocess the image
        image_array = np.array(image)
        image_array = image_array.reshape(1, 28, 28, 1)
        image_array = image_array.astype('float32')
        image_array /= 255
        
        # Pass the image through the model and get the prediction
        prediction = model.predict(image_array)
        digit = np.argmax(prediction)
        
        # Display the prediction in a message box
        # tk.messagebox.showinfo('Prediction', f'The recognized digit is: {digit}')
        tkMessageBox.showinfo('Prediction', f'The recognized digit is: {digit}')

    def clear_canvas(self):
        self.canvas.clear()

# Start the GUI app
root = tk.Tk()
app = DrawingApp(root)
app.mainloop()
