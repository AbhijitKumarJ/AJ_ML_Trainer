import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os

def plot_data(rlhfapp:tk.Tk):
        for widget in rlhfapp.plot_frame.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(rlhfapp.X, rlhfapp.y)
        ax.set_xlabel('X')
        ax.set_ylabel('y')
        ax.set_title('Generated Data')

        canvas = FigureCanvasTkAgg(fig, master=rlhfapp.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)





def plot_results(rlhfapp:tk.Tk):
        for widget in rlhfapp.plot_frame.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(rlhfapp.X, rlhfapp.y, label='Data')
        
        with torch.no_grad():
            predictions = rlhfapp.model(rlhfapp.X)
        ax.plot(rlhfapp.X, predictions, color='r', label='Model')
        
        ax.set_xlabel('X')
        ax.set_ylabel('y')
        ax.set_title('Model Predictions')
        ax.legend()

        canvas = FigureCanvasTkAgg(fig, master=rlhfapp.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)