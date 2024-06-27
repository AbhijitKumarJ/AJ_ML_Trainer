import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import ModelCodes
import Utils


def save_model(rlhfapp:tk.Tk):
        if rlhfapp.model is None:
            messagebox.showerror("Error", "No model to save. Please train a model first.")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".pth")
        if file_path:
            torch.save(rlhfapp.model.state_dict(), file_path)
            rlhfapp.log_action(f"Model saved to {file_path}")
            messagebox.showinfo("Success", "Model saved successfully.")

def load_model(rlhfapp:tk.Tk):
    file_path = filedialog.askopenfilename(filetypes=[("PyTorch Model", "*.pth")])
    if file_path:
        hidden_size = rlhfapp.hidden_size_var.get()
        rlhfapp.model = ModelCodes.XSquare.XSquareModel.XSquareModel(input_size=1, hidden_size=hidden_size, output_size=1)
        rlhfapp.model.load_state_dict(torch.load(file_path))
        rlhfapp.model.eval()
        rlhfapp.log_action(f"Model loaded from {file_path}")
        messagebox.showinfo("Success", "Model loaded successfully.")

def train_model(rlhfapp:tk.Tk):
    if rlhfapp.X is None or rlhfapp.y is None:
        messagebox.showerror("Error", "Please generate data first.")
        return

    hidden_size = rlhfapp.hidden_size_var.get()
    lr = rlhfapp.lr_var.get()
    epochs = rlhfapp.epochs_var.get()

    rlhfapp.model = ModelCodes.XSquare.XSquareModel.XSquareModel(input_size=1, hidden_size=hidden_size, output_size=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(rlhfapp.model.parameters(), lr=lr)

    for epoch in range(epochs):
        outputs = rlhfapp.model(rlhfapp.X)
        loss = criterion(outputs, rlhfapp.y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            rlhfapp.log_action(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    Utils.plot_results(rlhfapp)
    rlhfapp.log_action("Model training completed")
