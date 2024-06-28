import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import importlib
import os
import AJML.ModelCodes

class TrainingUI:
    def __init__(self, parent):
        self.frame = ttk.Frame(parent)
        self.current_model_type = None
        self.training_module = None
        self.create_widgets()

    def create_widgets(self):
        self.data_frame = ttk.LabelFrame(self.frame, text="Training Data")
        self.data_frame.pack(padx=10, pady=10, fill="both", expand=True)

        self.data_path = tk.StringVar()
        ttk.Label(self.data_frame, text="Data File:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(self.data_frame, textvariable=self.data_path).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(self.data_frame, text="Browse", command=self.browse_data).grid(row=0, column=2, padx=5, pady=5)

        self.train_button = ttk.Button(self.frame, text="Train Model", command=self.train_model)
        self.train_button.pack(pady=10)

    def update_ui(self, model_type):
        print(model_type)
        mod_path=f"..ModelCodes.{model_type}.Training"
        print(mod_path)

        self.current_model_type = model_type
        try:
            self.training_module = importlib.import_module(f"AJML.ModelCodes.{model_type}.Training")
        except ImportError:
            messagebox.showerror("Error", f"Training module for {model_type} not found.")
            self.training_module = None

    def browse_data(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.data_path.set(file_path)

    def train_model(self):
        if not self.training_module:
            messagebox.showerror("Error", "No training module loaded.")
            return

        data_path = self.data_path.get()
        if not data_path:
            messagebox.showerror("Error", "Please select a data file.")
            return

        config_path = os.path.join("AJML/ModelCodes", self.current_model_type, f"Config.json")
        
        try:
            self.training_module.train_model(config_path, data_path)
            messagebox.showinfo("Training Complete", "Model training has been completed.")
        except Exception as e:
            messagebox.showerror("Training Error", f"An error occurred during training: {str(e)}")