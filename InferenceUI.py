import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import importlib
import os

class InferenceUI:
    def __init__(self, parent):
        self.frame = ttk.Frame(parent)
        self.current_model_type = None
        self.inference_module = None
        self.create_widgets()

    def create_widgets(self):
        self.model_frame = ttk.LabelFrame(self.frame, text="Model")
        self.model_frame.pack(padx=10, pady=10, fill="both", expand=True)

        self.model_path = tk.StringVar()
        ttk.Label(self.model_frame, text="Model File:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(self.model_frame, textvariable=self.model_path).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(self.model_frame, text="Browse", command=self.browse_model).grid(row=0, column=2, padx=5, pady=5)

        self.input_frame = ttk.LabelFrame(self.frame, text="Input")
        self.input_frame.pack(padx=10, pady=10, fill="both", expand=True)

        self.input_text = tk.Text(self.input_frame, height=5)
        self.input_text.pack(padx=5, pady=5, fill="both", expand=True)

        self.infer_button = ttk.Button(self.frame, text="Run Inference", command=self.run_inference)
        self.infer_button.pack(pady=10)

        self.output_frame = ttk.LabelFrame(self.frame, text="Output")
        self.output_frame.pack(padx=10, pady=10, fill="both", expand=True)

        self.output_text = tk.Text(self.output_frame, height=5, state="disabled")
        self.output_text.pack(padx=5, pady=5, fill="both", expand=True)

    def update_ui(self, model_type):
        self.current_model_type = model_type
        try:
            self.inference_module = importlib.import_module(f"ModelCodes.{model_type}.Inference", package="AJ_ML_Trainer")
        except ImportError:
            messagebox.showerror("Error", f"Inference module for {model_type} not found.")
            self.inference_module = None

    def browse_model(self):
        file_path = filedialog.askopenfilename(filetypes=[("PyTorch Model", "*.pth")])
        if file_path:
            self.model_path.set(file_path)

    def run_inference(self):
        if not self.inference_module:
            messagebox.showerror("Error", "No inference module loaded.")
            return

        model_path = self.model_path.get()
        if not model_path:
            messagebox.showerror("Error", "Please select a model file.")
            return

        input_text = self.input_text.get("1.0", tk.END).strip()
        if not input_text:
            messagebox.showerror("Error", "Please enter input text for inference.")
            return

        config_path = os.path.join("ModelCodes", self.current_model_type, f"{self.current_model_type}Config.txt")

        try:
            result = self.inference_module.run_inference(model_path, config_path, input_text)
            self.output_text.config(state="normal")
            self.output_text.delete("1.0", tk.END)
            self.output_text.insert(tk.END, result)
            self.output_text.config(state="disabled")
        except Exception as e:
            messagebox.showerror("Inference Error", f"An error occurred during inference: {str(e)}")