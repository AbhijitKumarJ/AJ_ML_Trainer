import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import os
from TrainingUI import TrainingUI
from InferenceUI import InferenceUI
from ModelConfigUI import ModelConfigUI

class NNManagerUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Neural Network Manager")
        self.master.geometry("900x700")

        self.notebook = ttk.Notebook(self.master)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.model_types = self.get_model_types()
        self.current_model_type = tk.StringVar()
        self.current_model_type.set(self.model_types[0] if self.model_types else "")
        self.current_config = {}

        self.create_model_selection_frame()
        self.model_config = ModelConfigUI(self.notebook, self.current_model_type, self.load_config)
        self.training_ui = TrainingUI(self.notebook)
        self.inference_ui = InferenceUI(self.notebook)

        self.notebook.add(self.model_config.frame, text="Model Configuration")
        self.notebook.add(self.training_ui.frame, text="Training")
        self.notebook.add(self.inference_ui.frame, text="Inference")

        self.create_menu()

    def get_model_types(self):
        return [d for d in os.listdir("AJML/ModelCodes") if os.path.isdir(os.path.join("AJML/ModelCodes", d)) and d not in ["__pycache__","Sample"]]

    def create_model_selection_frame(self):
        frame = ttk.Frame(self.master)
        frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(frame, text="Select Model Type:").pack(side=tk.LEFT)
        model_type_combo = ttk.Combobox(frame, textvariable=self.current_model_type, values=self.model_types)
        model_type_combo.pack(side=tk.LEFT, padx=5)
        model_type_combo.bind("<<ComboboxSelected>>", self.on_model_type_changed)

    def on_model_type_changed(self, event):
        self.load_config()
        self.model_config.update_ui(self.current_config)
        self.training_ui.update_ui(self.current_model_type.get())
        self.inference_ui.update_ui(self.current_model_type.get())

    def load_config(self):
        model_type = self.current_model_type.get()
        config_path = os.path.join("AJML/ModelCodes", model_type, f"Config.json")
        try:
            with open(config_path, 'r') as f:
                self.current_config = json.load(f)
        except FileNotFoundError:
            messagebox.showerror("Error", f"Configuration file for {model_type} not found.")
            self.current_config = {}

    def create_menu(self):
        menubar = tk.Menu(self.master)
        self.master.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Switch to Model Configuration", command=lambda: self.notebook.select(0))
        file_menu.add_command(label="Switch to Training", command=lambda: self.notebook.select(1))
        file_menu.add_command(label="Switch to Inference", command=lambda: self.notebook.select(2))
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.master.quit)

if __name__ == "__main__":
    root = tk.Tk()
    app = NNManagerUI(root)
    root.mainloop()