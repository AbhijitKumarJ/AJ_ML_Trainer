import tkinter as tk
from tkinter import ttk, messagebox
import json
import os

class ModelConfigUI:
    def __init__(self, parent, current_model_type, load_config_func):
        self.frame = ttk.Frame(parent)
        self.current_model_type = current_model_type
        self.load_config = load_config_func
        self.config_widgets = {}
        self.create_widgets()

    def create_widgets(self):
        self.config_frame = ttk.LabelFrame(self.frame, text="Model Configuration")
        self.config_frame.pack(padx=10, pady=10, fill="both", expand=True)

        self.save_button = ttk.Button(self.frame, text="Save Configuration", command=self.save_configuration)
        self.save_button.pack(pady=10)

    def update_ui(self, config):
        for widget in self.config_frame.winfo_children():
            widget.destroy()
        
        self.config_widgets = {}
        for key, value in config.items():
            ttk.Label(self.config_frame, text=f"{key}:").grid(row=len(self.config_widgets), column=0, padx=5, pady=5, sticky="w")
            if isinstance(value, bool):
                var = tk.BooleanVar(value=value)
                widget = ttk.Checkbutton(self.config_frame, variable=var)
            elif isinstance(value, (int, float)):
                var = tk.StringVar(value=str(value))
                widget = ttk.Entry(self.config_frame, textvariable=var)
            elif isinstance(value, list):
                var = tk.StringVar(value=", ".join(map(str, value)))
                widget = ttk.Entry(self.config_frame, textvariable=var)
            else:
                var = tk.StringVar(value=str(value))
                widget = ttk.Entry(self.config_frame, textvariable=var)
            widget.grid(row=len(self.config_widgets), column=1, padx=5, pady=5)
            self.config_widgets[key] = (var, widget)

    def save_configuration(self):
        config = {}
        for key, (var, widget) in self.config_widgets.items():
            if isinstance(var, tk.BooleanVar):
                config[key] = var.get()
            elif isinstance(widget, ttk.Entry):
                try:
                    value = json.loads(var.get())
                    config[key] = value
                except json.JSONDecodeError:
                    config[key] = var.get()
        
        model_type = self.current_model_type.get()
        config_path = os.path.join("AJML/ModelCodes", model_type, f"Config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        messagebox.showinfo("Configuration Saved", f"Model configuration for {model_type} has been saved.")