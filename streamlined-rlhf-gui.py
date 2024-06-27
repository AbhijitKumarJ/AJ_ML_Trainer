import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

class RLHFApplication(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("RLHF Application")
        self.geometry("800x600")

        self.model = None
        self.X = None
        self.y = None

        self.create_widgets()

    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Parameters frame
        params_frame = ttk.LabelFrame(main_frame, text="Training Parameters")
        params_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        ttk.Label(params_frame, text="Hidden Size:").grid(row=0, column=0, padx=5, pady=5)
        self.hidden_size_var = tk.IntVar(value=20)
        ttk.Entry(params_frame, textvariable=self.hidden_size_var).grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(params_frame, text="Learning Rate:").grid(row=1, column=0, padx=5, pady=5)
        self.lr_var = tk.DoubleVar(value=0.01)
        ttk.Entry(params_frame, textvariable=self.lr_var).grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(params_frame, text="Epochs:").grid(row=2, column=0, padx=5, pady=5)
        self.epochs_var = tk.IntVar(value=1000)
        ttk.Entry(params_frame, textvariable=self.epochs_var).grid(row=2, column=1, padx=5, pady=5)

        # Buttons frame
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        ttk.Button(buttons_frame, text="Generate Data", command=self.generate_data).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(buttons_frame, text="Train Model", command=self.train_model).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(buttons_frame, text="Save Model", command=self.save_model).grid(row=0, column=2, padx=5, pady=5)
        ttk.Button(buttons_frame, text="Load Model", command=self.load_model).grid(row=0, column=3, padx=5, pady=5)

        # Inference frame
        inference_frame = ttk.LabelFrame(main_frame, text="Inference")
        inference_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")

        ttk.Label(inference_frame, text="Input:").grid(row=0, column=0, padx=5, pady=5)
        self.inference_input_var = tk.DoubleVar()
        ttk.Entry(inference_frame, textvariable=self.inference_input_var).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(inference_frame, text="Infer", command=self.infer).grid(row=0, column=2, padx=5, pady=5)

        self.inference_output_var = tk.StringVar()
        ttk.Label(inference_frame, textvariable=self.inference_output_var).grid(row=1, column=0, columnspan=3, padx=5, pady=5)

        # Log frame
        log_frame = ttk.LabelFrame(main_frame, text="Interaction Log")
        log_frame.grid(row=3, column=0, padx=10, pady=10, sticky="nsew")

        self.log_text = tk.Text(log_frame, height=10, width=70)
        self.log_text.pack(padx=5, pady=5)

        ttk.Button(log_frame, text="Save Log", command=self.save_log).pack(pady=5)

        # Plot frame
        self.plot_frame = ttk.Frame(main_frame)
        self.plot_frame.grid(row=0, column=1, rowspan=4, padx=10, pady=10, sticky="nsew")

        # Configure grid weights
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(3, weight=1)

    def generate_data(self):
        self.X = torch.linspace(-5, 5, 200).reshape(-1, 1)
        self.y = self.X**2 # + torch.randn(200, 1) * 2

        # Normalize the data
        self.X = (self.X - self.X.mean()) / self.X.std()
        self.y = (self.y - self.y.mean()) / self.y.std()

        self.plot_data()
        self.log_action("Data generated")

    def plot_data(self):
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(self.X, self.y)
        ax.set_xlabel('X')
        ax.set_ylabel('y')
        ax.set_title('Generated Data')

        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def train_model(self):
        if self.X is None or self.y is None:
            messagebox.showerror("Error", "Please generate data first.")
            return

        hidden_size = self.hidden_size_var.get()
        lr = self.lr_var.get()
        epochs = self.epochs_var.get()

        self.model = NeuralNetwork(input_size=1, hidden_size=hidden_size, output_size=1)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            outputs = self.model(self.X)
            loss = criterion(outputs, self.y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 100 == 0:
                self.log_action(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

        self.plot_results()
        self.log_action("Model training completed")

    def plot_results(self):
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(self.X, self.y, label='Data')
        
        with torch.no_grad():
            predictions = self.model(self.X)
        ax.plot(self.X, predictions, color='r', label='Model')
        
        ax.set_xlabel('X')
        ax.set_ylabel('y')
        ax.set_title('Model Predictions')
        ax.legend()

        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def save_model(self):
        if self.model is None:
            messagebox.showerror("Error", "No model to save. Please train a model first.")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".pth")
        if file_path:
            torch.save(self.model.state_dict(), file_path)
            self.log_action(f"Model saved to {file_path}")
            messagebox.showinfo("Success", "Model saved successfully.")

    def load_model(self):
        file_path = filedialog.askopenfilename(filetypes=[("PyTorch Model", "*.pth")])
        if file_path:
            hidden_size = self.hidden_size_var.get()
            self.model = NeuralNetwork(input_size=1, hidden_size=hidden_size, output_size=1)
            self.model.load_state_dict(torch.load(file_path))
            self.model.eval()
            self.log_action(f"Model loaded from {file_path}")
            messagebox.showinfo("Success", "Model loaded successfully.")

    def infer(self):
        if self.model is None:
            messagebox.showerror("Error", "No model loaded. Please train or load a model first.")
            return

        x = self.inference_input_var.get()
        x_tensor = torch.tensor([[x]]).float()
        print(x_tensor)
        with torch.no_grad():
            prediction = self.model(x_tensor)
        
        result = f"Input: {x:.2f}, Prediction: {prediction.item():.2f}"
        self.inference_output_var.set(result)
        self.log_action(f"Inference: {result}")

    def log_action(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)

    def save_log(self):
        log_content = self.log_text.get("1.0", tk.END)
        file_path = filedialog.asksaveasfilename(defaultextension=".txt")
        if file_path:
            with open(file_path, "w") as f:
                f.write(log_content)
            messagebox.showinfo("Success", "Log saved successfully.")

if __name__ == "__main__":
    app = RLHFApplication()
    app.mainloop()
