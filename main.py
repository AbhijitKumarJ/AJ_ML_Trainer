
import tkinter as tk
import sys
# import os

# sys.path.append(os.path.join(os.path.dirname(__file__), 'AJML'))

import NNManagerUI

if __name__ == "__main__":
    root = tk.Tk()
    app = NNManagerUI.NNManagerUI(root)
    root.mainloop()