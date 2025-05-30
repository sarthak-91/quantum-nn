import matplotlib.pyplot as plt
import numpy as np 
import os 
import pandas as pd
import torch 
from scripts.config import * 
def plot_loss_curve(losses, save_path = "training.png"):
    plt.figure(figsize=(10, 6))

    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid()
    plt.savefig(save_path)

def plot_energy_curve(energies, n=1,l=0,path = CONVERGE_PATH):
    plt.figure(figsize=(10, 6))

    plt.plot(energies[500:])
    plt.xlabel("Epoch")
    plt.ylabel("Energy")
    plt.title(f"Energy Convergence, n={n}, l={l}")
    plt.grid()
    plt.savefig(os.path.join(path,f"converge_n_{n}_l_{l}.png"))


def plot_all_wavefunctions(r, model_class,hidden_layers=[10,10], plot_path=PLOT_PATH, model_path=MODEL_PATH, csv_file=CSV_FILE):
    
    def psi_10(x): return x * 2 * np.exp(-x)
    def psi_21(x): return x * (1 / (np.sqrt(6) * 2)) * x * np.exp(-x / 2)
    def psi_20(x): return x * (1 / np.sqrt(2)) * (1 - x / 2) * np.exp(-x / 2)
    def psi_32(x): return x * (4 / (9 * np.sqrt(30))) * (x / 3)**2 * np.exp(-x / 3)
    def psi_30(x): return x * (2 / (3 * np.sqrt(3))) * (1 - (2 * x / 3) + (2 * x**2 / 27)) * np.exp(-x / 3)
    def psi_31(x): return x * (8 / (27 * np.sqrt(6))) * (1 - x / 6) * x * np.exp(-x / 3)
    def psi_40(x): return x * (1 / 4) * (1 - 12 * x / 16 + 32 * x**2 / 256 - 64 / 3 * x**3 / (16**3)) * np.exp(-x / 4)

    analytical_map = {
        (1, 0): psi_10,
        (2, 1): psi_21,
        (2, 0): psi_20,
        (3, 2): psi_32,
        (3, 0): psi_30,
        (3, 1): psi_31,
        (4, 0): psi_40
    }
    if r.ndim == 1:
        r = r.unsqueeze(-1)
    r_np = r.squeeze().cpu().numpy()

    df = pd.read_csv(csv_file)

    for _, row in df.iterrows():
        n = int(row['n'])
        l = int(row['l'])
        energy = float(row['energy'])
        param_file = row['param_file']
        model_file = os.path.join(model_path, param_file)

        if not os.path.exists(model_file):
            print(f"Model file not found: {model_file}")
            continue


        model = model_class(hidden_layers)
        state_dict = torch.load(model_file, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()

        with torch.no_grad():
            psi = model(r,l).squeeze().cpu().numpy()

        plt.figure(figsize=(6, 6))
        plt.plot(r_np, psi**2, label=f"NN $\\psi_{{{n}{l}}}^2$, E={energy:.5f}", color='blue')

        if (n, l) in analytical_map:
            psi_analytical = analytical_map[(n, l)](r_np)
            plt.plot(r_np, psi_analytical**2, '--', label=fr"Exact $\psi_{{{n}{l}}}^2$", color='orange')

        plt.title(f"Wavefunction: n={n}, l={l}")
        plt.xlabel("r")
        plt.ylabel(r"$\psi_{nl}^2(r)$")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        dir_path = os.path.join(plot_path, f"n{n}")
        os.makedirs(dir_path, exist_ok=True)

        save_path = os.path.join(dir_path, f"wavefunction_n{n}_l{l}.png")
        plt.savefig(save_path)
        plt.close()