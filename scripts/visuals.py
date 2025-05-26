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
    plt.savefig("training.png")

def plot_energy_curve(energies, n=1,l=0,path = CONVERGE_PATH):
    plt.figure(figsize=(10, 6))

    plt.plot(energies[5000:])
    plt.xlabel("Epoch")
    plt.ylabel("Energy")
    plt.title("Energy Convergence")
    plt.grid()
    plt.savefig(os.path.join(path,f"converge_n_{n}_l_{l}.png"))


def plot_all_wavefunctions(r_tensor, plot_path=PLOT_PATH, wave_path=WF_PATH,csv_file=CSV_FILE):

    # Define analytical hydrogenic radial wavefunctions (unnormalized)
    def psi_10(x): return x * 2 * np.exp(-x)
    def psi_21(x): return x * (1 / (np.sqrt(6) * 2)) * x * np.exp(-x / 2)
    def psi_20(x): return x * (1 / np.sqrt(2)) * (1 - x / 2) * np.exp(-x / 2)
    def psi_32(x): return x * (4 / (9 * np.sqrt(30))) * (x / 3)**2 * np.exp(-x / 3)
    def psi_30(x): return x * (2 / (3 * np.sqrt(3))) * (1 - (2 * x / 3) + (2 * x**2 / 27)) * np.exp(-x / 3)

    analytical_map = {
        (1, 0): psi_10,
        (2, 1): psi_21,
        (2, 0): psi_20,
        (3, 2): psi_32,
        (3, 0): psi_30,
    }

    df = pd.read_csv(csv_file)

    xss = np.linspace(0.01, 30, 500)

    for idx, row in df.iterrows():
        n = int(row['n'])
        l = int(row['l'])
        energy = float(row['energy'])

        # Create directory if it doesn't exist
        dir_path = os.path.join(plot_path, f"n{n}")
        os.makedirs(dir_path, exist_ok=True)

        # Load and align wavefunction
        wf_path = os.path.join(wave_path, row['wf_file'])
        psi = np.load(wf_path)
        if len(psi) != len(xss):
            psi = np.interp(xss, np.linspace(0.01, 30, len(psi)), psi)

        plt.figure(figsize=(6, 6))
        label = f"NN $\psi_{{{n}{l}}}$, E={energy:.5f}"
        plt.plot(xss, psi**2, label=label, color='blue')


        key = (n, l)
        if key in analytical_map:
            analytical = analytical_map[key](xss)
            plt.plot(xss, analytical**2, '--', label=fr"Exact $\psi_{{{n}{l}}}^2$", color='orange')

        plt.title(f"Wavefunction for n={n}, l={l}")
        plt.xlabel("r")
        plt.ylabel(r"$\psi_{nl}^2(r)$")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        save_name = os.path.join(dir_path, f"wavefunction_n{n}_l{l}.png")
        plt.savefig(save_name)
        plt.close()