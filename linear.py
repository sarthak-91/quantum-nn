import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from scripts.config import *
import csv
from scripts.diff import fourth_order_second_derivative
from scripts.visuals import plot_all_wavefunctions

def load_wavefunctions_for_ortho(n, l, wave_path=WF_PATH, csv_path=CSV_FILE,print_=False):
    df = pd.read_csv(csv_path)
    
    # specific (n, l) pairs to load for orthogonalization
    states_to_load_map = {
        (4, 0): [(3,0),(2, 0),(1, 0)],
        (3, 2): [],
        (3, 1): [(2, 1)],
        (3, 0): [(2, 0),(1,0)],
        (2, 1): [],
        (2, 0): [(1, 0)],
        (1, 0): [] 
    }

    target_states = states_to_load_map.get((n, l), [])

    if not target_states:
        if print_:print(f"No specific prior wavefunctions defined for (n={n}, l={l})")
        return []

    wavefunctions = []
    found_count = 0

    for n_prev, l_prev in target_states:
        filtered_row = df[(df['n'] == n_prev) & (df['l'] == l_prev)]

        if not filtered_row.empty:
            row = filtered_row.iloc[0]
            psi_path = os.path.join(wave_path, row['wf_file'])
            
            if os.path.exists(psi_path):
                psi = np.load(psi_path)
                psi_tensor = torch.tensor(psi, dtype=torch.float32)
                E0 = row['energy']

                wavefunctions.append((psi_tensor, E0))
                found_count += 1
                if print_:print(f"Loaded orthogonalization state: n={row['n']}, l={row['l']}, energy={E0:.6f}")
            else:
                if print_:print(f"Warning: Wavefunction file not found for n={n_prev}, l={l_prev} at {psi_path}")
        else:
            if print_:print(f"Warning: No entry found in registry for n={n_prev}, l={l_prev}")

    if print_ and found_count == 0 and target_states:
        print(f"None of the required prior wavefunctions for (n={n}, l={l}) were found.")

    return wavefunctions


# === Neural Network Definition ===
class HydrogenNet(nn.Module):
    def __init__(self, hidden_layers=[32, 32]):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_layers[0]),
            nn.Tanh(),
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            nn.Tanh(),
            nn.Linear(hidden_layers[1], 1)
        )

    def forward(self, r):
        return self.net(r)

# === Utility: Second derivative using finite differences ===
def second_derivative(f, r):
    dr = r[1] - r[0]
    f_dd = torch.zeros_like(f)
    f_dd[1:-1] = (f[:-2] - 2 * f[1:-1] + f[2:]) / (dr ** 2)
    f_dd[0] = f_dd[1]
    f_dd[-1] = f_dd[-2]
    return f_dd

# === Construct physical wavefunction from NN output ===
def build_wavefunction(model_output, r, energy, l):
    decay = torch.sqrt(torch.clamp(-2 * energy.detach(), min=1e-8))
    return model_output * r**(l + 1) * torch.exp(-decay * r)

# === Orthogonality loss ===
def ortho_loss_fn(psi, wf_list, r):
    dr = r[1] - r[0]
    loss = torch.tensor(0.0, device=psi.device)
    for psi0, _ in wf_list:
        psi0 = psi0.to(psi.device)
        overlap = torch.sum(psi.squeeze() * psi0.squeeze() * dr)
        loss += overlap**2
    return loss

# === Loss Function ===
def energy_and_loss(model, r, energy_prev, l=0, excited=False, wf_list=[]):
    model_output = model(r)
    psi = build_wavefunction(model_output, r, energy_prev, l)
    dr = r[1] - r[0]
    psi_dd = fourth_order_second_derivative(psi, r)

    # Hamiltonian
    H_psi = -0.5 * psi_dd - psi / r + l * (l + 1) * psi / (2 * r ** 2)
    psi_H_psi = psi * H_psi

    norm = torch.sum(psi ** 2 * dr)
    energy = torch.sum(psi_H_psi * dr) / norm
    residual = H_psi - energy * psi
    loss = torch.mean(residual**2) + (1 - norm)**2

    # Orthogonality term for excited states
    if excited and wf_list:
        loss += ortho_loss_fn(psi, wf_list, r)

    return energy, loss, psi

# === Training Loop ===
def train(model, r, n, l=0, epochs=2000, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    energy_estimate = torch.tensor(-0.5, requires_grad=False)

    # Load orthogonality wavefunctions if needed
    wf_list = load_wavefunctions_for_ortho(n, l)
    excited = len(wf_list) > 0
    energy_history = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        energy, loss, _ = energy_and_loss(model, r, energy_estimate, l, excited, wf_list)
        loss.backward()
        optimizer.step()
        energy_estimate = 0.9 * energy_estimate + 0.1 * energy.detach()
        energy_history.append(energy.item())
        if loss<1e-6:break
        if epoch % 5000 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:5d} | Energy: {energy.item():.6f} | Loss: {loss.item():.6e}")

    return model, energy_estimate.item(), energy_history

# === Main ===
def store(model, r, energy, n, l, wave_path=WF_PATH, model_path=MODEL_PATH, csv_file=CSV_FILE):
    os.makedirs(wave_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    param_filename = f"state_n_{n}_l_{l}.pt"
    wf_filename = f"wf_n_{n}_l_{l}.npy"

    torch.save(model.state_dict(), os.path.join(model_path, param_filename))

    r_tensor = r.clone().detach().unsqueeze(1)
    with torch.no_grad():
        model_output = model(r_tensor)
        psi = build_wavefunction(model_output, r_tensor, energy, l).squeeze().numpy()
    np.save(os.path.join(wave_path, wf_filename), psi)

    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["n", "l", "energy", "param_file", "wf_file"])
        writer.writerow([n, l, energy.item(), param_filename, wf_filename])

# === Main ===
if __name__ == '__main__':
    torch.manual_seed(0)
    N = 1000
    r_vals = torch.linspace(1e-3, 30.0, N).view(-1, 1)

    n, l = 3, 0 # Example: excited state
    model = HydrogenNet()
    model, final_energy, energy_list = train(model, r_vals, n=n, l=l, epochs=100000, lr=5e-4)

    # Evaluate final wavefunction
    with torch.no_grad():
        model_output = model(r_vals)
        psi_final = build_wavefunction(model_output, r_vals, torch.tensor(final_energy), l)
        norm = torch.sum(psi_final**2 * (r_vals[1] - r_vals[0]))
        psi_normalized = psi_final / torch.sqrt(norm)

    # Save model and wavefunction

    # Save model and wavefunction
    store(model, r_vals.squeeze(), torch.tensor(final_energy), n=n, l=l)

    # Plot all wavefunctions using visuals.py
    plot_all_wavefunctions()
