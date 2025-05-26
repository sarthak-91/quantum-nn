import torch 
import os 
import numpy as np 
import pandas as pd
import csv
from scripts.config import *

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

def find_nearest_state(n, model_path=MODEL_PATH,csv_path=CSV_FILE):
    df = pd.read_csv(csv_path)


    df_filtered = df[df['n'] < n]
    if not df_filtered.empty:
        best_row = df_filtered.sort_values('n', ascending=False).iloc[0]
        return os.path.join(model_path, best_row['param_file'])

    df_higher = df[df['n'] > n]
    if not df_higher.empty:
        best_row = df_higher.sort_values('n').iloc[0]
        return os.path.join(model_path, best_row['param_file'])

    return None

def store(model, r,energy, n, l, wave_path=WF_PATH,model_path=MODEL_PATH, csv_file= CSV_FILE):
    os.makedirs(wave_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    param_filename = f"state_n_{n}_l_{l}.pt"
    wf_filename = f"wf_n_{n}_l_{l}.npy"

    torch.save(model.state_dict(), os.path.join(model_path, param_filename))

    r_tensor = r.clone().detach().unsqueeze(1)
    with torch.no_grad():
        psi = model(r_tensor,l).squeeze().numpy()
    np.save(os.path.join(wave_path, wf_filename), psi)


    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["n", "l", "energy", "param_file", "wf_file"])
        writer.writerow([n, l, energy.item(), param_filename, wf_filename])
