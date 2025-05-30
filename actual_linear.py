import torch
import torch.nn as nn
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import os, csv

# === Define Wavefunction Model ===
class RadialNN(nn.Module):
    def __init__(self, hidden_layers=[20, 20]):
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

# === Build physical wavefunction ψ(r) = NN(r) * r^{l+1} * exp(-sqrt(-2E) * r) ===
def build_wavefunction(model_output, r, energy, l):
    decay = torch.sqrt(torch.clamp(-2 * energy.detach(), min=1e-8))
    return model_output * r**(l + 1) * torch.exp(-decay * r)

# === Second derivative using central difference ===
def fourth_order_second_derivative(psi, r):
    dr = r[1] - r[0]
    d2psi = torch.zeros_like(psi)
    d2psi[2:-2] = (-psi[:-4] + 16*psi[1:-3] - 30*psi[2:-2] + 16*psi[3:-1] - psi[4:]) / (12 * dr ** 2)
    # fallback to 2nd-order near boundaries
    d2psi[1] = (psi[0] - 2*psi[1] + psi[2]) / dr**2
    d2psi[-2] = (psi[-3] - 2*psi[-2] + psi[-1]) / dr**2
    d2psi[0] = d2psi[1]
    d2psi[-1] = d2psi[-2]
    return d2psi
# === Hamiltonian action Hψ ===
def hamiltonian(psi, r, l):
    psi_dd = fourth_order_second_derivative(psi, r)
    return -0.5 * psi_dd - psi / r + l * (l + 1) * psi / (2 * r**2)

# === Compute overlap and Hamiltonian matrices ===
def assemble_matrices(psi, H_psi, jac, dr):
    S = jac.T @ (jac * dr)
    H = jac.T @ (H_psi.unsqueeze(1) * dr)
    g = jac.T @ (psi.detach().unsqueeze(1) * dr)
    return S.detach().numpy(), H.detach().numpy(), g.detach().numpy()


# === Jacobian of NN output wrt parameters ===
def compute_jacobian(model, r):
    params = list(model.parameters())
    flat_params = torch.cat([p.view(-1) for p in params])
    flat_params = flat_params.detach().requires_grad_(True)

    def set_model_params(p):
        idx = 0
        for param in model.parameters():
            num = param.numel()
            param.data.copy_(p[idx:idx + num].view_as(param))
            idx += num

    set_model_params(flat_params)
    outputs = model(r).squeeze()  # [N]
    N = outputs.shape[0]
    P = flat_params.shape[0]
    J = torch.zeros(N, P)

    for i in range(N):
        grad = torch.autograd.grad(outputs[i], model.parameters(), retain_graph=True, create_graph=False)
        grad_flat = torch.cat([g.contiguous().view(-1) for g in grad])
        J[i] = grad_flat

    return J, flat_params

# === Linear Method Solver ===
def solve_linear_method(S, H, g, damping=1e-2):
    S += damping * np.eye(S.shape[0])
    H_eff = H - g @ g.T
    eigvals, eigvecs = scipy.linalg.eigh(H_eff, S)
    delta_theta = eigvecs[:, np.argmin(eigvals)]
    return delta_theta


# === Parameter update ===
def apply_update(model, delta, alpha=0.01, max_norm=1.0):
    delta = torch.tensor(delta, dtype=torch.float32)
    norm = torch.norm(delta)
    if norm > max_norm:
        delta = delta * (max_norm / norm)

    idx = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            update = delta[idx:idx+n].view_as(p)
            p.data += alpha * update
            idx += n


# === Training loop ===
def train_linear_method(steps=30, N=300, l=0):
    r_vals = np.linspace(0.01, 30.0, N)
    r = torch.tensor(r_vals, dtype=torch.float32).view(-1, 1)
    dr = r[1] - r[0]
    model = RadialNN()
    energy_est = torch.tensor(-0.5)
    energies = []

    for step in range(steps):
        model_output = model(r)
        psi = build_wavefunction(model_output, r, energy_est, l).squeeze()
        H_psi = hamiltonian(psi, r.squeeze(), l)

        energy = torch.sum(psi * H_psi * dr) / torch.sum(psi**2 * dr)
        energies.append(energy.item())
        energy_est = 0.9 * energy_est + 0.1 * energy.detach()

        J, _ = compute_jacobian(model, r)
        S, H, g = assemble_matrices(psi, H_psi, J, dr)
        delta = solve_linear_method(S, H, g)
        apply_update(model, delta)

        print(f"Step {step:02d} | Energy: {energy.item():.6f}")

    return model, r_vals, energies, energy_est

# === Main ===
if __name__ == "__main__":
    model, r_vals, energies, final_energy = train_linear_method(steps=1000, N=500, l=0)

    with torch.no_grad():
        r_torch = torch.tensor(r_vals, dtype=torch.float32).unsqueeze(1)
        psi_final = build_wavefunction(model(r_torch), r_torch, final_energy, l=0).squeeze()
        norm = torch.sum(psi_final**2 * (r_torch[1] - r_torch[0]))
        psi_normalized = psi_final / torch.sqrt(norm)

    # Plot
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(energies)
    plt.xlabel("Iteration")
    plt.ylabel("Energy")
    plt.title("Energy Convergence")

    plt.subplot(1, 2, 2)
    plt.plot(r_vals, psi_normalized.numpy())
    plt.xlabel("r")
    plt.ylabel("ψ(r)")
    plt.title("Learned Wavefunction")
    plt.grid()
    plt.tight_layout()
    plt.show()
