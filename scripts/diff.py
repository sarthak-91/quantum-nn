import torch 
import numpy as np 

def chebyshev_points(N, a=0.0, b=1.0):
    """Generate Chebyshev-Gauss-Lobatto points in [a, b]"""
    k = torch.arange(0, N + 1)
    x = torch.cos(np.pi * k / N)
    # Map from [-1,1] to [a,b]
    return 0.5 * (b - a) * (x + 1) + a

def chebyshev_diff_matrix(N):
    """Chebyshev differentiation matrix D and points x"""
    x = torch.cos(np.pi * torch.arange(0, N + 1) / N)
    c = torch.ones(N + 1)
    c[0] = 2
    c[-1] = 2
    c = c * (-1) ** torch.arange(0, N + 1)

    X = x.repeat(N + 1, 1)
    dX = X - X.T

    D = (c.unsqueeze(1) / c.unsqueeze(0)) / (dX + torch.eye(N + 1))  # Add I to avoid div by 0
    D = D - torch.diag(D.sum(1))

    return D, x

def chebyshev_second_derivative_matrix(N, a=0.0, b=1.0):
    """Returns second derivative matrix D2 and Chebyshev points x mapped to [a,b]"""
    D, x_cheb = chebyshev_diff_matrix(N)
    x_mapped = 0.5 * (b - a) * (x_cheb + 1) + a  # rescale
    D = D * (2.0 / (b - a))  # rescale D for domain [a,b]
    D2 = torch.matmul(D, D)
    return D2, x_mapped

def gradient(y: torch.Tensor, x: torch.Tensor, order:int =1)->torch.Tensor:
    derivative = y
    for i in range(order):
        derivative =torch.autograd.grad(
                derivative,x,
                torch.ones_like(x),
                create_graph=True,retain_graph=True)[0]
    return derivative

def finite_second_derivative(psi, r):
    dr = r[1] - r[0]
    d2psi = torch.zeros_like(psi)
    d2psi[1:-1] = (psi[:-2] - 2*psi[1:-1] + psi[2:]) / (dr ** 2)
    d2psi[0] = d2psi[1]  
    d2psi[-1] = d2psi[-2]
    return d2psi

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