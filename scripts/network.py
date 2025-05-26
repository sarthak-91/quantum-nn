import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinAct(torch.nn.Module):
    """custom sine activation function"""
    @staticmethod
    def forward(input):
        return torch.sin(input)

class NN(torch.nn.Module):

    def __init__(self,hidden_layers:list,output_num:int = 1):
        super(NN,self).__init__()
        self.activation = nn.Tanh()
        self.E = nn.Parameter(data=-torch.abs(torch.rand(())))
        input_layer = torch.nn.Linear(1,hidden_layers[0])
        self.Hidden_Layers = nn.ModuleList()
        self.Hidden_Layers.append(input_layer)
        for i in range(0,len(hidden_layers)-1):
            layer = torch.nn.Linear(hidden_layers[i],hidden_layers[i+1])
            self.Hidden_Layers.append(layer)
        output_layer = nn.Linear(hidden_layers[-1],output_num)
        self.Hidden_Layers.append(output_layer)

    def forward(self, input_tensor, l = 0):
        y_out = input_tensor
        for i in range(0,len(self.Hidden_Layers)-1):
            z_out = self.Hidden_Layers[i](y_out)
            y_out = self.activation(z_out)

        exponential_coeff = torch.sqrt(torch.abs(2*self.E))
        output_tensor = self.Hidden_Layers[-1](y_out)
        self.output = output_tensor
        return output_tensor * torch.exp(-exponential_coeff*input_tensor) * input_tensor ** (l+1)
    
    def l2_reg_coeffs(self):
        l2_loss = 0.0
        for param in self.parameters():
            l2_loss += torch.sum(param**2)
        return l2_loss
    
class RBFModel(nn.Module):
    def __init__(self, n_basis=200, r_min=0.01, r_max=30.0, init_epsilon=1.0, n_grid=500):
        super().__init__()
        self.n_basis = n_basis

        # Fixed Gaussian centers and grid for normalization
        self.register_buffer('centers', torch.linspace(r_min, r_max, n_basis).view(1, -1))
        self.register_buffer('r_grid', torch.linspace(r_min, r_max, n_grid).view(-1, 1))
        self.dr = (r_max - r_min) / (n_grid - 1)
        self.E = nn.Parameter(data=-torch.abs(torch.rand(())))
        # Learnable parameters
        self.coefficients = nn.Parameter(torch.randn(n_basis))
        self.log_eps = nn.Parameter(torch.tensor(init_epsilon).log())

    def forward(self, r, l=0):
        r = r.view(-1, 1)  # [batch, 1]
        epsilon = 2
        exponential_coeff = torch.sqrt(2*torch.abs(self.E))
        # Gaussian basis: ψ_raw(r)
        dist_sq = (r - self.centers) ** 2
        basis = torch.exp(-epsilon * dist_sq)
        psi_raw = basis @ self.coefficients  # [batch, 1]
        psi_raw = psi_raw.squeeze()  # [batch]

        # Final form: ψ(r) = exp(-r) * ψ_raw(r) * r^{l+1}
        r_flat = r.view(-1)
        psi_full = psi_raw * r_flat**(l + 1) * torch.exp(-exponential_coeff*r_flat)  # [batch]

        psi_normalized = psi_full 
        return psi_normalized.view(-1, 1)
    def l2_reg_coeffs(self):
        """Returns the L2 regularization term for the coefficients."""
        return torch.sum(self.coefficients**2)


