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
        self.E = nn.Parameter(data=-torch.tensor(-1.0))
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
        soft = nn.Softplus()
        self.exponential_coeff = soft(self.E)
        output_tensor = self.Hidden_Layers[-1](y_out)
        self.output = output_tensor
        return output_tensor * torch.exp(-self.exponential_coeff*input_tensor) * input_tensor**(l+1)
    
    def l2_reg_coeffs(self):
        l2_loss = 0.0
        for param in self.parameters():
            l2_loss += torch.sum(param**2)
        return l2_loss
    

class GaussianWavefunction(nn.Module):
    def __init__(self, num_basis):
        super().__init__()
        self.num_basis = num_basis
        alpha_init = torch.log(torch.linspace(0.1, 2.0, num_basis))
        self.log_alpha = nn.Parameter(alpha_init)
        self.coeffs = nn.Parameter(torch.randn(num_basis)) 

    def forward(self, r,l=0):
        alpha = torch.exp(self.log_alpha).unsqueeze(1)  
        gaussians = torch.exp(-alpha * r**2)            
        psi = torch.sum(self.coeffs.unsqueeze(1) * gaussians, dim=0) * r ** (l+1) 
        return psi