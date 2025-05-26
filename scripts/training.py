import torch 
import os 
import numpy as np 
import pandas as pd
from scripts.visuals import *
from scripts.load_store import find_nearest_state
from scripts.loss import norm_loss_fn

def train(model, r_torch, loss_fn, epochs = 300000, excited_state=False, wf_list=[],n=None, l=0,lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    losses = []
    energies = []
    last_E_value = None  

    for epoch in range(epochs):
        total_loss = torch.zeros(1)
        optimizer.zero_grad()
        energy, pde_loss, norm_loss, ortho_loss, pos_loss = loss_fn(model, r_torch,excited_state,wf_list=wf_list, n=n, l=l)
        e_diff = (energy - model.E)**2
        total_loss += energy
        total_loss += 100*pde_loss 
        total_loss += 100*norm_loss
        total_loss += 100*ortho_loss 
        total_loss += 100*e_diff
        #total_loss += pos_loss 
        #total_loss += 0.1*model.l2_reg_coeffs()
        losses.append(total_loss.item())
        total_loss.backward()
        current_E_value = energy

        if last_E_value is not None:  
            if (torch.abs(current_E_value - last_E_value) < 1e-6 and 
                ortho_loss.item() < 1e-7 and 
                pde_loss.item()<5e-7 and 
                norm_loss.item()<1e-7 and 
                e_diff.item()<1e-7):
                print("last E = ", last_E_value.item(), "energy = ", current_E_value.item(), "Convergence reached")
                print(f"pde: {pde_loss.item():.5f} |",end=' ')
                print(f"norm: {norm_loss.item():.5f} |",end=' ')
                break
        
        last_E_value = current_E_value
        optimizer.step()
        lr = 0
        energies.append(energy.item())
        if (epoch) % 5000 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:4d} |",end='')
            print(f"Energy: {current_E_value.item():.7f} |",end=' ')
            print(f"pde: {pde_loss.item():.7f} |",end=' ')
            print(f"norm: {norm_loss.item():.7f} |",end=' ')
            #print(f"replusive: {replusive_loss.item():.6f}",end=' ')
            print(f"difference in E: {e_diff.item():.7f} |",end=' ')
            if excited_state:
                print(f"ortho = {ortho_loss.item():.7f} |",end=' ')
            print("\n")

    plot_loss_curve(losses, save_path = "training.png")
    plot_energy_curve(energies,n=n, l=l, path="convergence")
    return energy

