import torch 
import os 
import numpy as np 
import pandas as pd
from scripts.visuals import *
from scripts.load_store import log_errors
from scripts.loss import norm_loss_fn

def update_lr(optimizer, pde_loss, base_lr=1e-3, min_lr=1e-6, max_loss=5e-5, min_loss=1e-7):
    clamped_loss = max(min(pde_loss.item(), max_loss), min_loss)
    alpha = (clamped_loss - min_loss) / (max_loss - min_loss)
    new_lr = min_lr + alpha * (base_lr - min_lr)
    for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
    
    return new_lr


def train(model, r_torch, loss_fn, epochs = 300000, excited_state=False, wf_list=[],n=None, l=0,lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    energies = []
    pde_errors = []
    ortho_errors = []
    norm_errors = []
    last_E_value = None  
    energy_weight = 10 if n ==1 else 1
    pde_weight = 100 
    norm_weight = 100 
    ortho_weight = 100 

    for epoch in range(epochs):
        total_loss = torch.zeros(1)
        optimizer.zero_grad()
        energy, pde_loss, norm_loss, ortho_loss, pos_loss = loss_fn(model, r_torch,excited_state,wf_list=wf_list, n=n, l=l)
        
        total_loss += energy_weight*energy
        total_loss += pde_weight*pde_loss 
        total_loss += norm_weight*norm_loss
        total_loss += ortho_weight*ortho_loss 
        total_loss += 100*pos_loss


        losses.append(total_loss.item())
        total_loss.backward()
        current_E_value = energy
        if n!=1 and norm_loss<1e-5 and ortho_loss<1e-5: current_lr = update_lr(optimizer,pde_loss)
        else:current_lr = lr

        convergence_criteria = 1e-7 if n!=1 else 5e-7

        if last_E_value is not None:  
            if (ortho_loss.item() < 1e-7 and 
                pde_loss.item()<convergence_criteria and 
                norm_loss.item()<1e-7):
                print(f"energy = {current_E_value.item():.8f} Convergence Reached with",end=' ')
                print(f"pde: {pde_loss.item():.8f} |",end=' ')
                print(f"norm: {norm_loss.item():.8f} |",end=' ')
                print(f"orthogonal loss: {ortho_loss.item():.8f}")
                break
        
        last_E_value = current_E_value
        optimizer.step()
        energies.append(energy.item())
        pde_errors.append(pde_loss.item())
        norm_errors.append(norm_loss.item())
        ortho_errors.append(ortho_loss.item())
        if (epoch) % 5000 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:4d} |",end='')
            print(f"Energy: {current_E_value.item():.7f} |",end=' ')
            print(f"pde: {pde_loss.item():.8f} |",end=' ')
            print(f"norm: {norm_loss.item():.7f} |",end=' ')
            print(f"using lr:{current_lr:.7f}|",end=' ')
            if excited_state:
                print(f"ortho = {ortho_loss.item():.7f} |",end=' ')
            print("\n")

    log_errors(n=n,l=l,energy_list=energies,
               norm_list=norm_errors,ortho_list=ortho_errors,pde_list=pde_errors)
    plot_loss_curve(losses, save_path = "training.png")
    plot_energy_curve(energies,n=n, l=l, path="convergence")
    return energy

