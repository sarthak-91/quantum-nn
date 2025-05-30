import numpy as np 
import torch 
from scripts.network import * 
from scripts.training import * 
from scripts.visuals import * 
from scripts.load_store import * 
from scripts.loss import * 

if __name__ == "__main__":
    hidden_layers = [50]
    wave_form= NN(hidden_layers)
    n = 1
    l = 0
    N = 2000 
    r_min = 0.01 
    r_max_map = {1:20,2:30,3:35,4:55}
    r = np.linspace(r_min,r_max_map[n],N)
    r_torch = torch.tensor(r, dtype=torch.float32).unsqueeze(1)
    if n>1:
        excited = True
    else:
        excited=False
    wf_list=[]
    if excited:wf_list=load_wavefunctions_for_ortho(n=n,l=l,r=r_torch,hidden_layers=hidden_layers,model_class=NN, print_=True)
    energy = train(wave_form, r_torch=r_torch, 
                   epochs=400000,loss_fn=loss_fn_rayleigh, 
                   excited_state=excited,lr=1e-3,
                   wf_list=wf_list,n=n, l=l)
    print("Energy acquired = ",energy.item())
    store(wave_form, energy=energy,r=r_torch, n=n, l=l)
    plot_all_wavefunctions(r_torch,model_class=NN,hidden_layers=hidden_layers)
