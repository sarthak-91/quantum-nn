import numpy as np 
import torch 
from scripts.network import * 
from scripts.training import * 
from scripts.visuals import * 
from scripts.load_store import * 
from scripts.loss import * 

if __name__ == "__main__":
    N = 1000 
    r = np.linspace(0.001,30,N)
    r_torch = torch.tensor(r, dtype=torch.float32).unsqueeze(1)
    wave_form= NN([10,10])
    n = 3
    l = 2
    if n>1:
        excited = True
    else:
        excited=False
    wf_list=[]
    if excited:wf_list=load_wavefunctions_for_ortho(n,l,print_=True)
    energy = train(wave_form, r_torch=r_torch, 
                   epochs=300000,loss_fn=loss_fn_rayleigh, 
                   excited_state=excited,lr=1e-3,
                   wf_list=wf_list,n=n, l=l)
    print("Energy acquired = ",energy.item())
    store(wave_form, energy=energy,r=r_torch, n=n, l=l)
    plot_all_wavefunctions(r_tensor=r)
