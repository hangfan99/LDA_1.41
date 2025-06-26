import os
import torch
import numpy as np

class Latent_Var:
    def __init__(self, forecast_model, AE, Back_err, device):
        self.model = forecast_model
        self.AE = AE
        self.stastic_B_flatten = torch.from_numpy(((Back_err**2).sum(axis=0)/(Back_err.shape[0]-1)).flatten()).to(device)
        self.mean_layer = torch.from_numpy(np.load('../preprocessing_params/mean_layer.npy')).float().to(device).reshape(1,-1, 1, 1)
        self.std_layer = torch.from_numpy(np.load('../preprocessing_params/std_layer.npy')).float().to(device).reshape(1,-1, 1, 1)
    
    def back_loss(self,z, zb, B_flatten):
        loss = torch.sum((z.flatten()-zb.flatten())**2/(B_flatten+1e-6))/2
        # print('back_loss:', loss)
        return loss
    
    def y_loss(self,x_pred, y, H, obs_noise):
        loss = torch.sum( (H * x_pred.contiguous() - y.contiguous()) ** 2 / obs_noise**2 ) / 2
        # print('y_loss:', loss)
        return loss
    
    def latent_Var_adam(self, background, y, H,  max_step=100, lr=0.1, obs_noise=0.1, B_flatten=None, window_interval=1):

        if not B_flatten:
           B_flatten = self.stastic_B_flatten
            
        DA_cycle = H.shape[0]


        background = (background - self.mean_layer)/self.std_layer
        y = ((y - self.mean_layer)/self.std_layer)*H
        
        background_latent = self.AE.encode(background).mode()
        
        z = torch.autograd.Variable(torch.ones_like(background_latent)*background_latent, requires_grad=True)
        
        optimizer = torch.optim.Adam([z], lr=lr,)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",factor=0.5,patience=5,verbose=False,)
    
        best_loss = 1e19
        best_ = 0
        
        if H.shape[0] <= 2:
            ckp=False
        else:
            ckp=True
            
        self.AE.eval()
        for _ in range(max_step):
            optimizer.zero_grad()
            x = self.AE.decode(z)[0]
            x_list = [x,]
            x = x*self.std_layer[0] + self.mean_layer[0]

            for i in range(y.shape[0]-1):
                x = self.model.integrate(x, window_interval, detach=False, ckp=ckp) 
                x_list.append((x-self.mean_layer[0])/self.std_layer[0])
            x_pred = torch.stack(x_list, 0)
    
            loss = ((self.back_loss(z, background_latent, B_flatten) + self.y_loss(x_pred, y, H, obs_noise)))/1e4  # This is an empirical scaling factor :).
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_z = z
                best_ = _
            elif _ - best_ > 20:
                break
    
            z.retain_grad()
            loss.backward(retain_graph=True)
    
            torch.nn.utils.clip_grad_norm_(x, max_norm=1.0)
    
            optimizer.step()
            scheduler.step(loss)
            
        torch.cuda.empty_cache()
        
        x = self.AE.decode(best_z)[0]
        x = x*self.std_layer[0] + self.mean_layer[0]
        x_list = [x,] 
        for i in range(DA_cycle-1):
            x = self.model.integrate(x, window_interval)
            x_list.append(x)
        x_ana = torch.stack(x_list, 0)

        return x_ana