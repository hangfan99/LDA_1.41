import os
import torch
import numpy as np
import yaml
from collections import OrderedDict
from networks.forecast_net import forecast_net
import torch.utils.checkpoint as checkpoint

class era5_reader:
    def __init__(self,data_dir, device):
        self.device = device
        self.data_dir = data_dir

    def get_state(self, tstamp):
        state = []
        single_level_vnames = ['msl', 'u10', 'v10', 't2m', ]
        multi_level_vnames = ['z','q', 'u', 'v', 't']
        height_level = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
        for vname in single_level_vnames:
            file = os.path.join('single/'+str(tstamp.year), str(tstamp.to_datetime64()).split('.')[0]).replace('T', '/')
            url = f"{self.data_dir}/{file}-{vname}.npy"
            state.append(np.load(url).reshape(1, 128, 256))
        for vname in multi_level_vnames:
            file = os.path.join(str(tstamp.year), str(tstamp.to_datetime64()).split('.')[0]).replace('T', '/')
            for idx in range(13):
                height = height_level[idx]
                url = f"{self.data_dir}/{file}-{vname}-{height}.0.npy"
                state.append(np.load(url).reshape(1, 128, 256))
        state = np.concatenate(state, 0)
        return torch.from_numpy(state).to(self.device)


class forecast_model:
    def __init__(self, device):
        self.device     = device
        self.model_mean, self.model_std = self.get_model_mean_std()
        self.model = self.init_model()
        self.nlon          = 256
        self.nlat          = 128
        self.nchannel      = 69
        self.forecast_indices = torch.tensor([1, 2, 3, 0] + [i+4 for i in range(65)]).cuda()
        self.ae_indices = torch.tensor([3, 0, 1, 2,] + [i+4 for i in range(65)]).cuda()
    
    def get_model_mean_std(self):
        mean_layer = np.array([-0.14186215714480854, 0.22575792335029873, 278.7854495405721, 100980.83590625007, 199832.31609374992, 157706.1917968749, 132973.8087890624, 115011.55044921875, 100822.13164062506, 88999.83613281258, 69620.0044531249, 53826.54542968748, 40425.96180664062, 28769.254521484374, 13687.02337158203, 7002.870792236329, 777.5631800842285, 2.8248029025235157e-06, 2.557213611567022e-06, 4.689598504228342e-06, 1.7365863168379306e-05, 5.37612270545651e-05, 0.00012106754767955863, 0.0003586592462670523, 0.0007819174298492726, 0.0014082587775192225, 0.002245682779466732, 0.004328316930914292, 0.005698622210184111, 0.006659231842495503, 4.44909584343433, 10.046632840633391, 14.321160042285918, 15.298378415107727, 14.48938421010971, 12.895844810009004, 9.628437678813944, 7.07798705458641, 5.110536641478544, 3.4704639464616776, 1.2827875773236155, 0.3961004569224316, -0.18604825597634778, 0.012106836824341376, 0.1010729405652091, 0.2678451650420902, 0.2956721917196408, 0.21001753183547414, 0.03872977272505523, -0.04722135595180817, 0.0007164070030103152, -0.022026948712546065, 0.0075308467486320295, 0.014846984493779027, -0.062323193841984835, -0.15797925526494516, 214.66564151763913, 210.3573041915893, 215.23375904083258, 219.73181056976318, 223.53410289764412, 228.6614455413818, 241.16466262817383, 251.74072200775146, 259.84156120300344, 265.99485839843743, 272.77368919372566, 275.3001181793211, 278.5929747772214])
        std_layer = np.array([5.610453475051704, 4.798220612223473, 21.32010786700973, 1336.2115992274876, 3755.2810557402927, 4357.588191568988, 5253.301115477269, 5540.73074484052, 5405.73040397736, 5020.194961603476, 4104.233456672573, 3299.702929930327, 2629.7201995715513, 2060.9872289877453, 1399.3410970050247, 1187.5419349409494, 1098.9952409939283, 1.1555282996146702e-07, 4.2315237954921815e-07, 3.1627283344500357e-06, 2.093742795871515e-05, 7.02963683704546e-05, 0.00016131853114827985, 0.00048331132466880735, 0.001023028433607086, 0.0016946778969914426, 0.0024928432426471183, 0.004184742037434761, 0.005201345241925773, 0.00611814321149996, 11.557361639969054, 11.884088705628045, 15.407016747306344, 17.286773058038722, 17.720698660431694, 17.078782531259524, 14.509924979003983, 12.215305549952125, 10.503871726997783, 9.286354460633103, 8.179197305830433, 7.93264239491015, 6.126056325796786, 8.417864770061094, 8.178248048405905, 9.998695230009567, 11.896325029659364, 13.360381609448558, 13.474533447403218, 11.44656476066317, 9.321096224035244, 7.835396470389893, 6.858187372121642, 6.186618416862026, 6.345356147017278, 5.23175612906023, 9.495652698988557, 13.738672642636256, 9.090666595626503, 5.933385737657316, 7.389004707914384, 10.212310312072752, 12.773099916244078, 13.459313552230206, 13.858620163486986, 15.021590351519892, 16.00275340237577, 16.88523210573196, 18.59201174892538])
        mean_layer_gpu = torch.from_numpy(mean_layer).float().to(self.device)
        std_layer_gpu  = torch.from_numpy(std_layer).float().to(self.device)
        return mean_layer_gpu, std_layer_gpu

    def init_model(self,):
        with open("../preprocessing_params/forecast_model.yaml", 'r') as cfg_file:
            cfg_params = yaml.load(cfg_file, Loader = yaml.FullLoader)
        model = forecast_net(**cfg_params["model"]["network_params"])
        checkpoint_dict = torch.load("../ckps/forecast_checkpoint.pth",weights_only=True)
        checkpoint_model = checkpoint_dict['model']
        new_state_dict = OrderedDict()
        for k, v in checkpoint_model.items():
            if "module" == k[:6]:
                name = k[7:]
            else:
                name = k
            if not name == "max_logvar" and not name == "min_logvar":
                new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        model.to(self.device)
        model.eval()
        return model

    def integrate(self, xa, step, detach=True, ckp=False):
        xa = xa[self.forecast_indices]
        xa = (xa - self.model_mean.reshape(-1, 1, 1)) / self.model_std.reshape(-1, 1, 1)
        xa = xa.unsqueeze(0)
        for i in range(step):
            if detach:
                xa = self.model(xa)[:, :self.nchannel].detach()
            elif ckp:
                xa = checkpoint.checkpoint(self.model, xa)[:, :self.nchannel]
            else:
                xa = self.model(xa)[:, :self.nchannel]

        xa = xa[0]
        xa = xa.reshape(self.nchannel, self.nlat, self.nlon) * self.model_std.reshape(-1, 1, 1) + self.model_mean.reshape(-1, 1, 1)
        xa = xa[self.ae_indices]
        return xa


class GDAS_loader:
    def __init__(self, device, data_root='../../data/gdas/2017_conventional',err_scale=[4, 4, 4,  1, 1, 1, 1]):
        self.data_root = data_root
        self.file_list = os.listdir(data_root)
        self.file_list.sort()
        self.t2_correction = np.load("../preprocessing_params/t2m_elevation_correction.npy")
        self.device = device
        self.max_error_layer = self.init_max_error_layer(err_scale)

    def get_obs(self, current_time):
        obs_data, obs_mask = (np.load(current_time.strftime(self.data_root+'/%Y-%m-%d_%H.npy')))
        obs_data[3] += self.t2_correction
        
        obs_data = torch.from_numpy(obs_data).unsqueeze(0).to(self.device).float()
        obs_mask = torch.from_numpy(obs_mask).unsqueeze(0).to(self.device)
        
        obs_data[obs_data>30] = 0
        obs_mask[obs_data>30] = 0
        obs_data[obs_data<-30] = 0
        obs_mask[obs_data<-30] = 0
        
        mask_01 = obs_mask.clone()
        mask_01[mask_01>0.1] = 1        
        obs_data = obs_data * mask_01
        
        return obs_data, obs_mask

    def quality_control_obs(self, current_time, background, if_print=False):
        obs_data, obs_mask = self.get_obs(current_time)
        
        before_num = (obs_mask>0).sum()
        obs_mask[(background.to(self.device)-obs_data).abs()>self.max_error_layer.to(self.device)] = 0
        obs_data[(background.to(self.device)-obs_data).abs()>self.max_error_layer.to(self.device)] = 0
        
        after_num = (obs_mask>0).sum()

        if if_print:
            print( (1-(after_num/before_num).item())*100, '% observations are dropped')
        
        return obs_data , obs_mask

    def init_max_error_layer(self, err_scale):
        history_err_raw = np.load('../preprocessing_params/ERA5_48h_deviation.npy')
        err_layer_mean = history_err_raw.mean(axis=(1,2))
        ratio_layer = np.ones_like(err_layer_mean)
        
        ratio_layer[0] *= err_scale[0]
        ratio_layer[1:3] *= err_scale[1]
        ratio_layer[3] *= err_scale[2]
        
        ratio_layer[4:17] *= err_scale[3]
        ratio_layer[17:30] *= err_scale[4]
        ratio_layer[30:43] *= err_scale[5]
        ratio_layer[43:56] *= err_scale[5]
        ratio_layer[56:69] *= err_scale[6]
        
        err_layer = ratio_layer*err_layer_mean
        err_layer = torch.from_numpy(err_layer).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(self.device)
        return err_layer

    def valid_saperate_col(self, obs, mask, upr_station=16, surface_staion=300, north_ratio=0.7):
        mask_01 = mask.clone()
        mask_01[mask>0] = 1
        
        column_mask =mask_01.sum(axis=(1))[0]
        alues, flat_indices = torch.topk(column_mask[:64].flatten(), int(upr_station*north_ratio))

        north_lats = flat_indices // column_mask.shape[1]  
        north_lons = flat_indices % column_mask.shape[1]
        
        alues, flat_indices = torch.topk(column_mask[64:].flatten(), int(upr_station*(1-north_ratio)))
        south_lats = flat_indices // column_mask.shape[1] + 64
        south_lons = flat_indices % column_mask.shape[1] 
        lats = torch.cat([north_lats,south_lats])
        lons = torch.cat([north_lons,south_lons])
        
        surface_mask =mask_01[:,:4].sum(axis=(1))[0]
        alues, flat_indices = torch.topk(surface_mask.flatten(), int(surface_staion))
        surface_lats = flat_indices // surface_mask.shape[1]
        surface_lons = flat_indices % surface_mask.shape[1] 
        
        lons = list(lons.cpu().numpy())
        lats = list(lats.cpu().numpy())
        surface_lons = list(surface_lons.cpu().numpy())
        surface_lats = list(surface_lats.cpu().numpy())
        
        valid_obs_mask = torch.zeros_like(mask)
        for i in range(len(lons)):
            valid_obs_mask[:,:,lats[i],lons[i]] = 1
        for i in range(len(surface_lons)):
            valid_obs_mask[:,:4,surface_lats[i],surface_lons[i]] = 1
    
        da_obs_mask = mask_01 - valid_obs_mask
    
        da_obs = obs*da_obs_mask
        da_mask = mask * da_obs_mask
        valid_obs = obs*valid_obs_mask
        valid_mask = mask*valid_obs_mask
        return da_obs, da_mask, valid_obs, valid_mask


### Metrics 
@torch.jit.script
def lat(j: torch.Tensor, num_lat: int) -> torch.Tensor:
    return 90. - j * 180./float(num_lat-1)

@torch.jit.script
def latitude_weighting_factor_torch(j: torch.Tensor, num_lat: int, s: torch.Tensor) -> torch.Tensor:
    return num_lat * torch.cos(torch.pi/180. * lat(j, num_lat)) / s

@torch.jit.script
def weighted_rmse_torch_channels(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    num_lat = pred.shape[2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)

    s = torch.sum(torch.cos(torch.pi/180. * lat(lat_t, num_lat)))
    weight = torch.reshape(latitude_weighting_factor_torch(lat_t, num_lat, s), (1, 1, -1, 1))
    result = torch.sqrt(torch.mean(weight*(pred - target)**2., dim=(-1,-2)))
    return result

@torch.jit.script
def weighted_rmse_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    result = weighted_rmse_torch_channels(pred, target)
    return torch.mean(result, dim=0)

def WRMSE(pred, gt):
    return weighted_rmse_torch(pred, gt)

def err_at_obs(field, obs, obs_mask):
    mask_01 = obs_mask.clone()
    mask_01[mask_01>0.1] = 1
    squared_error = ((field - obs) * mask_01) ** 2
    valid_obs_count = mask_01.sum(axis=(0,2,3))
    sum_squared_error = squared_error.sum(axis=(0,2,3))
    rmse = torch.sqrt(sum_squared_error / (valid_obs_count+1e-10))
    return rmse