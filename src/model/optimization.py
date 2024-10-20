import torch


class OptimHandler():
    def __init__(self, config):
        self.config = config
    
    def initialize_optimizer(self, params, lrs_dict, tracking=True):
        lrs = lrs_dict
        param_groups = [{'params': [v], 'name': k, 'lr': lrs[k]} for k, v in params.items()]

        if tracking:
            return torch.optim.Adam(param_groups)
        else:
            return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)

    def early_check(
            self,
            early_stop_count,
            last_loss,
            loss, 
            early_stop_eval,
            early_stop_time_thresh=20,
            early_stop_thresh=0.0001):
         
        if self.config['early_stop']:
            if abs(last_loss - loss.detach().clone().item()) < early_stop_thresh:
                early_stop_count += 1
                if early_stop_count == early_stop_time_thresh:
                    early_stop_eval = True
            else:
                early_stop_count = 0
            last_loss = loss.detach().clone().item()

        return early_stop_eval, early_stop_count, last_loss