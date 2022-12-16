import torch

class LaLPairDataset(torch.utils.data.Dataset):
    def __init__(self, features, click_rates):
        
        self.features = torch.tensor(features.values, dtype=torch.float32)
        self.click_rates = torch.tensor(click_rates.values, dtype=torch.float32)    
        
    def __len__(self):
        return len(self.click_rates)
    
    
    def __getitem__(self, idx):
        # define primary user and his click_rate
        prime_click_rate = self.click_rates[idx]
        primary_user = self.features[idx]
        
        # Generate primary users copies tensor
#         primary_user_tensor = torch.stack([primary_user]*len(self.click_rates), axis=0)
#         primary_user_click_rate_tensor = torch.stack([prime_click_rate]*len(self.click_rates), axis=0)
        
        # generate other users stack tensor
        other_users_tensor = list()
        other_users_click_rate_tensor = list()
        for user_index in range(len(self.click_rates)):
            other_users_tensor.append(self.features[user_index])
            other_users_click_rate_tensor.append(self.click_rates[user_index])
        
#         return primary_user_tensor, torch.stack(other_users_tensor, axis=0), primary_user_click_rate_tensor, torch.stack(other_users_click_rate_tensor, axis=0)
        return primary_user, torch.stack(other_users_tensor, axis=0), prime_click_rate, torch.stack(other_users_click_rate_tensor, axis=0)
            
            