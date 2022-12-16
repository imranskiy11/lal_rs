import torch

class LaLLoss(torch.nn.Module):

    def __init__(self, margin=0.0):
        super(LaLLoss, self).__init__()
        self.margin = margin
#         self.embedding_cos_similarity = torch.nn.CosineEmbeddingLoss()
        self.cos_sim_ =  torch.nn.CosineSimilarity(dim=1, eps=1e-08)
        
    def click_rate_similarity(self, cr1, cr2):
        return 1 - abs(cr1 - cr2)
    
    def _loss_diff(self, emb_sim, click_sim):
        return abs(emb_sim - click_sim)

    def forward(self, 
                comparable_user_tensor: torch.Tensor,
                other_users_tensor: torch.Tensor, 
                comparable_user_click_rate: torch.Tensor, 
                other_users_click_rate: torch.Tensor,
                target_tensor: torch.Tensor = torch.ones(1),
                previous_iteration_loss_value = 0,
                batch_size=1) -> torch.Tensor:

        click_rate_sim = self.click_rate_similarity(comparable_user_click_rate, other_users_click_rate) # -> tensor with shape click_rate1or2.shape
     
        if batch_size==1:            
#             cosine_sim_embeddings = torch.from_numpy(cosine_similarity(
#                 comparable_user_tensor.numpy(), 
#                 other_users_tensor.squeeze(0).numpy()))
#             print(cosine_sim_embeddings.shape)
            cosine_sim_embeddings = self.cos_sim_(other_users_tensor.squeeze(0), comparable_user_tensor)
            
#         else:
#             cosine_sim_embeddings = torch.from_numpy(cosine_similarity(
#                 comparable_user_tensor.view((1,) + tuple(other_users_tensor.shape[1:])).numpy(), 
#                 other_users_tensor.numpy()))

#         assert click_rate_sim.shape == cosine_sim_embeddings.shape, 'Tensors shapes error'
        assert isinstance(click_rate_sim, type(cosine_sim_embeddings)), 'Tensors type isistance error'
        
        similarity_residual = self._loss_diff(click_rate_sim, cosine_sim_embeddings)
        
        return torch.mean(similarity_residual) + previous_iteration_loss_value