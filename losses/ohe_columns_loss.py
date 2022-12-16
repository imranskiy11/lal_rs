from torch.nn.modules.loss import _Loss
from torch import argmax
from typing import List
from torch import functional as F

class OHELoss(_Loss):
    
  def __init__(self, indexes_ohe: List[tuple]):
    super(OHELoss, self).__init__()
    self.indexes_one = indexes_ohe
    
  def forward(self, input, target):
    """ loss function called at runtime """ 
    sum_loss = 0
    for _slice in self.indexes_one:
        current_loss = F.nll_loss(
            F.log_softmax(input[:, _slice[0]:_slice[1]], dim=1),
            argmax(target[:, _slice[0]:_slice[1]])
        )
        sum_loss += current_loss
    
    return sum_loss
    
    
