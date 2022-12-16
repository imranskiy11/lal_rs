import torch.nn as nn

class SiameseLaLNetwork(nn.Module):
    def __init__(self, input_shape=80):
        super(SiameseLaLNetwork, self).__init__()

        self.encode = nn.Sequential(
            nn.Linear(input_shape, 64),
#             nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Dropout(0.1),

            nn.Linear(64, 32),
            nn.ReLU(True),
#             nn.BatchNorm1d(32),
            nn.Dropout(0.1),
            
            nn.Linear(32, 32)
        )
        self.last_linear = nn.Linear(32, 32)



    def forward(self, input1, input2):
        return self.encode(input1), self.encode(input2)