import torch

import torch.nn.functional as F
import torch.nn as nn
class Encoder(nn.Module):
  def __init__(self, input_dim = 15, hidden_dim = 256, output_dim = 5):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )

  def forward(self, x):
      v = self.encoder(x)
      return F.normalize(v, p=2, dim=-1)

class ClassifierHead(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=512, output_dim=105):
        super(ClassifierHead, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.classifier = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, output_dim),
        )
    
    def forward(self, x):
        return self.classifier(x)