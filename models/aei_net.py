import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import MultilevelAttributesEncoder, AADGenerator


class AEINet(nn.Module):
    def __init__(self, num_features=512):
        """

        Parameters
          num_features: int
            Number of features of the identity embedded vectors.
        """
        super(AEINet, self).__init__()
        self.encoder = MultilevelAttributesEncoder()
        self.generator = AADGenerator(num_features)

    def forward(self, Xt, z_id):
        attr = self.encoder(Xt)
        Y = self.generator(attr, z_id)
        return Y, attr

    def get_attr(self, X):
        return self.encoder(X)