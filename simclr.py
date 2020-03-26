import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50


def get_model(name='resnet18',  pretrained=False):
    """ get proper model from torchvision models. """
    model_list = ['resnet18', 'resnet34', 'resnet50']
    assert name in model_list, '{} not available, choose from {}'.format(name, model_list)
    classifier = eval(name)(pretrained=pretrained)

    return classifier


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class SimCLR(nn.Module):
    """
    We opt for simplicity and adopt the commonly used ResNet (He et al., 2016) 
    to obtain hi = f(x ̃i) = ResNet(x ̃i) where hi ∈ Rd is the output after the average pooling layer.
    """

    def __init__(self, backbone='resnet18', projection_dim=64, pretrained=False, normalize=True):
        super(SimCLR, self).__init__()

        self.normalize = normalize
        self.encoder = get_model(name=backbone, pretrained=pretrained)

        self.n_features = self.encoder.fc.in_features  # get dimensions of fc layer
        self.encoder.fc = Identity()  # remove fully-connected layer after pooling layer

        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features, projection_dim, bias=False),
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)

        if self.normalize:
            z = nn.functional.normalize(z, dim=1)
        return h, z
