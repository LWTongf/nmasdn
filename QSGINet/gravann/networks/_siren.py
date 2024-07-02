# From https://github.com/vsitzmann/siren/blob/master/explore_siren.ipynb
from torch import nn
import torch
import numpy as np
import torch.nn.functional as F
# from ._abs_layer import AbsLayer


class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, outermost_activation=None,
                 first_omega_0=30, hidden_omega_0=30.):

        super().__init__()

        self.in_features = in_features
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))
            if i == 5:  # Add self-attention layer after the 7th layer
                self.net.append(SelfAttentionLayer(hidden_features))


        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                             np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
            self.net.append(outermost_activation)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(
            True)  # allows to take derivative w.r.t. input
        # We must force the putput to be positive as it represents a density.
        output = self.net(coords)
        return output

    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)

                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()

                activations['_'.join(
                    (str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else:
                x = layer(x)

                if retain_grad:
                    x.retain_grad()

            activations['_'.join(
                (str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations
class SelfAttentionLayer(nn.Module):
    def __init__(self, in_features):
        super(SelfAttentionLayer, self).__init__()
        self.in_features = in_features
        self.query = nn.Linear(in_features, in_features)


        self.key = nn.Linear(in_features, in_features)


        self.value = nn.Linear(in_features, in_features)
        with torch.no_grad():
            self.query.weight.uniform_(-np.sqrt(6 / self.in_features),
                                       np.sqrt(6 / self.in_features))
            self.key.weight.uniform_(-np.sqrt(6 / self.in_features),
                                     np.sqrt(6 / self.in_features))
            self.value.weight.uniform_(-np.sqrt(6 / self.in_features),
                                    np.sqrt(6 / self.in_features))

    def forward(self, x):
        d_out=x.shape[1]
        cc=np.sqrt(d_out)
        cc=torch.tensor(cc)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        q_mul_k=torch.matmul(q, k.transpose(-2, -1))
        q_mul_k = map_norm(q_mul_k)
        q_mul_k = torch.sin(q_mul_k)
        scores= F.softmax(q_mul_k / cc, dim=-1)
        out=torch.matmul(scores, v)



        return out

def map_norm(kq_maps, eps=1e-05, dim=-1):
        mean_map = kq_maps.mean(dim)
        var_map = kq_maps.var(dim)
        mean_map=mean_map.view(200,1)
        var_map=var_map.view(200,1)
        # uu=(kq_maps - mean_map)
        kq_maps = (kq_maps - mean_map) / (torch.sqrt(var_map + eps))
        return kq_maps
if __name__ == "__main__":
    model=Siren(in_features=3, out_features=1, hidden_features=int(100),
          hidden_layers=int(9), outermost_linear=True, outermost_activation=nn.Tanh(),
          first_omega_0=30, hidden_omega_0=30)
    print(model)