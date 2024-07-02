import numpy as np
import torch
import torch.nn as nn
class AbsLayer(nn.Module):
    """Very simple activation layer to allow different abs layer activations of the  QSGINet
    """

    def __init__(self):
        super().__init__()
        self.tanh=nn.Tanh()
    def forward(self, input):
        # Differentiated training
        # activation=self.tanh(input)
        # Direct training
        activation = torch.abs(input)
        return activation

class Decoder(nn.Module):
    def forward(self, *args, **kwargs):
        return self.fc_block(*args, **kwargs)


class Network(nn.Module):
    def __init__(self, in_features, decoder_hidden_dim=256, nl='sine', encoder_type=None,
                 decoder_n_hidden_layers=8, init_type='siren', neuron_type='quadratic'):
        super().__init__()
        self.encoder_type = encoder_type
        self.in_features = in_features
        self.init_type = init_type
        self.decoder = Decoder()

        if neuron_type == 'quadratic':
            self.decoder.fc_block = QuaNet(d_in=in_features, nl=nl, n_layers=decoder_n_hidden_layers,
                                           d_hidden=decoder_hidden_dim,
                                           init_type=init_type)
        else:
            raise ValueError("unsupported neuron type")

    def forward(self, points, mnfld_pnts=None):
        batch_size = points.shape[0]
        rho = self.decoder(points.view(-1, points.shape[-1])).reshape(batch_size, -1)
        return rho




class QuaNet(nn.Module):
    def __init__(
            self,
            d_in=3,
            d_out=1,
            d_hidden=100,
            n_layers=9,
            nl='softplus',
            init_type='siren',
    ):
        super().__init__()
        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]
        self.num_layers = len(dims)
        self.init_type = init_type

        for l in range(self.num_layers - 1):
            if l == 0:
                qua = QuadraticLayer(dims[l], dims[l + 1])
            else:
                qua = QuadraticLayer(dims[l], dims[l + 1])
            setattr(self, "qua" + str(l), qua)
        self.abs_layer = AbsLayer()


        self.apply(sine_init)
        self.qua0.apply(first_layer_sine_init)


        if nl == 'softplus':
            self.activation = nn.Softplus(beta=100)
        elif nl == 'tanh':
            self.activation = nn.Tanh()
        elif nl == "sine":
            self.activation = Sine()
        elif nl == "relu":
            self.activation = nn.ReLU()

    def forward(self, inputs):
        x = inputs
        for l in range(self.num_layers - 1):
            qua = getattr(self, "qua" + str(l))
            if l == self.num_layers - 2:
                x = qua(x)
                x = self.abs_layer(x)
            else:
                x = qua(x)
                x = self.activation(x)
        return x




class QuadraticLayer(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.lin1 = nn.Linear(d_in, d_out)
        self.lin2 = nn.Linear(d_in, d_out)
        self.lin3 = nn.Linear(d_in, d_out)

    def forward(self, x):
        return torch.mul(self.lin1(x), self.lin2(x)) + self.lin3(torch.square(x))


def init_lin2_lin3(m):
    # 原始
    nn.init.normal_(m.lin2.weight, mean=0.0, std=1e-16)
    nn.init.ones_(m.lin2.bias)
    nn.init.normal_(m.lin3.weight, mean=0.0, std=1e-16)
    nn.init.zeros_(m.lin3.bias)



################################# SIREN's initialization ###################################
# This file is borrowed from DiGS: https://github.com/Chumbyte/DiGS
def sine_init(m):
    with torch.no_grad():
        if hasattr(m, "lin1") and hasattr(m, "lin2") and hasattr(m, "lin3"):
            num_input = m.lin1.weight.size(-1)
            # See SIREN paper supplement Sec. 1.5 for discussion of factor 30
            m.lin1.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)
            init_lin2_lin3(m)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, "lin1") and hasattr(m, "lin2") and hasattr(m, "lin3"):
            num_input = m.lin1.weight.size(-1)
            # See SIREN paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.lin1.weight.uniform_(-1 / num_input, 1 / num_input)
            init_lin2_lin3(m)


class Sine(nn.Module):
    def forward(self, input):
        # See SIREN paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)


