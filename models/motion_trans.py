import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class Encoder(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_heads, n_layers, batch_size, device):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.device = device
        self.embed = nn.Linear(input_size, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=n_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        # self.gru = nn.ModuleList([nn.GRUCell(hidden_size, hidden_size) for i in range(self.n_layers)])
        self.mu_net = nn.Linear(hidden_size, output_size)
        self.logvar_net = nn.Linear(hidden_size, output_size)
        self.hidden = self.init_hidden()

    def init_hidden(self, num_samples=None):
        batch_size = num_samples if num_samples is not None else self.batch_size
        hidden = []
        for i in range(self.n_layers):
            hidden.append(torch.zeros(batch_size, self.hidden_size).requires_grad_(False).to(self.device))
        self.hidden = hidden
        return hidden

    def reparameterize(self, mu, logvar):
        s_var = logvar.mul(0.5).exp_()
        eps = s_var.data.new(s_var.size()).normal_()
        return eps.mul(s_var).add_(mu)

    def forward(self, inputs):
        embedded = self.embed(inputs.view(-1, self.input_size))
        # print(embedded.shape)
        h_in = embedded
        # embedded = self.pos_encoder(embedded)
        # print(embedded.shape)
        transformer_output = self.transformer_encoder(embedded)
        # print(transformer_output.shape)
        # final_output = transformer_output[-1]
        # for i in range(self.n_layers):
        #     self.hidden[i] = self.gru[i](h_in, self.hidden[i])
        #     h_in = self.hidden[i]
        mu = self.mu_net(transformer_output)
        logvar = self.logvar_net(transformer_output)

        z = self.reparameterize(mu, logvar)
        return z, mu, logvar, h_in, transformer_output


class Decoder(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_heads, n_layers, batch_size, device):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.device = device
        self.embed = nn.Linear(input_size, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size)
        # self.gru = nn.ModuleList([nn.GRUCell(hidden_size, hidden_size) for i in range(self.n_layers)])
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=n_heads)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        self.output = nn.Linear(hidden_size, output_size)
        self.hidden = self.init_hidden()

    def init_hidden(self, num_samples=None):
        batch_size = num_samples if num_samples is not None else self.batch_size
        hidden = []
        for i in range(self.n_layers):
            hidden.append(torch.zeros(batch_size, self.hidden_size).requires_grad_(False).to(self.device))
        self.hidden = hidden
        return hidden

    def forward(self, inputs, memory):
        embedded = self.embed(inputs.view(-1, self.input_size))
        h_in = embedded
        # embedded = self.pos_encoder(embedded)

        # for i in range(self.n_layers):
        #     self.hidden[i] = self.gru[i](h_in, self.hidden[i])
        #     h_in = self.hidden[i]
        transformer_output = self.transformer_decoder(embedded, memory)
        # final_output = transformer_output[-1]
        
        return self.output(transformer_output), transformer_output

# generator with Lie algbra parameters, root joint has no rotations
class DecoderGRULie(DecoderGRU):
    def __init__(self, input_size, output_size, hidden_size, n_heads, n_layers, batch_size, device):
        super(DecoderGRULie, self).__init__(input_size,
                                            output_size,
                                            hidden_size,
                                            n_heads,
                                            n_layers,
                                            batch_size,
                                            device)
        self.output_lie = nn.Linear(output_size - 3, output_size - 3)
        self.PI = 3.1415926
        self.tanh = nn.Tanh()

    def forward(self, inputs, memory):
        hidden_output, h_mid = super(DecoderGRULie, self).forward(inputs, memory)
        root_trans = hidden_output[..., :3]
        lie_hid = hidden_output[..., 3:]
        lie_hid = self.tanh(lie_hid)
        lie_out = self.output_lie(lie_hid)
        lie_out = self.tanh(lie_out) * self.PI
        output = torch.cat((root_trans, lie_out), dim=-1)
        # print(output.shape)
        return output, h_mid
