import torch
import torch.nn as nn

def _init_rnn_weights(rnn: nn.Module,
                      init_xavier: bool=True,
                      init_normal: bool=True,
                      init_gain: float=1.0,
                      ):
    for name, p in rnn.named_parameters():
        if 'bias' in name:
            p.data.fill_(0)
            if isinstance(rnn, (torch.nn.LSTM, torch.nn.LSTMCell)):
                n = p.nelement()
                p.data[n // 4:n // 2].fill_(1)  # forget bias
        elif 'weight' in name:
            if init_xavier:
                if init_normal:
                    nn.init.xavier_normal(p, init_gain)
                else:
                    nn.init.xavier_uniform(p, init_gain)
            else:
                if init_normal:
                    nn.init.normal(p, init_gain)
                else:
                    nn.init.uniform(p, init_gain)


class RNNModel(nn.Module):
    """Simple RNNmodel
    """
    def __init__(self,
                 ntoken,
                 input_size,
                 hidden_size,
                 rnn_type='LSTM',                 
                 num_layers=1,
                 bidirectional=False,
                 dropout=0.0,
                 batch_first=True,
                 init_xavier: bool=True,
                 init_normal: bool=True,
                 init_gain: float=1.0,
                 concat: bool=True,
                 ):
        super(RNNModel, self).__init__()
        self.in_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.concat = concat
        self.rnn_type = rnn_type
        if not self.rnn_type in ['LSTM', 'GRU', 'RNN']:
            raise NotImplementedError(self.rnn_type)
        self.rnn =\
                   getattr(nn, rnn_type)(input_size,
                                         hidden_size,
                                         num_layers,
                                         bidirectional=bidirectional,
                                         dropout=dropout,
                                         batch_first=batch_first)
        _init_rnn_weights(self.rnn,
                          init_xavier=init_xavier,
                          init_normal=init_normal,
                          init_gain=init_gain
                          )

        self.encoder = nn.Embedding(ntoken, input_size)        
        self.decoder = nn.Linear(hidden_size, ntoken)

    def forward(self, x, hx=None):
        emb = self.encoder(x)

        self.rnn.flatten_parameters()
        output, hx = self.rnn(emb, hx)
        self.rnn.flatten_parameters()
        
        if (not self.concat) and self.bidirectional:
            B, T, F = output.size()
            output = output[:, :, :F//2] + output[:, :, F//2:]
            
        return output, hx                    
        
