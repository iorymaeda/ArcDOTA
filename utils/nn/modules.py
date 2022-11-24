import math
import time
import random
from typing import Literal, Any

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .tools import no_dropout, no_layer_norm, get_indicator, get_module_device 

GATED_ACT = ['swiglu', 'glu']
# ----------------------------------------------------------------------------------------------- #
# Stuff
def swiglu(x:torch.Tensor):
    # |x| : (..., Any)
    x, gate = x.chunk(2, dim=-1)
    x = F.silu(gate) * x
    # |x| : (..., Any//2)
    return x

class SwiGLU(nn.Module):
    """https://arxiv.org/abs/2002.05202"""
    def forward(self, x:torch.Tensor):
        return swiglu(x)
        
class LayerNorm(nn.Module):
    """LayerNorm without bias"""
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x:torch.Tensor):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

class SparseConnectedLayer(nn.Module):
    """Linear layer with DropConnect"""
    
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, pw: float = 0.2, pb: float = 0.2,
                 bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(SparseConnectedLayer, self).__init__()
        
        self.pw = pw
        self.pb = pb
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = self.bias
        
        w = F.dropout(self.weight, self.pw, training=self.training)
        b = b if b is None else F.dropout(b, self.pb, training=self.training)
        return F.linear(x, w, b)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}, pw={}, pb={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.pw, self.pb
        )

class Callable(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Callable, self).__init__()
        
    def forward(self, x: Any) -> Any:
        return x

class WeightDrop(nn.Module):
    def __init__(self, module, weights, dropout=0, variational=False):
        super(WeightDrop, self).__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self.variational = variational
        self._setup()

    def widget_demagnetizer_y2k_edition(*args, **kwargs):
        # We need to replace flatten_parameters with a nothing function
        # It must be a function rather than a lambda as otherwise pickling explodes
        # We can't write boring code though, so ... WIDGET DEMAGNETIZER Y2K EDITION!
        # (╯°□°）╯︵ ┻━┻
        return

    def _setup(self):
        # Terrible temporary solution to an issue regarding compacting weights re: CUDNN RNN
        if issubclass(type(self.module), torch.nn.RNNBase):
            self.module.flatten_parameters = self.widget_demagnetizer_y2k_edition

        for name_w in self.weights:
            # print('Applying weight drop of {} to {}'.format(self.dropout, name_w))
            if isinstance(self.module, ZoneoutRNN):
                w = getattr(self.module.frame.forward_cell, name_w)
                del self.module.frame.forward_cell._parameters[name_w]
                self.module.frame.forward_cell.register_parameter(name_w + '_raw', nn.Parameter(w.data))
            else:
                w = getattr(self.module, name_w)
                del self.module._parameters[name_w]
                self.module.register_parameter(name_w + '_raw', nn.Parameter(w.data))

    def _setweights(self):
        for name_w in self.weights:
            if isinstance(self.module, ZoneoutRNN):
                raw_w = getattr(self.module.frame.forward_cell, name_w + '_raw')
            else:
                raw_w = getattr(self.module, name_w + '_raw')

            w = None
            if self.variational:
                mask = torch.autograd.Variable(torch.ones(raw_w.size(0), 1))
                if raw_w.is_cuda: mask = mask.cuda()
                mask = F.dropout(mask, p=self.dropout, training=True)
                w = mask.expand_as(raw_w) * raw_w
            else:
                w = F.dropout(raw_w, p=self.dropout, training=self.training)

            if isinstance(self.module, ZoneoutRNN):
                setattr(self.module.frame.forward_cell, name_w, torch.nn.Parameter(w))
            else:
                setattr(self.module, name_w, torch.nn.Parameter(w))

    def forward(self, *args, **kwargs):
        self._setweights()
        return self.module.forward(*args, **kwargs)

class LockedDropout(nn.Module):
    def __init__(self, batch_first=True):
        super().__init__()
        assert batch_first == True
        self.batch_first = batch_first

    def forward(self, x: torch.Tensor, dropout=0.5):
        # |x| : (batch_size, T, embed_dim) if batch_first
        # |x| : (T, batch_size, embed_dim) if not batch_first
        if not self.training or not dropout:
            return x

        if self.batch_first:
            if len(x.shape) == 3:
                m = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - dropout)
            elif len(x.shape) == 2:
                m = x.data.new(x.size(0), x.size(1)).bernoulli_(1 - dropout)
            else:
                raise Exception
        else:
            raise Exception

        mask = torch.autograd.Variable(m, requires_grad=False) / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, dropout:float=0., init:str='none', init_kwargs:dict={}, max_norm:float=None, padding_idx:int=None):
        super(Embedding, self).__init__()
        assert init in ['none', 'kaiming_uniform', 'uniform', 'normal']

        self.embeddings = nn.Embedding(num_embeddings, embedding_dim, max_norm=max_norm, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout)
        if init == 'kaiming_uniform':
            nn.init.kaiming_uniform_(self.embeddings.weight, **init_kwargs)
        elif init == 'uniform':
            nn.init.uniform_(self.embeddings.weight, **init_kwargs)
        elif init == 'normal':
            nn.init.normal_(self.embeddings.weight, **init_kwargs)

    def forward(self, inputs: torch.IntTensor | torch. FloatTensor) -> torch.Tensor:
        # |inputs| : (*)
        if inputs.dtype is torch.int64:
            outputs = self.embeddings(inputs)
            outputs = self.dropout(outputs)
            # |outputs| : (*, H), where H - embedding_dim
            return outputs

        else: raise NotImplementedError

def apply_atcivation(f:str, x:torch.Tensor):
    match f.lower():
        case None | 'none' | 'linear':
            return x
        case 'swiglu':
            return swiglu(x)
        case 'glu':
            return F.glu(x)
        case 'relu':
            return F.relu(x)
        case 'gelu':
            return F.gelu(x)
        case 'hardtanh':
            return F.hardtanh(x)
        case 'hardswish':
            return F.hardswish(x)
        case 'tanh':
            return F.tanh(x)
        case 'sigmoid':
            return F.sigmoid(x)
        case 'selu':
            return F.selu(x)
        case 'elu':
            return F.elu(x)
        case 'leaky_relu':
            return F.leaky_relu(x)
        case 'mish':
            return F.mish(x)
        case _:
            raise Exception("Unexcepted activation")

def apply_seq_batchnorm(x: torch.Tensor, layer: nn.BatchNorm1d, mask: torch.BoolTensor=None):
    # |x| : (B, S, F) torch tensor
    B, S, F = x.size()
    x = x.reshape(B * S, F)
    if mask is not None:
        B, S = mask.size()
        batch_mask = mask.reshape(B*S)
        x[~batch_mask] = layer(x[~batch_mask])

    else:
        x = layer(x)

    x = x.reshape(B, S, F)
    # More slowly
    # x = x.transpose(1, 2)
    # x = layer(x)
    # x = x.transpose(1, 2)
    return x

def layer_norm(x: torch.Tensor):
    return F.layer_norm(x, x.shape[-1])

def get_norm(norm: str) -> nn.BatchNorm1d | nn.LayerNorm | Callable:
    if norm == 'batch':
        norm = nn.BatchNorm1d
    elif norm == 'masked_batch':
        norm = nn.BatchNorm1d
    elif norm == 'layer':
        norm = nn.LayerNorm
    else:
        norm = Callable
    
    return norm

def get_activation(act: str) -> nn.Module:
    match act.lower():
        case None | 'none' | 'linear':
            return Callable
        case 'swiglu':
            return SwiGLU()
        case 'glu':
            return nn.GLU()
        case 'relu':
            return nn.ReLU()
        case 'gelu':
            return nn.GELU()
        case 'hardtanh':
            return nn.Hardtanh()
        case 'hardswish':
            return nn.Hardswish()
        case 'tanh':
            return nn.Tanh()
        case 'sigmoid':
            return nn.Sigmoid()
        case 'selu':
            return nn.SELU()
        case 'elu':
            return nn.ELU()
        case 'leaky_relu':
            return nn.LeakyReLU()
        case 'mish':
            return nn.Mish()
        case _:
            raise Exception("Unexcepted activation")
# ----------------------------------------------------------------------------------------------- #
# Cells
class LSTMFrame(nn.Module):
    def __init__(self, rnn_cells, batch_first=False, dropout=0, bidirectional=False):
        """
        :param rnn_cells: ex) [(cell_0_f, cell_0_b), (cell_1_f, cell_1_b), ..]
        :param dropout:
        :param bidirectional:
        """
        super().__init__()

        if bidirectional:
            assert all(len(pair) == 2 for pair in rnn_cells)
        elif not any(isinstance(rnn_cells[0], iterable)
                     for iterable in [list, tuple, nn.ModuleList]):
            rnn_cells = tuple((cell,) for cell in rnn_cells)

        self.rnn_cells = nn.ModuleList(nn.ModuleList(pair)
                                       for pair in rnn_cells)
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = len(rnn_cells)

        if dropout > 0 and self.num_layers > 1:
            # dropout is applied to output of each layer except the last layer
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = no_dropout

        self.batch_first = batch_first

    def align_sequence(self, seq, lengths, shift_right):
        """
        :param seq: (seq_len, batch_size, *)
        """
        multiplier = 1 if shift_right else -1
        example_seqs = torch.split(seq, 1, dim=1)
        max_length = max(lengths)
        shifted_seqs = [example_seq.roll((max_length - length) * multiplier, dims=0)
                        for example_seq, length in zip(example_seqs, lengths)]
        return torch.cat(shifted_seqs, dim=1)

    def forward(self, input, init_state=None):
        """
        :param input: a tensor(s) of shape (seq_len, batch, input_size)
        :param init_state: (h_0, c_0) where the size of both is (num_layers * num_directions, batch, hidden_size)
        :returns: (output, (h_n, c_n))
        - output: (seq_len, batch, num_directions * hidden_size)
        - h_n: (num_layers * num_directions, batch, hidden_size)
        - c_n: (num_layers * num_directions, batch, hidden_size)
        """

        if isinstance(input, torch.nn.utils.rnn.PackedSequence):
            input_packed = True
            # always batch_first=False --> trick to process input regardless of batch_first option
            input, lengths = pad_packed_sequence(input, batch_first=False)
            if max(lengths) == min(lengths):
                uniform_length = True
            else:
                uniform_length = False
            if isinstance(lengths, torch.Tensor):
                lengths = tuple(lengths.detach().cpu().numpy())
            assert max(lengths) == input.size()[0]
        else:
            input_packed = False
            if self.batch_first:
                input = input.transpose(0, 1)
            lengths = [input.size()[0]] * input.size()[1]
            uniform_length = True

        if not uniform_length:
            indicator = get_indicator(torch.tensor(lengths, device=get_module_device(self)))
            # valid_example_nums = indicator.sum(0)

        if init_state is None:
            # init_state with heterogenous hidden_size
            init_hidden = init_cell = [
                torch.zeros(
                    input.size()[1],
                    self.rnn_cells[layer_idx][direction].hidden_size,
                    device=get_module_device(self))
                for layer_idx in range(self.num_layers)
                for direction in range(self.num_directions)]
            init_state = init_hidden, init_cell

        init_hidden, init_cell = init_state

        last_hidden_list = []
        last_cell_list = []

        layer_output = input

        for layer_idx in range(self.num_layers):
            layer_input = layer_output
            if layer_idx != 0:
                layer_input = self.dropout(layer_input)

            direction_output_list = []

            for direction in range(self.num_directions):
                cell = self.rnn_cells[layer_idx][direction]
                state_idx = layer_idx * self.num_directions + direction
                step_state = (init_hidden[state_idx], init_cell[state_idx])

                direction_output = torch.zeros(
                    layer_input.size()[:2] + (cell.hidden_size,),
                    device=get_module_device(self))  # (seq_len, batch_size, hidden_size)
                step_state_list = []

                if direction == 0:
                    step_input_gen = enumerate(layer_input)
                else:
                    step_input_gen = reversed(list(enumerate(
                        layer_input if uniform_length else
                        self.align_sequence(layer_input, lengths, True))))

                for seq_idx, cell_input in step_input_gen:
                    # if not uniform_length:  # for speed enhancement
                    #     cell_input = cell_input[:valid_example_nums[seq_idx]]
                    #     step_state = step_state[:valid_example_nums[seq_idx]]
                    h, c = step_state = cell(cell_input, step_state)
                    # if uniform_length:
                    direction_output[seq_idx] = h
                    step_state_list.append(step_state)
                    # else:       # for speed enhancement
                    #     direction_output[seq_idx][? :?] = h
                    #     step_state_list.append(step_state)
                if direction == 1 and not uniform_length:
                    direction_output = self.align_sequence(
                        direction_output, lengths, False)

                if uniform_length:
                    # hidden & cell's size = (batch, hidden_size)
                    direction_last_hidden, direction_last_cell = step_state_list[-1]
                else:
                    direction_last_hidden, direction_last_cell = map(
                        lambda x: torch.stack([x[length - 1][example_id]
                                               for example_id, length in enumerate(lengths)], dim=0),
                        zip(*step_state_list))

                direction_output_list.append(direction_output)
                last_hidden_list.append(direction_last_hidden)
                last_cell_list.append(direction_last_cell)

            if self.num_directions == 2:
                layer_output = torch.stack(direction_output_list, dim=2).view(
                    direction_output_list[0].size()[:2] + (-1,))
            else:
                layer_output = direction_output_list[0]

        output = layer_output
        last_hidden_tensor = torch.stack(last_hidden_list, dim=0)
        last_cell_tensor = torch.stack(last_cell_list, dim=0)

        if not uniform_length:
            # the below one line code cleans out trash values beyond the range of lengths.
            # actually, the code is for debugging, so it can be removed to enhance computing speed slightly.
            output = (
                output.transpose(0, 1) * indicator).transpose(0, 1)

        if input_packed:
            output = pack_padded_sequence(output, lengths, batch_first=self.batch_first)
        elif self.batch_first:
            output = output.transpose(0, 1)

        return output, (last_hidden_tensor, last_cell_tensor)

class LayerNormLSTMCell(nn.RNNCellBase):
    """
    It's based on tf.contrib.rnn.LayerNormBasicLSTMCell
    Reference:
    - https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/LayerNormBasicLSTMCell
    - https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/contrib/rnn/python/ops/rnn_cell.py#L1335
    """

    def __init__(self, input_size, hidden_size, dropout=None, layer_norm_enabled=True, cell_ln=None):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fiou_linear = nn.Linear(
            input_size + hidden_size, hidden_size * 4, bias=not layer_norm_enabled)

        if dropout is not None:
            # recurrent dropout is applied
            if isinstance(dropout, nn.Dropout):
                self.dropout = dropout
            elif dropout > 0:
                self.dropout = nn.Dropout(dropout)
            else:
                assert dropout >= 0
                self.dropout = no_dropout
        else:
            self.dropout = no_dropout

        self.layer_norm_enabled = layer_norm_enabled
        if layer_norm_enabled:
            self.fiou_ln_layers = nn.ModuleList(
                nn.LayerNorm(hidden_size) for _ in range(4))
            # self.fiou_ln_layers = nn.ModuleList(
            #     nn.LayerNorm(hidden_size) for _ in range(3))
            # self.fiou_ln_layers.append(
            #     nn.LayerNorm(hidden_size) if u_ln is None else u_ln)
            self.cell_ln = nn.LayerNorm(
                hidden_size) if cell_ln is None else cell_ln
        else:
            assert cell_ln is None
            # assert u_ln is cell_ln is None
            self.fiou_ln_layers = (no_layer_norm,) * 4
            self.cell_ln = no_layer_norm
        # self.reset_parameters()

    # def reset_parameters(self):
    #     stdv = 1.0 / math.sqrt(self.hidden_size)
    #     for weight in self.parameters():
    #         weight.data.uniform_(-stdv, stdv)

    def forward(self, input, state):
        """
        :param input: a tensor of of shape (batch_size, input_size)
        :param state: a pair of a hidden tensor and a cell tensor whose shape is (batch_size, hidden_size).
                      ex. (h_0, c_0)
        :returns: hidden and cell
        """
        hidden_tensor, cell_tensor = state

        fiou_linear = self.fiou_linear(
            torch.cat([input, hidden_tensor], dim=1))
        fiou_linear_tensors = fiou_linear.split(self.hidden_size, dim=1)

        # if self.layer_norm_enabled:
        fiou_linear_tensors = tuple(ln(tensor) for ln, tensor in zip(
            self.fiou_ln_layers, fiou_linear_tensors))

        f, i, o = tuple(torch.sigmoid(tensor)
                        for tensor in fiou_linear_tensors[:3])
        u = self.dropout(torch.tanh(fiou_linear_tensors[3]))

        new_cell = self.cell_ln(i * u + (f * cell_tensor))
        new_h = o * torch.tanh(new_cell)

        return new_h, new_cell

class ZoneoutFrame(nn.Module):
    def __init__(self, forward_cell: LayerNormLSTMCell | nn.LSTMCell | nn.GRUCell | nn.RNNCell, zoneout_prob=0, bidrectional=False, zoneout_dropout=0., zoneout_layernorm=False):
        super(ZoneoutFrame, self).__init__()
        assert bidrectional == False

        self.zoneout_layernorm = zoneout_layernorm
        self.forward_cell = forward_cell
        self.zoneout_prob = zoneout_prob
        self.dropout_rate = zoneout_dropout

        if zoneout_layernorm:
            self.norm = nn.LayerNorm(forward_cell.hidden_size)

        if not isinstance(forward_cell, nn.RNNCellBase):
            raise TypeError("The cell is not a LSTMCell or GRUCell!")
        if isinstance(forward_cell, (nn.LSTMCell, LayerNormLSTMCell)):
            if not isinstance(zoneout_prob, tuple):
                raise TypeError("The LSTM zoneout_prob must be a tuple!")
        elif isinstance(forward_cell, nn.GRUCell):
            if not isinstance(zoneout_prob, float):
                raise TypeError("The GRU zoneout_prob must be a float number!")
        elif isinstance(forward_cell, nn.RNNCell):
            if not isinstance(zoneout_prob, float):
                raise TypeError("The RNN zoneout_prob must be a float number!")

    @property
    def hidden_size(self):
        return self.forward_cell.hidden_size

    @property
    def input_size(self):
        return self.forward_cell.input_size

    def forward(self, forward_input, forward_state):
        forward_new_state = self.forward_cell(forward_input, forward_state)
        if isinstance(self.forward_cell, (nn.LSTMCell, LayerNormLSTMCell)):
            forward_h, forward_c = forward_state
            forward_new_h, forward_new_c = forward_new_state

            zoneout_prob_c, zoneout_prob_h = self.zoneout_prob

            forward_new_h = (1 - zoneout_prob_h) * F.dropout(forward_new_h, p=self.dropout_rate,
                                                                training=self.training) + forward_h
            forward_new_c = (1 - zoneout_prob_c) * F.dropout(forward_new_c, p=self.dropout_rate,
                                                                training=self.training) + forward_c

            if self.zoneout_layernorm:
                forward_new_h = self.norm(forward_new_h)
                forward_new_c = self.norm(forward_new_c)

            forward_new_state = (forward_new_h, forward_new_c)
            forward_output = forward_new_h

        else:
            forward_h = forward_state
            forward_new_h = forward_new_state

            zoneout_prob_h = self.zoneout_prob

            forward_new_h = (1 - zoneout_prob_h) * F.dropout(forward_new_h, p=self.dropout_rate,
                                                                training=self.training) + forward_h

            if self.zoneout_layernorm:
                forward_new_h = self.norm(forward_new_h)

            forward_new_state = forward_new_h
            forward_output = forward_new_h
        return forward_output, forward_new_state

class IRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, 
    rec_activation='relu', weight_init=None, 
    dropout=0, rec_dropout=0, 
    rec_norm=False, layernorm=False, 
    prenorm=False, hiden_after_norm=True,
    ):
        super(IRNNCell, self).__init__()
        #Initialize weights for RNN cell
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rec_activation = rec_activation

        self.hiden_after_norm = hiden_after_norm
        self.layernorm = layernorm
        
        self.rec_norm = rec_norm
        self.prenorm = prenorm

        #Have to initialize to identity matrix
        self.W_x = nn.Parameter(torch.zeros(input_size, hidden_size))
        self.U_h = torch.nn.Parameter(torch.eye(hidden_size))
        self.b = nn.Parameter(torch.zeros(hidden_size))

        if layernorm:
            self.norm = nn.LayerNorm(hidden_size)

        if weight_init is None:
            self.W_x = nn.init.xavier_normal_(self.W_x)
        else:
            self.W_x = weight_init(self.W_x)

        #Set up dropout layer if requested
        self.dropout = nn.Dropout(dropout)
        self.rec_dropout = nn.Dropout(rec_dropout)

        #Initialize recurrent states h_t
        self.hidden_state = None

    def reset(self, batch_size, cuda=True):
        #Reset recurrent states
        w = torch.zeros(batch_size, self.hidden_size)
        if cuda:
            self.hidden_state = torch.autograd.Variable(w).cuda().float()
        else:
            self.hidden_state = torch.autograd.Variable(w).float()

    def forward(self, X_t):
        #Define forward calculations for inference time
        h_t_previous = self.hidden_state

        # ---------------------------------------------------- #
        X_t = self.dropout(X_t)
        h_t_previous = self.rec_dropout(h_t_previous)

        # ---------------------------------------------------- #
        if self.rec_norm:
            x = layer_norm(torch.mm(X_t, self.W_x))
            u = layer_norm(torch.mm(h_t_previous, self.U_h))
            x = layer_norm(x + u + self.b)
        else:
            x = torch.mm(X_t, self.W_x) + torch.mm(h_t_previous, self.U_h) + self.b
        
        # ---------------------------------------------------- #
        if not self.hiden_after_norm:
            self.hidden_state = x

        # ---------------------------------------------------- #
        if self.layernorm:
            if self.prenorm:
                x = self.norm(x)
                x = apply_atcivation(self.rec_activation, x)
            else:
                x = apply_atcivation(self.rec_activation, x)
                x = self.norm(x)
        else:
            x = apply_atcivation(self.rec_activation, x)

        # ---------------------------------------------------- #
        if self.hiden_after_norm:
            self.hidden_state = x
        
        return x

# ----------------------------------------------------------------------------------------------- #
# RNN
class LayerNormLSTM(LSTMFrame):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=False, dropout=0, r_dropout=0, bidirectional=False, layer_norm_enabled=True, **kwargs):

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.dropout = dropout
        self.r_dropout = r_dropout
        self.bidirectional = bidirectional
        self.layer_norm_enabled = layer_norm_enabled

        r_dropout_layer = nn.Dropout(r_dropout)
        rnn_cells = tuple(
            tuple(
                LayerNormLSTMCell(
                    input_size if layer_idx == 0 else hidden_size * (2 if bidirectional else 1),
                    hidden_size,
                    dropout=r_dropout_layer,
                    layer_norm_enabled=layer_norm_enabled)
                for _ in range(2 if bidirectional else 1))
            for layer_idx in range(num_layers))

        super().__init__(rnn_cells=rnn_cells, dropout=dropout,
                         batch_first=batch_first, bidirectional=bidirectional)

class GRU(nn.GRU):
    """This is helps to avoid unnecessary arguments in **kwargs"""
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0, bidirectional=False, **kwargs):
        super().__init__(input_size=input_size, hidden_size=hidden_size, 
        num_layers=num_layers, bias=bias, batch_first=batch_first, 
        dropout=dropout, bidirectional=bidirectional)

class LSTM(nn.LSTM):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0, bidirectional=False, proj_size=0, **kwargs):
        super().__init__(input_size=input_size, hidden_size=hidden_size, 
        num_layers=num_layers, bias=bias, batch_first=batch_first, 
        dropout=dropout, bidirectional=bidirectional, proj_size=proj_size)

class IRNN(nn.Module):
    def __init__(
        self, input_size, hidden_size,
        num_layers=1, dropout=None, rec_dropout=None, 
        activation='linear', rec_activation='relu',
        layernorm=False, prenorm=False, rec_norm=False,
        hiden_after_norm=True,
        **kwargs):
        super(IRNN, self).__init__()
        #Initialize deep RNN neural network

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.activation = activation

        #Initialize individual IRNN cells
        self.rnns = nn.ModuleList()
        for n in range(self.num_layers):
            self.rnns.append(IRNNCell(
                input_size=input_size if n == 0 else hidden_size, 
                hidden_size=hidden_size, rec_activation=rec_activation, 
                dropout=dropout, rec_dropout=rec_dropout, rec_norm=rec_norm,
                layernorm=layernorm, prenorm=prenorm, hiden_after_norm=hiden_after_norm,
            ))

    def reset(self, batch_size, cuda=True):
        #Reset recurrent states for all RNN cells defined
        for rnn in self.rnns:
            rnn: IRNNCell
            rnn.reset(batch_size=batch_size, cuda=cuda)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        # |x| : (batch_size, T, embed_dim)
        outputs = []
        max_time = x.shape[1]

        self.reset(batch_size=x.size(0), cuda=x.is_cuda)
        for rnn in self.rnns:
            for i in range(max_time):
                forward_input = x[:, i, :]
                forward_output = rnn(forward_input)
                forward_output: torch.Tensor
                outputs.append(forward_output.unsqueeze(1))
            x = torch.cat(outputs, dim=1)
            x = apply_atcivation(self.activation, x)
        return x

# ----------------------------------------------------------------------------------------------- #
# Wrappers
class ZoneoutRNN(nn.Module):
    RNNs = {'LSTM': nn.LSTMCell, 'GRU': nn.GRUCell, 'LSTMN': LayerNormLSTMCell}
    def __init__(self, rnn_type, input_size, hidden_size, zoneout_prob, bidrectional=False, num_layers=1, zoneout_dropout=0.25, batch_first=True, **kwargs):
        super(ZoneoutRNN, self).__init__()
        assert bidrectional == False, 'Zoneout does not supports `bidrectional`'
        assert batch_first == True
        assert num_layers == 1

        self.hidden_size = hidden_size
        self.frame = ZoneoutFrame(
            self.RNNs[rnn_type](input_size, hidden_size), 
            zoneout_dropout=zoneout_dropout,
            zoneout_prob=zoneout_prob, 
            bidrectional=bidrectional, 
        )
        
    def forward(self, x:torch.Tensor, forward_state:torch.Tensor|tuple[torch.Tensor, torch.Tensor]):
        if isinstance(forward_state, tuple):
            h, c = forward_state
            if len(h.shape) == 3:
                h = h[0]
            if len(c.shape) == 3:
                c = c[0]
            forward_state = (h, c)

        elif len(forward_state.shape) == 3:
            forward_state = forward_state[0]

        outputs = []
        max_time = x.shape[1]
        for i in range(max_time):
            forward_input = x[:, i, :]
            forward_output, forward_state = self.frame(forward_input, forward_state)
            forward_output: torch.Tensor
            outputs.append(forward_output.unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)
        return outputs, forward_state

class WRNN(nn.Module):
    """Wrapper for RNN"""
    RNNs = {'GRU': GRU, 'LSTM': LSTM, 'LSTMN': LayerNormLSTM, 'IRNN': IRNN}

    def __init__(self, rnn_type, input_size, hidden_size, num_layers, 
        activation='linear', norm='batch', prenorm=False, batch_first=True, 
        dropout=0, dropouti=0, dropouth=0, wdrop=0, 
        zoneout_prob=0, zoneout_layernorm=False, skip_connection:int=0,
        seq_permutation:dict=None, seq_masking:dict=None, **kwargs):
        super(WRNN, self).__init__()
        assert rnn_type in ['LSTM', 'GRU', 'LSTMN', 'IRNN'], 'RNN type is not supported'
        assert norm in ['batch', 'masked_batch', 'layer', 'none'], 'Incorrect norm type'

        if rnn_type == 'IRNN':
            if wdrop > 0:
                raise Exception('IRNN doesnt support `wdrop`')

        self.rnns = []
        for l in range(num_layers):
            if seq_permutation is not None:
                self.rnns.append(SeqPermutation(**seq_permutation))

            if seq_masking is not None:
                self.rnns.append(SeqMasking(**seq_masking))

            if zoneout_prob > 0:
                rnn = ZoneoutRNN(
                    rnn_type, 
                    input_size=input_size if l == 0 else hidden_size, 
                    hidden_size=hidden_size*2 if activation in GATED_ACT else hidden_size, 
                    num_layers=1, 
                    dropout=dropout, 
                    activation='linear',
                    zoneout_prob=zoneout_prob, 
                    zoneout_layernorm=zoneout_layernorm,
                )
                if wdrop > 0:
                    rnn = WeightDrop(rnn, ['weight_hh'], dropout=wdrop)
                self.rnns.append(rnn)
                
            else:
                rnn = self.RNNs[rnn_type](
                    input_size=input_size if l == 0 else hidden_size, 
                    hidden_size=hidden_size*2 if activation in GATED_ACT else hidden_size, 
                    batch_first=batch_first, 
                    activation='linear',
                    dropout=dropout if rnn_type == 'IRNN' else 0,
                    num_layers=1, 
                    **kwargs
                    )
                if wdrop > 0:
                    rnn = WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop)
                self.rnns.append(rnn)

            if norm == 'batch' or norm == 'masked_batch':
                if activation in GATED_ACT and prenorm:
                    self.rnns.append(nn.BatchNorm1d(hidden_size*2))
                else:
                    self.rnns.append(nn.BatchNorm1d(hidden_size))

            if norm == 'layer':
                if activation in GATED_ACT and prenorm:
                    self.rnns.append(nn.LayerNorm(hidden_size*2))
                else:
                    self.rnns.append(nn.LayerNorm(hidden_size))

        self.rnns = nn.ModuleList(self.rnns)
        self.lockdrop = LockedDropout(batch_first=batch_first)

        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size  = hidden_size 
        self.num_layers = num_layers
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.batch_first = batch_first
        self.zoneout_prob = zoneout_prob
        self.skip_connection = skip_connection
        self.activation = activation
        self.prenorm = prenorm
        self.norm = norm

    def reset(self):
        return 

    def init_weights(self):
        return

    def forward(self, input: torch.Tensor, hidden=None, key_padding_mask:torch.Tensor=None, seq_len:torch.Tensor=None) -> dict:
        """ 
        key_padding_mask: padding mask (B, S) shape. `True` value in key_padding_mask indicates that the corresponding key value will be IGNORED
        seq_len: original len of seq (B) shape.
        """
        if hidden is None:
            bsz = input.size(0) if self.batch_first else input.size(1)
            hidden = self.init_hidden(bsz)

        n = 0
        skip_connection = None
        raw_output = self.lockdrop(input, self.dropouti)
        for layer in self.rnns:
            if isinstance(layer, nn.BatchNorm1d):
                if self.prenorm:
                    raw_output = apply_seq_batchnorm(raw_output, layer, mask=key_padding_mask)
                    raw_output = apply_atcivation(self.activation, raw_output)
                else:
                    raw_output = apply_atcivation(self.activation, raw_output)
                    raw_output = apply_seq_batchnorm(raw_output, layer, mask=key_padding_mask)

            elif isinstance(layer, nn.LayerNorm):
                if self.prenorm:
                    raw_output = layer(raw_output)
                    raw_output = apply_atcivation(self.activation, raw_output)
                else:
                    raw_output = apply_atcivation(self.activation, raw_output)
                    raw_output = layer(raw_output)
                    
            elif isinstance(layer, SeqMasking):
                raw_output, key_padding_mask, seq_len = layer(raw_output, key_padding_mask, seq_len)

            elif isinstance(layer, SeqPermutation):
                raw_output = layer(raw_output, key_padding_mask, seq_len)

            else:
                raw_output, new_hidden = layer(raw_output, hidden[n])
                raw_output = self.lockdrop(raw_output, self.dropouth)
                if self.norm == 'linear':
                    raw_output = apply_atcivation(self.activation, raw_output)
                
                # ----------------------------------------------------------- #
                # Residual connection
                if self.skip_connection > 0:
                    if n == 0:
                        skip_connection = raw_output

                    elif (n+1)%self.skip_connection == 0:
                        raw_output = skip_connection = raw_output + skip_connection
                n+=1

        return {
            'output': raw_output,
            'hidden': new_hidden,
            'key_padding_mask': key_padding_mask,
            'seq_len': seq_len,
        }

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type in ['LSTM', 'LSTMN']:
            return [(weight.new(1, bsz, self.hidden_size).zero_(),
                    weight.new(1, bsz, self.hidden_size).zero_())]
        elif self.rnn_type == 'GRU':
            return [weight.new(1, bsz, self.hidden_size).zero_()
                    for l in range(self.num_layers)]
        elif self.rnn_type == 'IRNN':
            return [None for l in range(self.num_layers)]

class RNNAttention(nn.Module):
    """Self-attention for modded RNN

    source: https://github.com/gucci-j/imdb-classification-gru/blob/master/src/model_with_self_attention.py
    """
    def __init__(self, query_dim):
        # assume: query_dim = key/value_dim
        super(RNNAttention, self).__init__()
        self.scale = 1. / math.sqrt(query_dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, key_padding_mask: torch.BoolTensor=None):
        # query == hidden: (batch_size, hidden_dim)
        # key/value == gru_output: (batch_size, sentence_length, hidden_dim)
        # key_padding_mask: (batch_size, sentence_length)
        query = query.unsqueeze(1) # (batch_size, 1, hidden_dim)
        key = key.transpose(1, 2) # (batch_size, hidden_dim, sentence_length)

        # bmm: batch matrix-matrix multiplication
        attention_weight = torch.bmm(query, key) # (batch_size, 1, sentence_length)
        if key_padding_mask is not None:
            assert key_padding_mask.dtype is torch.bool
            attention_weight.masked_fill_(key_padding_mask.unsqueeze(1), -1e9)

        attention_weight = F.softmax(attention_weight.mul_(self.scale), dim=-1) # normalize sentence_length's dimension
        attention_output = torch.bmm(attention_weight, value) # (batch_size, 1, hidden_dim)

        # (batch_size, hidden_dim)
        return attention_output.squeeze(1), attention_weight.squeeze(1)

class RNN(nn.Module):
    def __init__(
        self, input_size:int, embed_dim:int, 
        num_layers:int=1, dropout: float=0,
        attention:bool=False, bidirectional=False,
        rnn_type:Literal['LSTM', 'LSTMN', 'GRU']='GRU', output_hidden:bool=False, 
        zoneout_prob:float=0, zoneout_layernorm:bool=False, **kwargs):
        super(RNN, self).__init__()

        self.rnn_type = rnn_type.upper()
        assert self.rnn_type in ['LSTM', 'LSTMN', 'GRU', 'IRNN']

        if not output_hidden and attention:
            raise Exception("We outputs only `hidden` while use attention, but `output_hidden==False`")

        if self.rnn_type == 'IRNN' and attention:
            raise Exception("`IRNN` dont support attention")

        if (output_hidden or bidirectional) and (self.rnn_type in ['LSTM', 'LSTMN', 'LSTMW']):
            raise Exception("Dont use `output_hidden` or `bidirectional` with `LSTM`")

        if bidirectional and zoneout_prob > 0:
            raise Exception("Zoneout does not supports `biderectional`")

        if zoneout_layernorm and zoneout_prob == 0:
            raise Exception("`zoneout_prob` must be greater than 0 with `zoneout_layernorm==True`")

        kwargs.update({
            'rnn_type': self.rnn_type,
            'input_size': input_size,
            'hidden_size': embed_dim,
            'zoneout_prob': zoneout_prob,
            'num_layers': num_layers,
            'dropout': dropout,
            'bidirectional': bidirectional,
            'zoneout_layernorm': zoneout_layernorm,
            'batch_first': True
        })
        self.output_hidden = output_hidden
        self.bidirectional = bidirectional
        self.attention = RNNAttention(2*embed_dim if bidirectional else embed_dim) if attention else False
        self.rnn = WRNN(**kwargs)
        
    def forward(self, x: torch.Tensor, key_padding_mask:torch.Tensor, seq_len:torch.Tensor):
        rnn_output = self.rnn(x, hidden=None, key_padding_mask=key_padding_mask, seq_len=seq_len)
        output = rnn_output['output']
        hidden = rnn_output['hidden']

        if self.rnn_type in ['LSTM', 'LSTMN']:
            hidden = hidden[1]

        output: torch.Tensor # (batch_size, sentence_length, embed_dim (*2 if bidirectional)) 
        hidden: torch.Tensor # (num_layers (*2 if bidirectional), batch_size, embed_dim) if GRU
        # ordered: [f_layer_0, b_layer_0, ...f_layer_n, b_layer n]

        # concat the final output of forward direction and backward direction
        if self.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
            # | hidden | : (batch_size, embed_dim * 2)
            rnn_output['hidden'] = hidden
            
        else:
            hidden = hidden[-1,:,:]
            # | hidden | : (batch_size, embed_dim)
            rnn_output['hidden'] = hidden

        if self.attention:
            rescaled_hidden, attention_weight = self.attention(query=hidden, key=output, value=output)
            rnn_output['hidden'] = rescaled_hidden
            rnn_output['attention_weight'] = attention_weight

            return rescaled_hidden, rnn_output

        return (hidden, rnn_output) if self.output_hidden else (output, rnn_output)

# ----------------------------------------------------------------------------------------------- #
# Augmentation
class SeqPermutation(nn.Module):
    """Shuffle random rows in Tensor"""
    def __init__(self, shuffles_num, max_step, p):
        super(SeqPermutation, self).__init__()
        self.shuffles_num = shuffles_num
        self.max_step = max_step
        self.p = p

    def forward(self, x: torch.Tensor, key_padding_mask:torch.Tensor=None, seq_len:torch.Tensor=None, p=None):
        # |x| : (batch_size, seq_len, ...)
        # shuffle by seq_len
        if p is None:
            p = self.p

        if self.p == 0 or not self.training:
            return x

        new_x = []
        max_len = x.size(1) - 1
        for seq_x in x:
            if random.random() < p:
                for _ in range(self.shuffles_num):
                    base_pos = random.randint(0, max_len)
                    start_pos = random.randint(base_pos - self.max_step//2, base_pos)
                    end_pos = random.randint(base_pos, base_pos + self.max_step//2)
                    
                    add_to_end_pos = 0
                    if start_pos < 0:
                        add_to_end_pos = 0 - start_pos
                        start_pos = 0
                        
                    add_to_start_pos = 0
                    if end_pos > max_len:
                        add_to_start_pos = max_len - end_pos
                        end_pos = max_len
                        
                    start_pos += add_to_start_pos
                    end_pos += add_to_end_pos
                    
                    seq_x = self.permute(seq_x, start_pos, end_pos)
            new_x.append(seq_x)

        new_x = torch.stack(new_x)
        return new_x

    def permute(self, x, n1, n2):
        idx = np.arange(0, len(x))
        idx[n1], idx[n2] = idx[n2], idx[n1]
        return x[idx]

class SeqMasking(nn.Module):
    """Mask random rows in Tensor and put this zero row in start of seq"""
    def __init__(self, p):
        super(SeqMasking, self).__init__()
        self.p = p

    def forward(self, x: torch.Tensor, key_padding_mask:torch.Tensor=None, seq_len:torch.Tensor=None, p=None):
        # |x| : (batch_size, seq_len, embed_dim)
        if p is None:
            p = self.p

        if p == 0 or not self.training:
            return x, key_padding_mask, seq_len

        new_x = []
        mask = torch.rand(x.size(0), x.size(1)) > p
        for seq_m, seq_x in zip(mask, x):

            batch_x = []
            for t_m, t_x in zip(seq_m, seq_x):
                if t_m: batch_x.append(t_x)
                    
            padded_arr = torch.zeros(x.size(1), x.size(2), dtype=x.dtype, device=x.device)
            batch_x = torch.vstack(batch_x)

            l = len(batch_x)
            if l > 0:
                padded_arr[-l:] = batch_x
            new_x.append(padded_arr)
            
        new_x = torch.stack(new_x)
        return new_x, key_padding_mask, seq_len

# ----------------------------------------------------------------------------------------------- #
# Transformer stuff
class PositionWiseFeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout, bias, wdropoout=0., bdropoout=0., activation: str='gelu', **kwargs):
        super(PositionWiseFeedForwardNetwork, self).__init__()
        assert ff_dim%2 == 0

        self.linear1 = SparseConnectedLayer(embed_dim, ff_dim*2 if activation in GATED_ACT else ff_dim, bias=bias, pw=wdropoout, pb=bdropoout)
        self.linear2 = SparseConnectedLayer(ff_dim, embed_dim, bias=bias, pw=wdropoout, pb=bdropoout)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        nn.init.normal_(self.linear1.weight, std=0.02)
        nn.init.normal_(self.linear2.weight, std=0.02)

    def forward(self, inputs):
        # |inputs| : (batch_size, seq_len, d_model)

        outputs = apply_atcivation(self.activation, self.linear1(inputs))
        outputs = self.dropout(outputs)
        # |outputs| : (batch_size, seq_len, d_ff)
        
        outputs = self.linear2(outputs)
        # |outputs| : (batch_size, seq_len, d_model)

        return outputs

class AttentionBase(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.15, bias=True, layer_norm='post', activation: str='gelu', **kwargs):
        super(AttentionBase, self).__init__()
        assert layer_norm in ['pre', 'post'], f'Unexcepted layer norm mode: {layer_norm}'

        self.layer_norm = layer_norm
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
            bias=bias,
        )
        self.ffn = PositionWiseFeedForwardNetwork(
            embed_dim=embed_dim,
            ff_dim=ff_dim,
            activation=activation,
            dropout=dropout,
            bias=bias,
            **kwargs
        )

class CrossAttention(AttentionBase):
    def forward(self, x1, x2, key_padding_mask=None, attn_mask=None):
        if self.layer_norm == 'pre':
            # ---------------------- #
            x1_ = self.layernorm1(x1)
            x2_ = self.layernorm1(x2)
            
            x1_attn_output, x1_attn_weights = self.attn(
                query=x1_, key=x2_, value=x2_, 
                key_padding_mask=key_padding_mask, attn_mask=attn_mask
            )
            x2_attn_output, x2_attn_weights = self.attn(
                query=x2_, key=x1_, value=x1_, 
                key_padding_mask=key_padding_mask, attn_mask=attn_mask
            )
            
            x1 = x1 + x1_attn_output
            x2 = x2 + x2_attn_output
            
            # ---------------------- #
            x1_ = self.layernorm2(x1)
            x2_ = self.layernorm2(x2)
            
            x1_fnn_output = self.ffn(x1_)
            x2_fnn_output = self.ffn(x2_)
            
            x1 = x1 + x1_fnn_output
            x2 = x2 + x2_fnn_output
            
            return x1, x2
        
        elif self.layer_norm == 'post':
            x1_attn_output, x1_attn_weights = self.attn(
                query=x1, key=x2, value=x2, 
                key_padding_mask=key_padding_mask, attn_mask=attn_mask
            )
            x2_attn_output, x2_attn_weights = self.attn(
                query=x2, key=x1, value=x1, 
                key_padding_mask=key_padding_mask, attn_mask=attn_mask
            )
            
            x1 = self.layernorm1(x1 + x1_attn_output)
            x2 = self.layernorm1(x2 + x2_attn_output)
            
            x1 = self.layernorm2(x1 + self.ffn(x1))
            x2 = self.layernorm2(x2 + self.ffn(x2))
            
            return x1, x2
        
        else: raise Exception

class SelfAttention(AttentionBase):
    def forward(self, inputs, key_padding_mask=None, attn_mask=None):
        if self.layer_norm == 'pre':
            inputs_ = self.layernorm1(inputs)
            
            attn_output, attn_weights = self.attn(
                query=inputs_, key=inputs_, value=inputs_, 
                key_padding_mask=key_padding_mask, attn_mask=attn_mask
            )

            x = inputs + attn_output
            x = x + self.ffn(self.layernorm2(x))
            return x
        
        elif self.layer_norm == 'post':
            attn_output, attn_weights = self.attn(
                query=inputs, key=inputs, value=inputs, 
                key_padding_mask=key_padding_mask, attn_mask=attn_mask
            )

            x = self.layernorm1(inputs + attn_output)
            x = self.layernorm2(x + self.ffn(x))
            return x
        
        else: raise Exception
