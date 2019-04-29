import torch.nn as nn
import torch
from attention import MultiHeadAttention
from layer_conn_attention import LayerConnAttention
from BlockLSTM import BlockLSTM

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False, use_cudnn_version=True,
                 use_adaptive_softmax=False, cutoffs=None, num_blocks=6):
        super(RNNModel, self).__init__()
        self.use_cudnn_version = use_cudnn_version
        self.drop = nn.Dropout(dropout)
        print('number of inputs, ninp', ninp)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.num_blocks = num_blocks
        self.nhid = nhid
        self.block_size = nhid // self.num_blocks
        print('number of blocks', self.num_blocks)

        self.sigmoid = nn.Sigmoid()
        if use_cudnn_version:
            raise Exception('not defined')
            if rnn_type in ['LSTM', 'GRU']:
                self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
            else:
                try:
                    nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
                except KeyError:
                    raise ValueError("""An invalid option for `--model` was supplied,
                                     options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
                self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        else:
            #tried reducing size
            self.mha = MultiHeadAttention(n_head=4, d_model=self.block_size, d_k=16, d_v=16, num_blocks=num_blocks)
            self.fan_in = 2
            self.layer_conn_att = LayerConnAttention(n_head=4*self.fan_in, d_model=self.block_size, d_k=self.block_size, d_v=self.block_size, d_out=nhid*self.fan_in)

            if rnn_type in ['LSTM', 'GRU']:
                rnn_type = str(rnn_type) + 'Cell'
                rnn_modulelist = []
                blockrnn_lst = []
                dropout_lst = []
                for i in range(nlayers):
                    blocklst = []
                    for block in range(num_blocks):
                        if i == 0:
                            self.block_lstm = BlockLSTM(ninp*num_blocks, nhid, k=num_blocks)
                            blocklst.append(getattr(nn, rnn_type)(ninp, nhid//num_blocks))
                        else:
                            blocklst.append(getattr(nn, rnn_type)(self.block_size, nhid//num_blocks))
                    blocklst = nn.ModuleList(blocklst)
                    rnn_modulelist.append(blocklst)
                    dropout_lst.append(nn.Dropout(dropout))

                print('number of layers', nlayers)
                print('number of modules', len(rnn_modulelist))
                self.rnn = nn.ModuleList(rnn_modulelist)
                self.dropout_lst = nn.ModuleList(dropout_lst)
            else:
                raise ValueError("non-cudnn version of (RNNCell) is not implemented. use LSTM or GRU instead")

        if not use_adaptive_softmax:
            self.use_adaptive_softmax = use_adaptive_softmax
            self.decoder = nn.Linear(nhid, ntoken)
            # Optionally tie weights as in:
            # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
            # https://arxiv.org/abs/1608.05859
            # and
            # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
            # https://arxiv.org/abs/1611.01462
            if tie_weights:
                print('tying weights!')
                if nhid != ninp:
                    raise ValueError('When using the tied flag, nhid must be equal to emsize')
                self.decoder.weight = self.encoder.weight
        else:
            # simple linear layer of nhid output size. used for adaptive softmax after
            # directly applying softmax at the hidden states is a bad idea
            self.decoder_adaptive = nn.Linear(nhid, nhid)
            self.use_adaptive_softmax = use_adaptive_softmax
            self.cutoffs = cutoffs
            if tie_weights:
                print("Warning: if using adaptive softmax, tie_weights cannot be applied. Ignored.")

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        if not self.use_adaptive_softmax:
            self.decoder.bias.data.zero_()
            self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):

        extra_loss = 0.0

        emb = self.drop(self.encoder(input))
        if self.use_cudnn_version:
            output, hidden = self.rnn(emb, hidden)
        else:
            # for loop implementation with RNNCell
            layer_input = emb
            new_hidden = [[], []]
            for idx_layer in range(0, self.nlayers):
                #print('idx layer', idx_layer)
                output = []
                self.block_lstm.blockify_params()
                #print('hidden shape', hidden[0].shape)
                hx, cx = hidden[0][idx_layer], hidden[1][idx_layer]
                for idx_step in range(input.shape[0]):
                    hxl = []
                    cxl = []

                    if idx_layer == 0:
                        inp_use = layer_input[idx_step]
                        inp_use = inp_use.repeat(1,self.num_blocks)
                        #print('inp use shape', inp_use.shape)
                    else:
                        #inp_use = layer_input[idx_step]#[:,block*self.block_size : (block+1)*self.block_size]
                        inp_use = layer_input[idx_step]
                        inp_use = inp_use.reshape((inp_use.shape[0], self.num_blocks, self.block_size))
                        inp_use,_,_ = self.layer_conn_att(hx.reshape((hx.shape[0], self.num_blocks, self.block_size)), inp_use, inp_use)
                        #inp_use,_,_ = self.layer_conn_att(inp_use, inp_use, inp_use)
                        #print('inp use shape', inp_use.shape)
                        inp_use = inp_use.reshape((inp_use.shape[0], self.nhid*self.num_blocks*self.fan_in))

                    #print('layer, inp shape', idx_layer, inp_use.shape)

                    hx_new, cx_new = self.block_lstm(inp_use, hx, cx)

                    hx = hx_new
                    cx = cx_new

                    #print(hxl[0].sum(), hx[:,0:100].sum(), hx.reshape((64,3,100))[:,0,:].sum(), 'should be same')
                    #print('hx shape cx shape', hx.shape, cx.shape)
                    
                    #TODO: attention turned off if following lines commented
                    hx = hx.reshape((hx.shape[0], self.num_blocks, self.block_size))
                    hx,attn_out,extra_loss_att = self.mha(hx,hx,hx)
                    hx = hx.reshape((hx.shape[0], self.nhid))
                    extra_loss += extra_loss_att

                    output.append(hx)
                output = torch.stack(output)
                if idx_layer + 1 < self.nlayers:
                    output = self.dropout_lst[idx_layer](output)
                layer_input = output
                new_hidden[0].append(hx)
                new_hidden[1].append(cx)
            new_hidden[0] = torch.stack(new_hidden[0])
            new_hidden[1] = torch.stack(new_hidden[1])
            hidden = tuple(new_hidden)

        output = self.drop(output)

        

        if not self.use_adaptive_softmax:
            #print('not use adaptive softmax, size going into decoder', output.size())
            decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
            return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden, extra_loss
        else:
            decoded = self.decoder_adaptive(output.view(output.size(0) * output.size(1), output.size(2)))
            return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden, extra_loss

    def init_hidden(self, bsz):
        weight = next(self.block_lstm.parameters())
        if self.rnn_type == 'LSTM' or self.rnn_type == 'LSTMCell':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)
