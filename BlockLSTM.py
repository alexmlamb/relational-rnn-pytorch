'''
Goal: an LSTM where the weight matrices have a block structure so that information flow is constrained

Data is assumed to come in [block1, block2, ..., block_n].  



'''
import time
import torch
import torch.nn as nn
from torch.autograd import grad, Variable

'''
Given an N x N matrix, and a grouping of size, set all elements off the block diagonal to 0.0
'''
def zero_matrix_elements(matrix, k):
    assert matrix.shape[0] % k == 0
    assert matrix.shape[1] % k == 0
    g1 = matrix.shape[0] // k
    g2 = matrix.shape[1] // k
    new_mat = torch.zeros_like(matrix)
    for b in range(0,k):
        new_mat[b*g1 : (b+1)*g1, b*g2 : (b+1)*g2] += matrix[b*g1 : (b+1)*g1, b*g2 : (b+1)*g2]

    matrix *= 0.0
    matrix += new_mat


class BlockLSTM(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ninp, nhid, k):
        super(BlockLSTM, self).__init__()

        assert ninp % k == 0
        assert nhid % k == 0

        self.k = k
        self.lstm = nn.LSTMCell(ninp, nhid)
        self.nhid = nhid
        self.ninp = ninp

    def blockify_params(self):
        pl = self.lstm.parameters()

        for p in pl:
            p = p.data
            if p.shape == torch.Size([self.nhid*4]):
                pass
                '''biases, don't need to change anything here'''
            if p.shape == torch.Size([self.nhid*4, self.nhid]) or p.shape == torch.Size([self.nhid*4, self.ninp]):
                for e in range(0,4):
                    zero_matrix_elements(p[self.nhid*e : self.nhid*(e+1)], k=self.k)

    def forward(self, input, h, c):

        #t0 = time.time()
        #self.blockify_params()
        #print(time.time() - t0, 'time to blockify')

        #t0 = time.time()
        hnext, cnext = self.lstm(input, (h, c))
        #print(time.time() - t0, 'time to rnn pass')

        return hnext, cnext


if __name__ == "__main__":

    Blocks = BlockLSTM(2, 6, k=2)
    opt = torch.optim.Adam(Blocks.parameters())

    pl = Blocks.lstm.parameters()

    inp = Variable(torch.randn(1,2), requires_grad=True)
    h = Variable(torch.randn(1,3*2), requires_grad=True)
    c = Variable(torch.randn(1,3*2), requires_grad=True)

    h2, c2 = Blocks(inp,h,c)

    L = c2[0,5]

    print(grad(L, h)[0])

    #L.backward()
    #opt.step()
    #opt.zero_grad()

    raise Exception('done')

    pl = Blocks.lstm.parameters()
    for p in pl:
        #print(p.shape)
        #print(torch.Size([Blocks.nhid*4]))
        if p.shape == torch.Size([Blocks.nhid*4]):
            print(p.shape, 'a')
            #print(p)
            '''biases, don't need to change anything here'''
        if p.shape == torch.Size([Blocks.nhid*4, Blocks.nhid]) or p.shape == torch.Size([Blocks.nhid*4, Blocks.ninp]):
            print(p.shape, 'b')
            for e in range(0,4):
                print(p[Blocks.nhid*e : Blocks.nhid*(e+1)])




