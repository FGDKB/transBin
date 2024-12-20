import math
import torch
import numpy as np
import torch.nn as nn
import copy
import torch.optim as optim
import torch.utils.data as Data
import transBin.transBintools as _transBintools
from math import log

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MyDataSet(Data.Dataset):
    def __init__(self, enc_inputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        # self.dec_inputs = dec_inputs
        self.dec_outputs = enc_inputs

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_outputs[idx]

def make_dataloader(rpkm, tnf, batchsize=64, destroy=False, cuda=False):
    """Create a DataLoader and a contig mask from RPKM and TNF.

    The dataloader is an object feeding minibatches of contigs to the VAE.
    The data are normalized versions of the input datasets, with zero-contigs,
    i.e. contigs where a row in either TNF or RPKM are all zeros, removed.
    The mask is a boolean mask designating which contigs have been kept.

    Inputs:
        rpkm: RPKM matrix (N_contigs x N_samples)
        tnf: TNF matrix (N_contigs x N_TNF)
        batchsize: Starting size of minibatches for dataloader
        destroy: Mutate rpkm and tnf array in-place instead of making a copy.
        cuda: Pagelock memory of dataloader (use when using GPU acceleration)

    Outputs:
        DataLoader: An object feeding data to the VAE
        mask: A boolean mask of which contigs are kept
    """

    if not isinstance(rpkm, np.ndarray) or not isinstance(tnf, np.ndarray):
        raise ValueError('TNF and RPKM must be Numpy arrays')

    if batchsize < 1:
        raise ValueError('Minimum batchsize of 1, not {}'.format(batchsize))

    if len(rpkm) != len(tnf):
        raise ValueError('Lengths of RPKM and TNF must be the same')

    if not (rpkm.dtype == tnf.dtype == np.float32):
        raise ValueError('TNF and RPKM must be Numpy arrays of dtype float32')

    mask = tnf.sum(axis=1) != 0

    # If multiple samples, also include nonzero depth as requirement for accept
    # of sequences
    if rpkm.shape[1] > 1:
        depthssum = rpkm.sum(axis=1)
        mask &= depthssum != 0
        depthssum = depthssum[mask]

    if mask.sum() < batchsize:
        raise ValueError('Fewer sequences left after filtering than the batch size.')

    if destroy:
        rpkm = _transBintools.numpy_inplace_maskarray(rpkm, mask)
        tnf = _transBintools.numpy_inplace_maskarray(tnf, mask)
    else:
        # The astype operation does not copy due to "copy=False", but the masking
        # operation does.
        rpkm = rpkm[mask].astype(np.float32, copy=False)
        tnf = tnf[mask].astype(np.float32, copy=False)

    # If multiple samples, normalize to sum to 1, else zscore normalize
    if rpkm.shape[1] > 1:
        rpkm /= depthssum.reshape((-1, 1))
    else:
        _transBintools.zscore(rpkm, axis=0, inplace=True)

    # Normalize arrays and create the Tensors (the tensors share the underlying memory)
    # of the Numpy arrays
    _transBintools.zscore(tnf, axis=0, inplace=True)
    depthstensor = torch.from_numpy(rpkm)
    tnftensor = torch.from_numpy(tnf)
    sumtensor= torch.cat((depthstensor,tnftensor),-1)

    # Create dataloader
    n_workers = 8
    dataset = MyDataSet(sumtensor)
    dataloader = Data.DataLoader(dataset=dataset, batch_size=batchsize, drop_last=True,
                             shuffle=True, num_workers=n_workers, pin_memory=cuda)

    return dataloader, mask

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k=64):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, dropout=None):
        '''
        Q: [batch_size, n_heads, d_k]
        K: [batch_size, n_heads, d_k]
        V: [batch_size, n_heads,d_v]
        attn_mask: [batch_size, n_heads, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        attn = nn.Softmax(dim=-1)(scores)
        attn = dropout(attn)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model = 512, d_k = 64, nheads = 8, dropout=0.2):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.nheads = nheads
        self.d_v = d_k
        self.dropout = nn.Dropout(p=dropout)
        self.norm = LayerNorm(d_model)
        self.W_Q = nn.Linear(self.d_model, self.d_k * self.nheads, bias=False)
        self.W_K = nn.Linear(self.d_model, self.d_k * self.nheads, bias=False)
        self.W_V = nn.Linear(self.d_model, self.d_v * self.nheads, bias=False)
        self.fc = nn.Linear(self.nheads * self.d_v, self.d_model, bias=False)

    def forward(self, input_Q, input_K, input_V):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        input_Q = self.norm(input_Q)
        input_K = self.norm(input_K)
        input_V = self.norm(input_V)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.nheads, self.d_k).transpose(1, 2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, self.nheads, self.d_k).transpose(1, 2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, self.nheads, self.d_v).transpose(1,
                                                                           2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention(d_k=self.d_k)(Q, K, V, dropout=self.dropout)
        context = context.transpose(1, 2).reshape(batch_size,self.nheads * self.d_v)  # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, d_model]
        return residual+self.dropout(output), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model = 512, d_ff = 2048, dropout = 0.2):
        super(PoswiseFeedForwardNet, self).__init__()
        self.d_model = d_model
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
        self.dropout = nn.Dropout(p=dropout)
        self.norm = LayerNorm(self.d_model)

    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(self.norm(inputs))
        return residual+self.dropout(output)  # [batch_size, seq_len, d_model]

class EncoderLayer(nn.Module):
    def __init__(self, d_model = 512, d_ff = 2048, d_k = 64, nheads = 8, dropout = 0.2):
        super(EncoderLayer, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.d_k = d_k
        self.nheads = nheads
        self.dropout = dropout
        self.enc_self_attn = MultiHeadAttention(d_model = self.d_model, d_k = self.d_k, nheads = self.nheads, dropout=self.dropout)
        self.pos_ffn = PoswiseFeedForwardNet(d_model = self.d_model, d_ff = self.d_ff, dropout = self.dropout)

    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        '''
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs)  # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn


class DecoderLayer(nn.Module):
    def __init__(self, d_model = 512, d_ff = 2048, d_k = 64, nheads = 8, dropout = 0.2):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.d_k = d_k
        self.nheads = nheads
        self.dropout = dropout
        self.dec_self_attn = MultiHeadAttention(d_model = self.d_model, d_k = self.d_k, nheads = self.nheads, dropout=self.dropout)
        self.pos_ffn = PoswiseFeedForwardNet(d_model = self.d_model, d_ff = self.d_ff, dropout = self.dropout)

    def forward(self, dec_inputs):
        '''
        dec_inputs: [batch_size, tgt_len, d_model]
        enc_outputs: [batch_size, src_len, d_model]
        dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        '''
        # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        dec_outputs, attn1 = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs)
        # dec_outputs: [batch_size, tgt_len, d_model], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
        dec_outputs = self.pos_ffn(dec_outputs)  # [batch_size, tgt_len, d_model]
        return dec_outputs, attn1


class Encoder(nn.Module):
    def __init__(self,nsamples, nlayers = 6, d_model = 512, d_ff = 2048, d_k = 64, nheads = 8, dropout = 0.2):
        super(Encoder, self).__init__()
        self.nsamples = nsamples
        self.dropout = dropout
        self.d_model = d_model
        self.d_ff = d_ff
        self.d_k = d_k
        self.nlayers = nlayers
        self.nheads = nheads
        self.lin = nn.Linear(self.nsamples+103, d_model)
        self.norm = LayerNorm(d_model)
        self.layers = clones(EncoderLayer(d_model = self.d_model, d_ff = self.d_ff, d_k = self.d_k, nheads = self.nheads, dropout = self.dropout), self.nlayers)

    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, src_len]
        '''
        enc_outputs = self.lin(enc_inputs)
          # [batch_size, src_len, d_model]
          # [batch_size, src_len, src_len]
        # enc_outputs = enc_inputs
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs)
            enc_self_attns.append(enc_self_attn)
        return self.norm(enc_outputs), enc_self_attns


class Decoder(nn.Module):
    def __init__(self, nlayers = 6, nlatent = 64, d_model = 512, d_ff = 2048, d_k = 64, nheads = 8, dropout = 0.2):
        super(Decoder, self).__init__()
        self.dropout = dropout
        self.d_model = d_model
        self.d_ff = d_ff
        self.d_k = d_k
        self.nlayers = nlayers
        self.nheads = nheads
        self.nlatent = nlatent
        self.lin = nn.Linear(self.nlatent, d_model)
        self.norm = LayerNorm(d_model)
        self.layers = clones(DecoderLayer(d_model = self.d_model, d_ff = self.d_ff, d_k = self.d_k, nheads = self.nheads, dropout = self.dropout), self.nlayers)

    def forward(self, dec_inputs):
        dec_outputs = self.lin(dec_inputs)
        # dec_outputs = dec_inputs
        dec_enc_attns = []
        for layer in self.layers:
            # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            dec_outputs, dec_enc_attn = layer(dec_outputs)
            dec_enc_attns.append(dec_enc_attn)
        return self.norm(dec_outputs), dec_enc_attns


class Transformer(nn.Module):
    def __init__(self,nsamples, nlatent=32, d_model = 512, d_ff = 2048, d_k = 64, nlayers = 5, nheads=8, dropout=0.2, alpha=None, beta=200, cuda=False):
        if nlatent < 1:
            raise ValueError('Minimum 1 latent neuron, not {}'.format(nlatent))

        if nsamples < 1:
            raise ValueError('nsamples must be > 0, not {}'.format(nsamples))

        # If only 1 sample, we weigh alpha and nhiddens differently
        if alpha is None:
            alpha = 0.15 if nsamples > 1 else 0.50

        if dropout is None:
            dropout = 0.2 if nsamples > 1 else 0.0

        if beta <= 0:
            raise ValueError('beta must be > 0, not {}'.format(beta))

        if not (0 < alpha < 1):
            raise ValueError('alpha must be 0 < alpha < 1, not {}'.format(alpha))

        if not (0 <= dropout < 1):
            raise ValueError('dropout must be 0 <= dropout < 1, not {}'.format(dropout))

        super(Transformer, self).__init__()
        self.nsamples = nsamples
        self.dropout = dropout
        self.d_model = d_model
        self.d_ff = d_ff
        self.d_k = d_k
        self.nlayers = nlayers
        self.nheads = nheads
        self.nlatent = nlatent
        self.alpha = alpha
        self.beta = beta
        self.softplus = nn.Softplus()
        self.usecuda = cuda

        self.encoder = Encoder(self.nsamples,nlayers = self.nlayers, d_model = self.d_model, d_ff = self.d_ff, d_k = self.d_k, nheads = self.nheads, dropout = self.dropout)
        self.decoder = Decoder(nlayers = self.nlayers, nlatent = self.nlatent, d_model = self.d_model, d_ff = self.d_ff, d_k = self.d_k, nheads = self.nheads, dropout = self.dropout)
        self.projection = nn.Linear(d_model, nsamples+103, bias=False)

        # Latent layers
        self.mu = nn.Linear(d_model, self.nlatent)
        self.logsigma = nn.Linear(d_model, self.nlatent)

        if cuda:
            self.cuda()

    def reparameterize(self, mu, logsigma):
        epsilon = torch.randn(mu.size(0), mu.size(1))

        if self.usecuda:
            epsilon = epsilon.cuda()

        epsilon.requires_grad = True

        # See comment above regarding softplus
        latent = mu + epsilon * torch.exp(logsigma/2)

        return latent

    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, src_len]
        dec_inputs: [batch_size, tgt_len]
        '''
        # tensor to store decoder outputs
        # outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)

        # enc_outputs: [batch_size, src_len, d_model], enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        mu = self.mu(enc_outputs)
        logsigma = self.softplus(self.logsigma(enc_outputs))
        latent = self.reparameterize(mu, logsigma)
        # dec_outpus: [batch_size, tgt_len, d_model], dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [n_layers, batch_size, tgt_len, src_len]
        dec_outputs, dec_enc_attns = self.decoder(latent)
        dec_logits = self.projection(dec_outputs)  # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        return dec_logits,mu,logsigma

    def calc_loss(self, dec_inputs, dec_logits, mu, logsigma):
        # If multiple samples, use cross entropy, else use SSE for abundance
        sse = (dec_inputs - dec_logits).pow(2).sum(dim=1).mean()
        kld = -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).sum(dim=1).mean()
        sse_weight = self.alpha / 103
        kld_weight = 1 / (self.nlatent * self.beta)
        loss = sse * sse_weight + kld * kld_weight
        return loss, sse, kld

    def trainepoch(self, data_loader, epoch, optimizer, logfile):
        self.train()

        epoch_loss = 0
        epoch_kldloss = 0
        epoch_sseloss = 0

        for enc,dec in data_loader:
            enc.requires_grad = True
            dec.requires_grad = True

            if self.usecuda:
                enc = enc.cuda()
                dec = dec.cuda()

            optimizer.zero_grad()

            dec_logits,mu,logsigma = self(enc)

            loss, sse, kld = self.calc_loss(dec, dec_logits, mu, logsigma)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.data.item()
            epoch_kldloss += kld.data.item()
            epoch_sseloss += sse.data.item()

        if logfile is not None:
            print('\tEpoch: {}\tLoss: {:.6f}\tSSE: {:.6f}\tKLD: {:.4f}\tBatchsize: {}'.format(
                epoch + 1,
                epoch_loss / len(data_loader),
                epoch_sseloss / len(data_loader),
                epoch_kldloss / len(data_loader),
                data_loader.batch_size,
            ), file=logfile)

            logfile.flush()

        return None


    def trainmodel(self, dataloader, nepochs=200, lrate=1e-3, logfile=None, modelfile=None):
        """Train the autoencoder from depths array and tnf array.

        Inputs:
            dataloader: DataLoader made by make_dataloader
            nepochs: Train for this many epochs before encoding [500]
            lrate: Starting learning rate for the optimizer [0.001]
            batchsteps: None or double batchsize at these epochs [25, 75, 150, 300]
            logfile: Print status updates to this file if not None [None]
            modelfile: Save models to this file if not None [None]

        Output: None
        """

        if lrate < 0:
            raise ValueError('Learning rate must be positive, not {}'.format(lrate))

        if nepochs < 1:
            raise ValueError('Minimum 1 epoch, not {}'.format(nepochs))


        # Get number of features
        ncontigs, nsamples = dataloader.dataset.enc_inputs.shape
        optimizer = optim.Adam(self.parameters(), lr=lrate)

        if logfile is not None:
            print('\tNetwork properties:', file=logfile)
            print('\tCUDA:', self.usecuda, file=logfile)
            print('\tAlpha:', self.alpha, file=logfile)
            print('\tBeta:', self.beta, file=logfile)
            print('\tDropout:', self.dropout, file=logfile)
            print('\tN latent:', self.nlatent, file=logfile)
            print('\n\tTraining properties:', file=logfile)
            print('\tN epochs:', nepochs, file=logfile)
            print('\tStarting batch size:', dataloader.batch_size, file=logfile)
            print('\tLearning rate:', lrate, file=logfile)
            print('\tN sequences:', ncontigs, file=logfile)
            print('\tN samples:', nsamples-103, file=logfile, end='\n\n')

        # Train
        for epoch in range(nepochs):
            self.trainepoch(dataloader, epoch, optimizer, logfile)

        # Save weights - Lord forgive me, for I have sinned when catching all exceptions
        if modelfile is not None:
            try:
                self.save(modelfile)
            except:
                pass

        return None

    def encode(self, data_loader):
        """Encode a data loader to a latent representation with VAE

        Input: data_loader: As generated by train_vae

        Output: A (n_contigs x n_latent) Numpy array of latent repr.
        """

        self.eval()

        new_data_loader = Data.DataLoader(dataset=data_loader.dataset,
                                      batch_size=data_loader.batch_size,
                                      shuffle=False,
                                      drop_last=False,
                                      num_workers=1,
                                      pin_memory=data_loader.pin_memory)

        array1 = data_loader.dataset.enc_inputs
        length = len(array1)

        # We make a Numpy array instead of a Torch array because, if we create
        # a Torch array, then convert it to Numpy, Numpy will believe it doesn't
        # own the memory block, and array resizes will not be permitted.
        latent = np.empty((length, self.nlatent), dtype=np.float32)

        row = 0
        with torch.no_grad():
            for enc,dec in new_data_loader:
                # Move input to GPU if requested
                if self.usecuda:
                    enc = enc.cuda()

                # Evaluate
                dec_logits, mu, logsigma = self(enc)

                if self.usecuda:
                    mu = mu.cpu()

                latent[row: row + len(mu)] = mu
                row += len(mu)

        assert row == length
        return latent

    def save(self, filehandle):
        """Saves the VAE to a path or binary opened file. Load with VAE.load

        Input: Path or binary opened filehandle
        Output: None
        """
        state = {'nsamples': self.nsamples,
                 'alpha': self.alpha,
                 'beta': self.beta,
                 'dropout': self.dropout,
                 'nlatent': self.nlatent,
                 'state': self.state_dict(),
                }

        torch.save(state, filehandle)

    @classmethod
    def load(cls, path, cuda=False, evaluate=True):
        """Instantiates a VAE from a model file.

        Inputs:
            path: Path to model file as created by functions VAE.save or
                  VAE.trainmodel.
            cuda: If network should work on GPU [False]
            evaluate: Return network in evaluation mode [True]

        Output: VAE with weights and parameters matching the saved network.
        """

        # Forcably load to CPU even if model was saves as GPU model
        dictionary = torch.load(path, map_location=lambda storage, loc: storage)

        nsamples = dictionary['nsamples']
        alpha = dictionary['alpha']
        beta = dictionary['beta']
        dropout = dictionary['dropout']
        nlatent = dictionary['nlatent']
        state = dictionary['state']

        transformer = cls(nsamples, nlatent, alpha, beta, dropout, cuda)
        transformer.load_state_dict(state)

        if cuda:
            transformer.cuda()

        if evaluate:
            transformer.eval()

        return transformer
import math
import torch
import numpy as np
import torch.nn as nn
import copy
import torch.optim as optim
import torch.utils.data as Data
import transBin.transBintools as _transBintools
from math import log

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MyDataSet(Data.Dataset):
    def __init__(self, enc_inputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        # self.dec_inputs = dec_inputs
        self.dec_outputs = enc_inputs

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_outputs[idx]

def make_dataloader(rpkm, tnf, batchsize=64, destroy=False, cuda=False):
    """Create a DataLoader and a contig mask from RPKM and TNF.

    The dataloader is an object feeding minibatches of contigs to the VAE.
    The data are normalized versions of the input datasets, with zero-contigs,
    i.e. contigs where a row in either TNF or RPKM are all zeros, removed.
    The mask is a boolean mask designating which contigs have been kept.

    Inputs:
        rpkm: RPKM matrix (N_contigs x N_samples)
        tnf: TNF matrix (N_contigs x N_TNF)
        batchsize: Starting size of minibatches for dataloader
        destroy: Mutate rpkm and tnf array in-place instead of making a copy.
        cuda: Pagelock memory of dataloader (use when using GPU acceleration)

    Outputs:
        DataLoader: An object feeding data to the VAE
        mask: A boolean mask of which contigs are kept
    """

    if not isinstance(rpkm, np.ndarray) or not isinstance(tnf, np.ndarray):
        raise ValueError('TNF and RPKM must be Numpy arrays')

    if batchsize < 1:
        raise ValueError('Minimum batchsize of 1, not {}'.format(batchsize))

    if len(rpkm) != len(tnf):
        raise ValueError('Lengths of RPKM and TNF must be the same')

    if not (rpkm.dtype == tnf.dtype == np.float32):
        raise ValueError('TNF and RPKM must be Numpy arrays of dtype float32')

    mask = tnf.sum(axis=1) != 0

    # If multiple samples, also include nonzero depth as requirement for accept
    # of sequences
    if rpkm.shape[1] > 1:
        depthssum = rpkm.sum(axis=1)
        mask &= depthssum != 0
        depthssum = depthssum[mask]

    if mask.sum() < batchsize:
        raise ValueError('Fewer sequences left after filtering than the batch size.')

    if destroy:
        rpkm = _transBintools.numpy_inplace_maskarray(rpkm, mask)
        tnf = _transBintools.numpy_inplace_maskarray(tnf, mask)
    else:
        # The astype operation does not copy due to "copy=False", but the masking
        # operation does.
        rpkm = rpkm[mask].astype(np.float32, copy=False)
        tnf = tnf[mask].astype(np.float32, copy=False)

    # If multiple samples, normalize to sum to 1, else zscore normalize
    if rpkm.shape[1] > 1:
        rpkm /= depthssum.reshape((-1, 1))
    else:
        _transBintools.zscore(rpkm, axis=0, inplace=True)

    # Normalize arrays and create the Tensors (the tensors share the underlying memory)
    # of the Numpy arrays
    _transBintools.zscore(tnf, axis=0, inplace=True)
    depthstensor = torch.from_numpy(rpkm)
    tnftensor = torch.from_numpy(tnf)
    sumtensor= torch.cat((depthstensor,tnftensor),-1)

    # Create dataloader
    n_workers = 8
    dataset = MyDataSet(sumtensor)
    dataloader = Data.DataLoader(dataset=dataset, batch_size=batchsize, drop_last=True,
                             shuffle=True, num_workers=n_workers, pin_memory=cuda)

    return dataloader, mask

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k=64):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, dropout=None):
        '''
        Q: [batch_size, n_heads, d_k]
        K: [batch_size, n_heads, d_k]
        V: [batch_size, n_heads,d_v]
        attn_mask: [batch_size, n_heads, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        attn = nn.Softmax(dim=-1)(scores)
        attn = dropout(attn)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model = 512, d_k = 64, nheads = 8, dropout=0.2):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.nheads = nheads
        self.d_v = d_k
        self.dropout = nn.Dropout(p=dropout)
        self.norm = LayerNorm(d_model)
        self.W_Q = nn.Linear(self.d_model, self.d_k * self.nheads, bias=False)
        self.W_K = nn.Linear(self.d_model, self.d_k * self.nheads, bias=False)
        self.W_V = nn.Linear(self.d_model, self.d_v * self.nheads, bias=False)
        self.fc = nn.Linear(self.nheads * self.d_v, self.d_model, bias=False)

    def forward(self, input_Q, input_K, input_V):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        input_Q = self.norm(input_Q)
        input_K = self.norm(input_K)
        input_V = self.norm(input_V)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.nheads, self.d_k).transpose(1, 2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, self.nheads, self.d_k).transpose(1, 2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, self.nheads, self.d_v).transpose(1,
                                                                           2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention(d_k=self.d_k)(Q, K, V, dropout=self.dropout)
        context = context.transpose(1, 2).reshape(batch_size,self.nheads * self.d_v)  # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, d_model]
        return residual+self.dropout(output), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model = 512, d_ff = 2048, dropout = 0.2):
        super(PoswiseFeedForwardNet, self).__init__()
        self.d_model = d_model
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
        self.dropout = nn.Dropout(p=dropout)
        self.norm = LayerNorm(self.d_model)

    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(self.norm(inputs))
        return residual+self.dropout(output)  # [batch_size, seq_len, d_model]

class EncoderLayer(nn.Module):
    def __init__(self, d_model = 512, d_ff = 2048, d_k = 64, nheads = 8, dropout = 0.2):
        super(EncoderLayer, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.d_k = d_k
        self.nheads = nheads
        self.dropout = dropout
        self.enc_self_attn = MultiHeadAttention(d_model = self.d_model, d_k = self.d_k, nheads = self.nheads, dropout=self.dropout)
        self.pos_ffn = PoswiseFeedForwardNet(d_model = self.d_model, d_ff = self.d_ff, dropout = self.dropout)

    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        '''
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs)  # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn


class DecoderLayer(nn.Module):
    def __init__(self, d_model = 512, d_ff = 2048, d_k = 64, nheads = 8, dropout = 0.2):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.d_k = d_k
        self.nheads = nheads
        self.dropout = dropout
        self.dec_self_attn = MultiHeadAttention(d_model = self.d_model, d_k = self.d_k, nheads = self.nheads, dropout=self.dropout)
        self.pos_ffn = PoswiseFeedForwardNet(d_model = self.d_model, d_ff = self.d_ff, dropout = self.dropout)

    def forward(self, dec_inputs):
        '''
        dec_inputs: [batch_size, tgt_len, d_model]
        enc_outputs: [batch_size, src_len, d_model]
        dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        '''
        # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        dec_outputs, attn1 = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs)
        # dec_outputs: [batch_size, tgt_len, d_model], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
        dec_outputs = self.pos_ffn(dec_outputs)  # [batch_size, tgt_len, d_model]
        return dec_outputs, attn1


class Encoder(nn.Module):
    def __init__(self,nsamples, nlayers = 6, d_model = 512, d_ff = 2048, d_k = 64, nheads = 8, dropout = 0.2):
        super(Encoder, self).__init__()
        self.nsamples = nsamples
        self.dropout = dropout
        self.d_model = d_model
        self.d_ff = d_ff
        self.d_k = d_k
        self.nlayers = nlayers
        self.nheads = nheads
        self.lin = nn.Linear(self.nsamples+103, d_model)
        self.norm = LayerNorm(d_model)
        self.layers = clones(EncoderLayer(d_model = self.d_model, d_ff = self.d_ff, d_k = self.d_k, nheads = self.nheads, dropout = self.dropout), self.nlayers)

    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, src_len]
        '''
        enc_outputs = self.lin(enc_inputs)
          # [batch_size, src_len, d_model]
          # [batch_size, src_len, src_len]
        # enc_outputs = enc_inputs
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs)
            enc_self_attns.append(enc_self_attn)
        return self.norm(enc_outputs), enc_self_attns


class Decoder(nn.Module):
    def __init__(self, nlayers = 6, nlatent = 64, d_model = 512, d_ff = 2048, d_k = 64, nheads = 8, dropout = 0.2):
        super(Decoder, self).__init__()
        self.dropout = dropout
        self.d_model = d_model
        self.d_ff = d_ff
        self.d_k = d_k
        self.nlayers = nlayers
        self.nheads = nheads
        self.nlatent = nlatent
        self.lin = nn.Linear(self.nlatent, d_model)
        self.norm = LayerNorm(d_model)
        self.layers = clones(DecoderLayer(d_model = self.d_model, d_ff = self.d_ff, d_k = self.d_k, nheads = self.nheads, dropout = self.dropout), self.nlayers)

    def forward(self, dec_inputs):
        dec_outputs = self.lin(dec_inputs)
        # dec_outputs = dec_inputs
        dec_enc_attns = []
        for layer in self.layers:
            # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            dec_outputs, dec_enc_attn = layer(dec_outputs)
            dec_enc_attns.append(dec_enc_attn)
        return self.norm(dec_outputs), dec_enc_attns


class Transformer(nn.Module):
    def __init__(self,nsamples, nlatent=32, d_model = 512, d_ff = 2048, d_k = 64, nlayers = 5, nheads=8, dropout=0.2, alpha=None, beta=200, cuda=False):
        if nlatent < 1:
            raise ValueError('Minimum 1 latent neuron, not {}'.format(nlatent))

        if nsamples < 1:
            raise ValueError('nsamples must be > 0, not {}'.format(nsamples))

        # If only 1 sample, we weigh alpha and nhiddens differently
        if alpha is None:
            alpha = 0.15 if nsamples > 1 else 0.50

        if dropout is None:
            dropout = 0.2 if nsamples > 1 else 0.0

        if beta <= 0:
            raise ValueError('beta must be > 0, not {}'.format(beta))

        if not (0 < alpha < 1):
            raise ValueError('alpha must be 0 < alpha < 1, not {}'.format(alpha))

        if not (0 <= dropout < 1):
            raise ValueError('dropout must be 0 <= dropout < 1, not {}'.format(dropout))

        super(Transformer, self).__init__()
        self.nsamples = nsamples
        self.dropout = dropout
        self.d_model = d_model
        self.d_ff = d_ff
        self.d_k = d_k
        self.nlayers = nlayers
        self.nheads = nheads
        self.nlatent = nlatent
        self.alpha = alpha
        self.beta = beta
        self.softplus = nn.Softplus()
        self.usecuda = cuda

        self.encoder = Encoder(self.nsamples,nlayers = self.nlayers, d_model = self.d_model, d_ff = self.d_ff, d_k = self.d_k, nheads = self.nheads, dropout = self.dropout)
        self.decoder = Decoder(nlayers = self.nlayers, nlatent = self.nlatent, d_model = self.d_model, d_ff = self.d_ff, d_k = self.d_k, nheads = self.nheads, dropout = self.dropout)
        self.projection = nn.Linear(d_model, nsamples+103, bias=False)

        # Latent layers
        self.mu = nn.Linear(d_model, self.nlatent)
        self.logsigma = nn.Linear(d_model, self.nlatent)

        if cuda:
            self.cuda()

    def reparameterize(self, mu, logsigma):
        epsilon = torch.randn(mu.size(0), mu.size(1))

        if self.usecuda:
            epsilon = epsilon.cuda()

        epsilon.requires_grad = True

        # See comment above regarding softplus
        latent = mu + epsilon * torch.exp(logsigma/2)

        return latent

    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, src_len]
        dec_inputs: [batch_size, tgt_len]
        '''
        # tensor to store decoder outputs
        # outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)

        # enc_outputs: [batch_size, src_len, d_model], enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        mu = self.mu(enc_outputs)
        logsigma = self.softplus(self.logsigma(enc_outputs))
        latent = self.reparameterize(mu, logsigma)
        # dec_outpus: [batch_size, tgt_len, d_model], dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [n_layers, batch_size, tgt_len, src_len]
        dec_outputs, dec_enc_attns = self.decoder(latent)
        dec_logits = self.projection(dec_outputs)  # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        return dec_logits,mu,logsigma

    def calc_loss(self, dec_inputs, dec_logits, mu, logsigma):
        # If multiple samples, use cross entropy, else use SSE for abundance
        sse = (dec_inputs - dec_logits).pow(2).sum(dim=1).mean()
        kld = -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).sum(dim=1).mean()
        sse_weight = self.alpha / 103
        kld_weight = 1 / (self.nlatent * self.beta)
        loss = sse * sse_weight + kld * kld_weight
        return loss, sse, kld

    def trainepoch(self, data_loader, epoch, optimizer, logfile):
        self.train()

        epoch_loss = 0
        epoch_kldloss = 0
        epoch_sseloss = 0

        for enc,dec in data_loader:
            enc.requires_grad = True
            dec.requires_grad = True

            if self.usecuda:
                enc = enc.cuda()
                dec = dec.cuda()

            optimizer.zero_grad()

            dec_logits,mu,logsigma = self(enc)

            loss, sse, kld = self.calc_loss(dec, dec_logits, mu, logsigma)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.data.item()
            epoch_kldloss += kld.data.item()
            epoch_sseloss += sse.data.item()

        if logfile is not None:
            print('\tEpoch: {}\tLoss: {:.6f}\tSSE: {:.6f}\tKLD: {:.4f}\tBatchsize: {}'.format(
                epoch + 1,
                epoch_loss / len(data_loader),
                epoch_sseloss / len(data_loader),
                epoch_kldloss / len(data_loader),
                data_loader.batch_size,
            ), file=logfile)

            logfile.flush()

        return None


    def trainmodel(self, dataloader, nepochs=200, lrate=1e-3, logfile=None, modelfile=None):
        """Train the autoencoder from depths array and tnf array.

        Inputs:
            dataloader: DataLoader made by make_dataloader
            nepochs: Train for this many epochs before encoding [500]
            lrate: Starting learning rate for the optimizer [0.001]
            batchsteps: None or double batchsize at these epochs [25, 75, 150, 300]
            logfile: Print status updates to this file if not None [None]
            modelfile: Save models to this file if not None [None]

        Output: None
        """

        if lrate < 0:
            raise ValueError('Learning rate must be positive, not {}'.format(lrate))

        if nepochs < 1:
            raise ValueError('Minimum 1 epoch, not {}'.format(nepochs))


        # Get number of features
        ncontigs, nsamples = dataloader.dataset.enc_inputs.shape
        optimizer = optim.Adam(self.parameters(), lr=lrate)

        if logfile is not None:
            print('\tNetwork properties:', file=logfile)
            print('\tCUDA:', self.usecuda, file=logfile)
            print('\tAlpha:', self.alpha, file=logfile)
            print('\tBeta:', self.beta, file=logfile)
            print('\tDropout:', self.dropout, file=logfile)
            print('\tN latent:', self.nlatent, file=logfile)
            print('\n\tTraining properties:', file=logfile)
            print('\tN epochs:', nepochs, file=logfile)
            print('\tStarting batch size:', dataloader.batch_size, file=logfile)
            print('\tLearning rate:', lrate, file=logfile)
            print('\tN sequences:', ncontigs, file=logfile)
            print('\tN samples:', nsamples-103, file=logfile, end='\n\n')

        # Train
        for epoch in range(nepochs):
            self.trainepoch(dataloader, epoch, optimizer, logfile)

        # Save weights - Lord forgive me, for I have sinned when catching all exceptions
        if modelfile is not None:
            try:
                self.save(modelfile)
            except:
                pass

        return None

    def encode(self, data_loader):
        """Encode a data loader to a latent representation with VAE

        Input: data_loader: As generated by train_vae

        Output: A (n_contigs x n_latent) Numpy array of latent repr.
        """

        self.eval()

        new_data_loader = Data.DataLoader(dataset=data_loader.dataset,
                                      batch_size=data_loader.batch_size,
                                      shuffle=False,
                                      drop_last=False,
                                      num_workers=1,
                                      pin_memory=data_loader.pin_memory)

        array1 = data_loader.dataset.enc_inputs
        length = len(array1)

        # We make a Numpy array instead of a Torch array because, if we create
        # a Torch array, then convert it to Numpy, Numpy will believe it doesn't
        # own the memory block, and array resizes will not be permitted.
        latent = np.empty((length, self.nlatent), dtype=np.float32)

        row = 0
        with torch.no_grad():
            for enc,dec in new_data_loader:
                # Move input to GPU if requested
                if self.usecuda:
                    enc = enc.cuda()

                # Evaluate
                dec_logits, mu, logsigma = self(enc)

                if self.usecuda:
                    mu = mu.cpu()

                latent[row: row + len(mu)] = mu
                row += len(mu)

        assert row == length
        return latent

    def save(self, filehandle):
        """Saves the VAE to a path or binary opened file. Load with VAE.load

        Input: Path or binary opened filehandle
        Output: None
        """
        state = {'nsamples': self.nsamples,
                 'alpha': self.alpha,
                 'beta': self.beta,
                 'dropout': self.dropout,
                 'nlatent': self.nlatent,
                 'state': self.state_dict(),
                }

        torch.save(state, filehandle)

    @classmethod
    def load(cls, path, cuda=False, evaluate=True):
        """Instantiates a VAE from a model file.

        Inputs:
            path: Path to model file as created by functions VAE.save or
                  VAE.trainmodel.
            cuda: If network should work on GPU [False]
            evaluate: Return network in evaluation mode [True]

        Output: VAE with weights and parameters matching the saved network.
        """

        # Forcably load to CPU even if model was saves as GPU model
        dictionary = torch.load(path, map_location=lambda storage, loc: storage)

        nsamples = dictionary['nsamples']
        alpha = dictionary['alpha']
        beta = dictionary['beta']
        dropout = dictionary['dropout']
        nlatent = dictionary['nlatent']
        state = dictionary['state']

        transformer = cls(nsamples, nlatent, alpha, beta, dropout, cuda)
        transformer.load_state_dict(state)

        if cuda:
            transformer.cuda()

        if evaluate:
            transformer.eval()

        return transformer
