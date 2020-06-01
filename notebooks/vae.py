import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from itertools import chain

class GumbelSoftmax(nn.Module):
    
    def __init__(self,eps=1e-20,t=0.01,is_cuda=True):
        super(GumbelSoftmax, self).__init__()
        self.eps = eps
        self.t = t
        self.is_cuda = is_cuda
        self.softmax = nn.Softmax(dim=-1)
        
    def gumbel_sample(self,size):
        U = torch.rand(size)
        if self.is_cuda:
            U = U.cuda()
        return -torch.log(-torch.log(U + self.eps) + self.eps)
    
    def gumbel_softmax_sample(self,logits):
        y = logits + self.gumbel_sample(logits.size())
        return self.softmax(y / self.t)
    
    def forward(self,x):
        return self.gumbel_softmax_sample(x)

class feedEmbed(nn.Module):
    
    def __init__(self,E):
        super(feedEmbed, self).__init__()
        self.E = E
        self.G = GumbelSoftmax() # Gumbel softmax
        
    def forward(self,u):
        O_hat = self.G(u)
        return (O_hat @ self.E.weight)
        


def disable_all_grads(model):
    """
        freeze model
    """
    for name,param in model.named_parameters():
        param.requires_grad = False

def enable_all_grads(model):
    """
        unfreeze model
    """
    for name,param in model.named_parameters():
        param.requires_grad = True
        


class RNN_VAE(nn.Module):
    """
    1. Hu, Zhiting, et al. "Toward controlled generation of text." ICML. 2017.
    2. Bowman, Samuel R., et al. "Generating sentences from a continuous space." arXiv preprint arXiv:1511.06349 (2015).
    3. Kim, Yoon. "Convolutional neural networks for sentence classification." arXiv preprint arXiv:1408.5882 (2014).
    """

    def __init__(self, n_vocab, h_dim, z_dim, c_dim, p_word_dropout=0.3, unk_idx=0, pad_idx=1, start_idx=2, 
                 eos_idx=3, max_sent_len=15, pretrained_embeddings=None, freeze_embeddings=False, gpu=False):
        super(RNN_VAE, self).__init__()

        self.UNK_IDX = unk_idx
        self.PAD_IDX = pad_idx
        self.START_IDX = start_idx
        self.EOS_IDX = eos_idx
        self.MAX_SENT_LEN = max_sent_len

        self.n_vocab = n_vocab
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.p_word_dropout = p_word_dropout
        
        
        self.gpu = gpu
        self.C = torch.randn([2,self.z_dim])
        if self.gpu:
            self.C = self.C.cuda()
        self.C = nn.Parameter(self.C)
        

        """
        Word embeddings layer
        """
        if pretrained_embeddings is None:
            self.emb_dim = h_dim
            self.word_emb = nn.Embedding(n_vocab, h_dim, self.PAD_IDX)
        else:
            self.emb_dim = pretrained_embeddings.size(1)
            self.word_emb = nn.Embedding(n_vocab, self.emb_dim, self.PAD_IDX)

            # Set pretrained embeddings
            self.word_emb.weight.data.copy_(pretrained_embeddings)

            if freeze_embeddings:
                self.word_emb.weight.requires_grad = False

        """
        Encoder is GRU with FC layers connected to last hidden unit
        """
        self.encoder = nn.GRU(self.emb_dim, h_dim)
        self.q_mu = nn.Linear(h_dim, z_dim)
        self.q_logvar = nn.Linear(h_dim, z_dim)

        """
        Decoder is GRU with `z` and `c` appended at its inputs
        """
        self.decoder = nn.GRU(self.emb_dim+2*z_dim+c_dim, 2*z_dim+c_dim, dropout=0.3)
        self.decoder_fc = nn.Linear(2*z_dim+c_dim, n_vocab)

        """
        Grouping the model's parameters: separating encoder, decoder, and discriminator
        """
        self.encoder_params = list(chain(
            self.encoder.parameters(), self.q_mu.parameters(),
            self.q_logvar.parameters()
        ))

        self.decoder_params = list(chain(
            self.decoder.parameters(), self.decoder_fc.parameters()
        ))

        self.vae_params = list(chain(
            self.word_emb.parameters(), self.encoder_params, self.decoder_params
        ))
        self.vae_params = list(filter(lambda p: p.requires_grad, self.vae_params)) + [self.C]

        """
        Use GPU if set
        """
        if self.gpu:
            self.cuda()

    def forward_encoder(self, inputs):
        """
        Inputs is batch of sentences: seq_len x mbsize
        """
        inputs = self.word_emb(inputs)
        return self.forward_encoder_embed(inputs)

    def forward_encoder_embed(self, inputs):
        """
        Inputs is embeddings of: seq_len x mbsize x emb_dim
        """
        _, h = self.encoder(inputs, None)

        # Forward to latent
        h = h.view(-1,self.h_dim)
        mu = self.q_mu(h)
        logvar = self.q_logvar(h)

        return mu, logvar

    def sample_z(self, mu, logvar):
        """
        Reparameterization trick: z = mu + std*eps; eps ~ N(0, I)
        """
        eps = Variable(torch.randn(self.z_dim))
        eps = eps.cuda() if self.gpu else eps
        return mu + torch.exp(logvar/2) * eps

    def sample_z_prior(self, mbsize):
        """
        Sample z ~ p(z) = N(0, I)
        """
        z = Variable(torch.randn(mbsize, self.z_dim))
        z = z.cuda() if self.gpu else z
        return z

    def sample_c_prior(self, mbsize):
        """
        Sample c ~ p(c) = Cat([0.5, 0.5])
        """
        c = Variable(
            torch.from_numpy(np.random.multinomial(1, [0.5, 0.5], mbsize).astype('float32'))
        )
        c = c.cuda() if self.gpu else c
        return c

    def forward_decoder(self, inputs, z, c):
        """
        Inputs must be embeddings: seq_len x mbsize
        """
        dec_inputs = self.word_dropout(inputs)

        # Forward
        seq_len = dec_inputs.size(0)

        # 1 x mbsize x (z_dim+c_dim)
        c = c.unsqueeze(0)
        c_idx =torch.argmax(c,-1)
        C = self.C[c_idx]
        z = z.unsqueeze(0)
        #print(z.shape,C.shape)
        z = torch.cat([z,C], dim=2)
        
        init_h = torch.cat([z, c], dim=2)
        inputs_emb = self.word_emb(dec_inputs)  # seq_len x mbsize x emb_dim
        inputs_emb = torch.cat([inputs_emb, init_h.repeat(seq_len, 1, 1)], 2)

        outputs, _ = self.decoder(inputs_emb, init_h)
        seq_len, mbsize, _ = outputs.size()

        outputs = outputs.view(seq_len*mbsize, -1)
        y = self.decoder_fc(outputs)
        y = y.view(seq_len, mbsize, self.n_vocab)

        return y



    #def forward_target(self, sentence)

    def forward(self, sentence,c):
        """
 
        """
        self.train()

        mbsize = sentence.size(1)

        # sentence: '<start> I want to fly <eos>'
        # enc_inputs: '<start> I want to fly <eos>'
        # dec_inputs: '<start> I want to fly <eos>'
        # dec_targets: 'I want to fly <eos> <pad>'
        pad_words = Variable(torch.LongTensor([self.PAD_IDX])).repeat(1, mbsize)
        pad_words = pad_words.cuda() if self.gpu else pad_words

        enc_inputs = sentence
        dec_inputs = sentence
        dec_targets = torch.cat([sentence[1:], pad_words], dim=0)

        # Encoder: sentence -> z
        mu, logvar = self.forward_encoder(enc_inputs)
        z = self.sample_z(mu, logvar)
        
    
        # Decoder: sentence -> y
        y = self.forward_decoder(dec_inputs, z, c)
        
        recon_loss = F.nll_loss(
            F.log_softmax(y.view(-1, self.n_vocab),-1), dec_targets.view(-1), size_average=True,
        )
        kl_loss = torch.mean(0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1 - logvar, 1))

        return recon_loss, kl_loss, y

 


    def sample_sentence(self, z, c, raw=False, temp=1,top_k=1):
        """
        Sample single sentence from p(x|z,c) according to given temperature.
        `raw = True` means this returns sentence as in dataset which is useful
        to train discriminator. `False` means that this will return list of
        `word_idx` which is useful for evaluation.
        """
        #self.eval()

        word = torch.LongTensor([self.START_IDX])
        word = word.cuda() if self.gpu else word
        word = Variable(word)  # '<start>'

        z, c = z.view(1, 1, -1), c.view(1, 1, -1)

        h = torch.cat([z, c], dim=2)

        if not isinstance(h, Variable):
            h = Variable(h)

        outputs = []
        logits = []
        sent = []

        if raw:
            outputs.append(self.START_IDX)

        for i in range(self.MAX_SENT_LEN):
            emb = self.word_emb(word).view(1, 1, -1)
            emb = torch.cat([emb, z, c], 2)
            output, h = self.decoder(emb, h)
            y = self.decoder_fc(output)
            
            y = y.view(-1)
            logits.append(y)
            y = F.softmax(y/temp, dim=0)
            
            topk_y, topk_idx = torch.topk(y, top_k)
            
            idx = torch.multinomial(torch.tensor(topk_idx,dtype=torch.float32),1)[0]
            idx = topk_idx[idx]

            word = Variable(torch.LongTensor([int(idx)]))
            word = word.cuda() if self.gpu else word

            idx = int(idx)

            if not raw and idx == self.EOS_IDX:
                break
            outputs.append(output)
            sent.append(idx)
        logits = torch.stack(logits)
       
        
        # Back to default state: train
        self.train()

        
        return sent,logits
    
    def generate_sentences(self, batch_size):
        """
        Generate sentences and corresponding z of (batch_size x max_sent_len)
        """
        samples = []
        cs = []

        for _ in range(batch_size):
            z = self.sample_z_prior(1)
            c = self.sample_c_prior(1)
            z=torch.cat([z,self.C[torch.argmax(c)].unsqueeze(0)],-1)

            sample_idxs,sample_logits = self.sample_sentence(z, c,raw=True)
            sample_logits = sample_logits.unsqueeze(0)
            #sample_sent = dataset.idxs2sentence(sample_idxs)
            samples.append(sample_logits)
            cs.append(c)

        X_gen = torch.cat(samples, dim=0)
        c_gen = torch.cat(cs, dim=0)

        return X_gen, c_gen


    def word_dropout(self, inputs):
        """
        Do word dropout: with prob `p_word_dropout`, set the word to '<unk>'.
        """
        if isinstance(inputs, Variable):
            data = inputs.data.clone()
        else:
            data = inputs.clone()

        # Sample masks: elems with val 1 will be set to <unk>
        mask = torch.from_numpy(
            np.random.binomial(1, p=self.p_word_dropout, size=tuple(data.size()))
                     .astype('uint8')
        )

        if self.gpu:
            mask = mask.cuda()

        # Set to <unk>
        data[mask] = self.UNK_IDX

        return Variable(data)