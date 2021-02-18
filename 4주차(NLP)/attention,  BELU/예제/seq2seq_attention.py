from tqdm import tqdm
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import functional as F

import torch
import random

class Encoder(nn.Module):
    def __init__(self, vocab_size=100, embedding_size=256, hidden_size=512, num_layers=2, num_dir=2, drop_out=0.1):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.num_dir= num_dir
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.gru = nn.GRU(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True if num_dir > 1 else False,
            dropout=drop_out
        )
        #2 방향이기 때문에 hidden size를 Decoder의 hidden size와 맞추어 주어야한다.
        self.linear = nn.Linear(num_dir*hidden_size, hidden_size)

    def forward(self, batch, batch_lens):                                           # batch: (B : batch_size, S_L : max_len),   batch_lens: (B)
        batch_emb = self.embedding(batch)                                           # (B, S_L, d_w : embedding size)
        batch_emb = batch_emb.transpose(0, 1)                                       # batch size와 한 문장?당 들어있는 단어수 transpose

        packed_input = pack_padded_sequence(batch_emb, batch_lens)

        h_0 = torch.zeros((self.num_layers*self.num_dir, batch.shape[0], self.hidden_size))       # 나오는 방향*RNN layer수에 따라 개수가 달라지기 때문 concat?
        packed_outputs, h_n = self.gru(packed_input, h_0)
        outputs = pad_packed_sequence(packed_outputs)[0]                            #?
        outputs = torch.tanh(self.linear(outputs))
        
        # 4개의 output이 생성되는 것? 다음 레이어는 사용 안하는 것인가?
        forward_hidden = h_n[-2, :, :]
        backward_hidden = h_n[-1, :, :]
        hidden = torch.tanh(self.linear(torch.cat((forward_hidden, backward_hidden), dim=-1))).unsqueeze(0)     #(1, B, d_h)

        return outputs, hidden


class DotAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, decoder_hidden, encoder_hiddens):
        query = decoder_hidden.squeeze(0)
        key = encoder_hiddens.transpose(0, 1)

        energy = torch.sum(torch.mul(key, query.unsqueeze(1)), dim=-1)

        attn_scores = F.softmax(energy, dim=-1)
        attn_values = torch.sum(torch.mul(encoder_hiddens.transpose(0, 1), attn_scores.unsqueeze(2)), dim=1)

        return attn_values, attn_scores


class Decoder(nn.Module):
    def __init__(self, attention, vocab_size=100, embedding_size=256, hidden_size=512, drop_out=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.attention = attention
        self.gru = nn.GRU(
            input_size=embedding_size,
            hidden_size=hidden_size
        )
        self.output_linear = nn.Linear(2*hidden_size, vocab_size)

    def forward(self, batch, encoder_hiddens, hidden):
        batch_emb = self.embedding(batch)
        batch_emb = batch_emb.unsqueeze(0)
        outputs, hidden = self.gru(batch_emb, hidden)
        
        attn_values, attn_scores = self.attention(hidden, encoder_hiddens)
        concat_output = torch.cat((outputs, attn_values.unsqueeze(0)), dim=-1)

        return self.output_linear(concat_output).squeeze(0), hidden


class Seq2seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, src_batch, src_batch_lens, trg_batch, trg_max_len, vocab_size, teacher_forcing_prob=0.5):
        encoder_hiddens, hidden = self.encoder(src_batch, src_batch_lens)
        
        input_ids = trg_batch[:, 0]
        batch_size = src_batch.shape[0]
        
        outputs = torch.zeros(trg_max_len, batch_size, vocab_size)

        for t in range(1, trg_max_len):
            decoder_outputs, hidden = self.decoder(input_ids, encoder_hiddens, hidden)

            outputs[t] = decoder_outputs    #확률 값
            _, top_ids = torch.max(decoder_outputs, dim=-1)

            input_ids = trg_batch[:, t] if random.random() > teacher_forcing_prob else top_ids
        
        return outputs
    
if __name__ == "__main__":
    encoder = Encoder()

