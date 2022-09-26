import math
import torch
import torch.nn as nn
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import random

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), 'set_transformer'))

from model import SetTransformer

class TraceLSTM(nn.Module):
    def __init__(self, lstm_input_dim=128, lstm_hidden_dim=128, num_layers=4, target_type='ball'):
        super().__init__()

        self.target_type = target_type

        self.cnn = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )
        self.linear = nn.Linear(lstm_hidden_dim * 2, 2)

        self.progress = []

    def forward(self, x, coords):
        x_reshaped = x.view(x.size(0) * x.size(1), x.size(2), x.size(3), x.size(4))
        out = self.cnn(x_reshaped)
        out = out.view(x.size(0), x.size(1), -1)
        out, _ = self.lstm(out)
        out = self.linear(out)
        out = torch.sigmoid(out)

        device = x.device.type
        pitch_size = torch.tensor(x.shape[-2:]).to(device)

        if self.target_type == 'ball':
            return out * pitch_size
        else:
            # i.e. if self.mode == 'gk':
            return torch.cat([out[:, :, 0:2] * pitch_size, out[:, :, 2:4] * pitch_size], dim=2)

class TraceSetLSTM(nn.Module):
    def __init__(self, lstm_input_dim=128, lstm_hidden_dim=512, num_layers=4, target_type='ball'):
        super().__init__()
        self.target_type = target_type    # 'ball' or 'gk'
        self.rnn_dim = lstm_hidden_dim
        self.num_layers = num_layers
        
        self.num_players = 22 if self.target_type == 'ball' else 20

        self.set_tf = SetTransformer(in_dimension=2, out_dimension=self.rnn_dim // 2)
        
        self.output_dim = 2 if self.target_type == 'ball' else 4    # i.e. 4 if self.mode == 'gk'

        if self.target_type == 'ball':
            self.ball_rnn = nn.LSTM(
                input_size=self.rnn_dim + self.num_players * 5 + self.output_dim,
                hidden_size=lstm_hidden_dim, num_layers = num_layers, dropout=0.0
            )
            self.pos_rnn = nn.LSTM(
                input_size=self.rnn_dim + self.num_players * 5 + self.output_dim,
                hidden_size=lstm_hidden_dim, num_layers = num_layers, dropout=0.0
            )
        else:
            self.ball_rnn = nn.LSTM(
                input_size=self.rnn_dim + self.num_players * 2 + self.output_dim,
                hidden_size=lstm_hidden_dim, num_layers = num_layers, dropout=0.0
            )
            self.pos_rnn = nn.LSTM(
                input_size=self.rnn_dim + self.num_players * 2 + self.output_dim,
                hidden_size=lstm_hidden_dim, num_layers = num_layers, dropout=0.0
            )
            
        
        self.trajectory_rnn = nn.LSTM(input_size=2, hidden_size = lstm_hidden_dim //4, num_layers = self.num_layers)
        
        self.linear = nn.Linear(lstm_hidden_dim, self.output_dim * 2)
        self.linear_single = nn.Linear(lstm_hidden_dim // 4, 4)
        
        self.pos_linear = nn.Linear(lstm_hidden_dim, 3 * 2)
        
        self.glu = nn.GLU()


        self.progress = []

    def forward(self, x, coords, target_traces):

        bs = x.size(0)
        seq_len = x.size(1)

        if torch.cuda.is_available():
            x, coords = x.cuda(), coords.cuda()

        device = x.device.type
        pitch_size = torch.tensor(x.shape[-2:]).to(device)

        if self.target_type == 'ball':
            num_players = 22
        else:
            num_players = 20

        
        scale = torch.cat([pitch_size, torch.ones(2).to(device)], -1).repeat(num_players)
        
        
        coords = coords / scale
            
                                                
        team_1_coords_encoded = self.set_tf(coords.view(-1, num_players, 4)[:, :num_players//2, :2]).view(bs, seq_len, -1).transpose(0, 1)
        team_2_coords_encoded = self.set_tf(coords.view(-1, num_players, 4)[:, num_players//2:, :2]).view(bs, seq_len, -1).transpose(0, 1)

        coords_encoded = torch.cat([team_1_coords_encoded, team_2_coords_encoded], dim=-1) # seq_len x bs x hidden_dim


        hx = torch.zeros(self.num_layers, bs, self.rnn_dim)
        cx = torch.zeros(self.num_layers, bs, self.rnn_dim)
        hx2 = torch.zeros(self.num_layers, bs, self.rnn_dim)
        cx2 = torch.zeros(self.num_layers, bs, self.rnn_dim)
        
        single_hx = torch.zeros(self.num_layers, bs * self.num_players, self.rnn_dim // 4)
        single_cx = torch.zeros(self.num_layers, bs * self.num_players, self.rnn_dim // 4)

        if torch.cuda.is_available():
            hx = hx.cuda()
            cx = cx.cuda()
            hx2 = hx2.cuda()
            cx2 = cx2.cuda()
            
            single_hx = single_hx.cuda()
            single_cx = single_cx.cuda()

        rnn_output = []
        pos_output = []

        coords = coords.transpose(0, 1) # seq_len x bs x 80
        target_traces = target_traces.transpose(0, 1) # seq_len x 4

        for i in range(seq_len):

            if i == 0:
                step_out = torch.zeros(bs, self.output_dim).cuda()
            #else:
            #    if self.training and random.random() < 0.5:
            #        step_out = target_traces[i-1]
            
            player_x = coords[i, :, 0::4]
            player_y = coords[i, :, 1::4]
            player_trace = torch.stack([player_x, player_y], dim=-1)  # bs x players(20) x 2(x,y)

            if self.target_type == 'ball':
                pred_trace = torch.unsqueeze(step_out[:,-2:], dim=1)
                
                _, (single_hx, single_cx) = self.trajectory_rnn(player_trace.view(bs * self.num_players, -1).unsqueeze(0), (single_hx, single_cx))

                player_step_out = self.glu(self.linear_single(single_hx[-1].view(bs, self.num_players, -1))).view(bs, -1)# * pitch_size.repeat(22)        
                
                # print(player_step_out.shape)

                # coords_encoded = self.set_tf(player_step_out.view(bs, 22, 2))

                ball_dists = torch.linalg.norm(pred_trace - player_trace, dim=-1).sort(-1)[0]
                closest_dist = torch.min(ball_dists, dim=-1).values.unsqueeze(1)

                rnn_input = torch.cat([player_trace.view(bs, -1), coords_encoded[i], player_step_out, ball_dists, step_out[:,-1 * self.output_dim:]], dim=-1)
            else:
                rnn_input = torch.cat([player_trace.view(bs, -1), coords_encoded[i], step_out[:,-1 * self.output_dim:]], dim=-1)

            _, (hx, cx) = self.ball_rnn(rnn_input.unsqueeze(0), (hx, cx))
            _, (hx2, cx2) = self.pos_rnn(rnn_input.unsqueeze(0), (hx2, cx2))
                        

            step_out = self.glu(self.linear(hx[-1]))# * pitch_size  # batch_size x 4  
            
            if self.target_type == 'ball': 
                rnn_output.append(torch.cat([player_step_out, step_out], -1))
            else:
                rnn_output.append(step_out)
            
            pos_out = self.glu(self.pos_linear(hx2[-1]))
            pos_output.append(pos_out)

        out = torch.stack(rnn_output, dim=0).transpose(0, 1)
        pos_output = torch.stack(pos_output, dim=0).transpose(0, 1)

        if self.target_type == 'ball':
            return out * pitch_size.repeat(23), pos_output
        else:
            # i.e. if self.mode == 'gk':
            return out * pitch_size.repeat(2)

class TraceSeq2Seq(nn.Module):
    def __init__(self, hidden_dim=128, lstm_input=128, num_layers=2):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.cnn = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

        self.encoder_lstm = nn.LSTM(
            input_size=lstm_input,   # 45, see the data definition
            hidden_size=hidden_dim,  # Can vary
            num_layers=num_layers,
            bidirectional=True,
            batch_first=False
        )

        self.set_tf = SetTransformer(in_dimension=2, out_dimension=self.rnn_dim // 2)

        self.rnn_cell = nn.LSTMCell(
            input_size=lstm_input_dim + 44 + 3,
            hidden_size=lstm_hidden_dim * 2,
        )

        output_dim = 2 + 88 if self.target_type == 'ball' else 4    # i.e. 4 if self.mode == 'gk'
        self.linear = nn.Linear(lstm_hidden_dim * 2, output_dim)

        self.attention = Attention(hidden_dim, hidden_dim)

        self.progress = []
        

    def forward(self, x, coords):
        
        bs = x.size(0)
        seq_len = x.size(1)

        if torch.cuda.is_available():
            x, coords = x.cuda(), coords.cuda()

        device = x.device.type
        pitch_size = torch.tensor(x.shape[-2:]).to(device)

        if self.target_type == 'ball':
            num_players = 22
        else:
            num_players = 20
            
                                                
        team_1_coords_encoded = self.set_tf(coords.view(-1, num_players, 4)[:, :num_players//2, :2]).view(bs, seq_len, -1).transpose(0, 1)
        team_2_coords_encoded = self.set_tf(coords.view(-1, num_players, 4)[:, num_players//2:, :2]).view(bs, seq_len, -1).transpose(0, 1)

        coords_encoded = torch.cat([team_1_coords_encoded, team_2_coords_encoded], dim=-1) # seq_len x bs x hidden_dim


        hx = torch.zeros(bs, self.rnn_dim * 2)
        cx = torch.zeros(bs, self.rnn_dim * 2)

        if torch.cuda.is_available():
            hx = hx.cuda()
            cx = cx.cuda()

        rnn_output = []

        coords = coords.transpose(0, 1)

        for i in range(seq_len):

            if i == 0:
                step_out = torch.zeros(bs, 2).cuda()

            pred_trace = torch.unsqueeze(step_out[:,-2:], dim=1)
            player_x = coords[i, :, 0:88:4]
            player_y = coords[i, :, 1:88:4]
            player_trace = torch.stack([player_x, player_y], dim=-1)

            ball_dists = torch.linalg.norm(pred_trace - player_trace, dim=-1)
            closest_dist = torch.min(ball_dists, dim=-1).values.unsqueeze(1)
            
            # print(coords[i].shape, coords_encoded[i].shape, closest_dist.shape, step_out.shape)

            rnn_input = torch.cat([player_trace[i], coords_encoded[i], closest_dist, step_out[:,-2:]], dim=-1)

            hx, cx = self.rnn_cell(rnn_input, (hx, cx))

            step_out = self.linear(hx) * pitch_size.repeat(45)  # batch_size x 2

            rnn_output.append(step_out)


        out = torch.stack(rnn_output, dim=0).transpose(0, 1)
        #out = torch.sigmoid(out)



        if self.target_type == 'ball':
            return out# * pitch_size
        else:
            # i.e. if self.mode == 'gk':
            return torch.cat([out[:, :, -4:-2] * pitch_size, out[:, :, -2:] * pitch_size], dim=2)
        
        
        

        encoder_outputs, hidden = self.encoder_lstm(out)

        #print(hidden[0].shape, hidden[1].shape)

        hidden = (hidden[0][-self.num_layers:,:,:], hidden[1][-self.num_layers:,:,:])


        #tensor to store decoder outputs
        outputs = torch.zeros(seq_len, batch_size, self.hidden_dim).to(device)


        for t in range(seq_len):

            #print(hidden_cat.shape, encoder_outputs.shape)

            a = self.attention(hidden[0][-1], encoder_outputs)

            #a = [batch size, src len]

            a = a.unsqueeze(1)

            #a = [batch size, 1, src len]

            encoder_outputs = encoder_outputs.permute(1, 0, 2)

            #encoder_outputs = [batch size, src len, enc hid dim * 2]

            weighted = torch.bmm(a, encoder_outputs)

            encoder_outputs = encoder_outputs.permute(1, 0, 2)

            #weighted = [batch size, 1, enc hid dim * 2]

            weighted = weighted.permute(1, 0, 2)

            #weighted = [1, batch size, enc hid dim * 2]

            rnn_input = torch.cat((out[t].unsqueeze(0), weighted), dim = 2)

            #insert input token embedding, previous hidden state and all encoder hidden states
            #receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder_lstm(rnn_input, hidden)

            #place predictions in a tensor holding predictions for each token
            outputs[t] = output


        #output, hidden = self.decoder_lstm(rnn_input, (hidden[0].unsqueeze(0), context[0].unsqueeze(0)))

        outputs = outputs.transpose(0, 1)

        out = torch.sigmoid(self.linear(outputs))
        #out = self.linear(outputs)

        pitch_size = torch.tensor(x.shape[-2:]).to(device)
        return out * pitch_size


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)

    def forward(self, hidden, encoder_outputs):

        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim * 2]

        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        #repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        #hidden = [batch size, src len, dec hid dim]
        #encoder_outputs = [batch size, src len, enc hid dim * 2]

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2)))

        #energy = [batch size, src len, dec hid dim]

        attention = self.v(energy).squeeze(2)

        #attention = [batch size, src len]

        return F.softmax(attention, dim=1)


class TraceTransformer(nn.Module):
    def __init__(self, hidden_dim=128, lstm_input=16, num_layers=3, target_type='ball'):
        super().__init__()

        self.feature_dim = lstm_input
        self.rnn_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = 4
        self.target_type = target_type

        self.dropout = 0.1

        self.cnn = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, self.rnn_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

        self.pos_encoder = PositionalEncoding(self.rnn_dim, self.dropout)

        encoder_layers = TransformerEncoderLayer(self.rnn_dim, self.num_heads, self.rnn_dim * 4, self.dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, self.num_layers)

        output_dim = 2 + 44 if self.target_type == 'ball' else 4

        self.decoder = nn.Linear(self.rnn_dim, output_dim)

        self.init_weights()

        self.progress = []

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, coords, target_traces):

        bs = x.size(0)
        seq_len = x.size(1)

        if torch.cuda.is_available():
            x = x.cuda()

        x_reshaped = x.view(bs * seq_len, x.size(2), x.size(3), x.size(4))
        out = self.cnn(x_reshaped)
        rnn_input = out.view(bs, seq_len, -1).transpose(0, 1)

        # rnn_input = self.input_layer(out)
        rnn_input = self.pos_encoder(rnn_input)

        src_mask = self.generate_square_subsequent_mask(seq_len)
        if torch.cuda.is_available():
            src_mask = src_mask.cuda()

        output = self.transformer_encoder(rnn_input).transpose(0, 1)
        out = self.decoder(output)

        device = x.device.type
        pitch_size = torch.tensor(x.shape[-2:]).to(device)

        if self.target_type == 'ball':
            return out * pitch_size.repeat(23)
        else:
            # i.e. if self.mode == 'gk':
            return torch.cat([out[:, :, 0:2] * pitch_size, out[:, :, 2:4] * pitch_size], dim=2)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TraceSetTransformer(nn.Module):
    def __init__(self, hidden_dim=256, lstm_input=16, num_layers=3, target_type='ball'):
        super().__init__()
        self.target_type = target_type

        self.feature_dim = lstm_input
        self.rnn_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = 4

        self.dropout = 0.1

        self.set_tf = SetTransformer(in_dimension=2, out_dimension=self.rnn_dim // 2)

        self.pos_encoder = PositionalEncoding(self.rnn_dim + (44 if target_type =='ball' else 40) , self.dropout)

        encoder_layers = TransformerEncoderLayer(self.rnn_dim + (44 if target_type =='ball' else 40), self.num_heads, self.rnn_dim * 4, self.dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, self.num_layers)

        output_dim = 2 + 44 if self.target_type == 'ball' else 4

        self.team_1_decoder = nn.Linear(self.rnn_dim + (44 if target_type =='ball' else 40), 2)
        self.team_2_decoder = nn.Linear(self.rnn_dim + (44 if target_type =='ball' else 40), 2)

        self.init_weights()

        self.progress = []

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        self.team_1_decoder.bias.data.zero_()
        self.team_1_decoder.weight.data.uniform_(-initrange, initrange)
        self.team_2_decoder.bias.data.zero_()
        self.team_2_decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, coords, target_traces):

        bs = x.size(0)
        seq_len = x.size(1)

        if torch.cuda.is_available():
            x, coords = x.cuda(), coords.cuda()

        # coords_encoded = self.set_tf(coords.reshape(-1, 22, 2)).view(bs, seq_len, -1).transpose(0, 1)
        if self.target_type == 'ball':
            num_players = 22
        else:
            num_players = 20

        team_1_coords_encoded = self.set_tf(coords.view(-1, num_players, 4)[:, :num_players//2, :2]).view(bs, seq_len, -1).transpose(0, 1)
        team_2_coords_encoded = self.set_tf(coords.view(-1, num_players, 4)[:, num_players//2:, :2]).view(bs, seq_len, -1).transpose(0, 1)

        coords_encoded = torch.cat([team_1_coords_encoded, team_2_coords_encoded], dim=-1) # seq_len x bs x hidden_dim
        
        coords = coords.transpose(0, 1)
        
        player_x = coords[:, :, 0::4]
        player_y = coords[:, :, 1::4]
        player_trace = torch.stack([player_x, player_y], dim=-1).view(seq_len, bs, -1)
        
        rnn_input = torch.cat([player_trace, coords_encoded], -1)

        rnn_input = self.pos_encoder(rnn_input)

        src_mask = self.generate_square_subsequent_mask(seq_len)
        if torch.cuda.is_available():
            src_mask = src_mask.cuda()

        output = self.transformer_encoder(rnn_input).transpose(0, 1)

        team_1_out = torch.sigmoid(self.team_1_decoder(output)) 
        team_2_out = torch.sigmoid(self.team_2_decoder(output)) 
        # out = torch.sigmoid(self.decoder(output))

        device = x.device.type
        pitch_size = torch.tensor(x.shape[-2:]).to(device)

        if self.target_type == 'ball':
            return out * pitch_size.repeat(23)
        else:
            # i.e. if self.mode == 'gk':
            # return torch.cat([out[:, :, 0:2] * pitch_size, out[:, :, 2:4] * pitch_size], dim=2)
            return torch.cat([team_1_out * pitch_size, team_2_out * pitch_size], dim=2)
            
