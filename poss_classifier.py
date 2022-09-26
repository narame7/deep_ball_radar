import math
import torch
import torch.nn as nn
from torch import nn, Tensor
import torch.nn.functional as F

from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoderLayer, TransformerDecoder
from torchcrf import CRF


import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), 'set_transformer'))

from model import SetTransformer

class PossClassifier(nn.Module):
    def __init__(self, lstm_input_dim=128, lstm_hidden_dim=128, num_layers=4, ball_trace_given=True, mode='player'):
        super().__init__()
        
        if mode == 'player':
            if ball_trace_given:
                output_dim = 26
            else:
                output_dim = 27
        else:
            output_dim = 3
        
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
        self.linear = nn.Linear(in_features=lstm_hidden_dim * 2, out_features=output_dim)
        # self.softmax = nn.Softmax(dim=-1)
        
        self.progress = []
        
    def forward(self, x, coords):
        x_reshaped = x.view(x.size(0) * x.size(1), x.size(2), x.size(3), x.size(4))
        out = self.cnn(x_reshaped)
        out = out.view(x.size(0), x.size(1), -1)
        out, _ = self.lstm(out) 
        out = self.linear(out)
        return out

class PossTransformerClassifier(nn.Module):
    def __init__(self, lstm_input=128, hidden_dim=512, num_layers=6, ball_trace_given=True, mode='player'):
        super().__init__()
        
        if mode == 'player':
            if ball_trace_given:
                output_dim = 26
            else:
                output_dim = 27
        else:
            output_dim = 3
            

        self.feature_dim = lstm_input
        self.rnn_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = 8

        self.dropout = 0.1

        self.set_tf = SetTransformer(in_dimension=2, out_dimension=self.rnn_dim // 2)

        self.pos_encoder = PositionalEncoding(self.rnn_dim, self.dropout)

        encoder_layers = TransformerEncoderLayer(self.rnn_dim, self.num_heads, self.rnn_dim * 4, self.dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, self.num_layers)
        
        decoder_layer = TransformerDecoderLayer(d_model=self.rnn_dim, nhead=self.num_heads)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=self.num_layers)

        
        self.lstm = nn.LSTM(
            input_size=2 + 2 + 88,
            hidden_size=self.rnn_dim,
            num_layers=4,
            bidirectional=True,
            batch_first=True
        )
        
        self.crf = CRF(num_tags=3, batch_first=True)

        self.poss_decoder = nn.Linear(self.rnn_dim * 2, output_dim)
        
        self.first_ball_decoder = nn.Linear(self.rnn_dim, 2)
        
        self.ball_decoder = nn.Linear(self.rnn_dim * 2, 2)

        self.init_weights()

        self.progress = []

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        self.poss_decoder.bias.data.zero_()
        self.poss_decoder.weight.data.uniform_(-initrange, initrange)
        
        self.ball_decoder.bias.data.zero_()
        self.ball_decoder.weight.data.uniform_(-initrange, initrange)
        
    def predict(self, x, coords, target_traces, masking_ratio = 0.1, mask=True):
        
        bs = coords.size(0)
        seq_len = coords.size(1)        
        seq_unit = 50
        
        num_parts = seq_len // (seq_unit // 2)
        
        pos_list = []
        trace_list = []
        
        if mask:
            random_mask = (torch.cuda.FloatTensor(bs, seq_len).uniform_() < masking_ratio).unsqueeze(-1).cuda()
            target_traces = target_traces * random_mask
        
        if num_parts * seq_unit < seq_len:
            num_parts += 1
        
        for i in range(num_parts):
            pred_pos, pred_traces, _ = self.forward(x[:,i*(seq_unit // 2):(i+2)*(seq_unit // 2)], coords[:,i*(seq_unit // 2):(i+2)*(seq_unit // 2)], target_traces[:,i*(seq_unit // 2):(i+2)*(seq_unit // 2)])
            
            if i == 0:
                pos_list.append(torch.tensor(pred_pos))
                trace_list.append(pred_traces)
            else:
                pos_list.append(torch.tensor(pred_pos)[:, seq_unit // 2:])
                trace_list.append(pred_traces[:, seq_unit // 2:])
            
        return torch.cat(pos_list, 1), torch.cat(trace_list, 1), torch.cat(trace_list, 1)
    
    def predict2(self, x, coords, target_traces, masking_ratio = 0.1, mask=True):
        
        bs = coords.size(0)
        seq_len = coords.size(1)
        
        seq_unit = 50
        
        num_parts = seq_len // (seq_unit // 2)
        
        pos_list = []
        trace_list = []
        
        if num_parts * seq_unit < seq_len:
            num_parts += 1
            
        if mask:
            random_mask = (torch.cuda.FloatTensor(bs, seq_len).uniform_() < masking_ratio).unsqueeze(-1).cuda()
            target_traces = target_traces * random_mask
        
        for i in range(num_parts):
            pred_pos, pred_traces, first_ball_out = self.forward(x[:,i*(seq_unit // 2):(i+2)*(seq_unit // 2)], coords[:,i*(seq_unit // 2):(i+2)*(seq_unit // 2)], target_traces[:,i*(seq_unit // 2):(i+2)*(seq_unit // 2)])
            
            if i == 0:
                pos_list.append(torch.tensor(pred_pos))
                trace_list.append(pred_traces)
                # trace_list.append(first_ball_out)
            else:
                pos_list.append(torch.tensor(pred_pos)[:, seq_unit // 2:])
                trace_list.append(pred_traces[:, seq_unit // 2:])
                # trace_list.append(first_ball_out[:, seq_unit // 2:])
                
            if i != num_parts - 1:
                target_traces[:, :seq_unit // 2] = pred_traces[:, seq_unit // 2:]
                # target_traces[:, :seq_unit // 2] = first_ball_out[:, seq_unit // 2:]

                
            
        return torch.cat(pos_list, 1), torch.cat(trace_list, 1), torch.cat(trace_list, 1)
        

    def forward(self, x, coords, target, tags=None):

        bs = x.size(0)
        seq_len = x.size(1)

        if torch.cuda.is_available():
            coords = coords.cuda() # bs x seq_len x 88
        
        device = coords.device.type
        pitch_size = torch.tensor(x.shape[-2:]).to(device)

        num_players = 22

        team_1_coords_encoded = self.set_tf(coords.view(-1, num_players, 4)[:, :num_players//2, :2]).view(bs, seq_len, -1).transpose(0, 1) # s x b x encode_dim 
        team_2_coords_encoded = self.set_tf(coords.view(-1, num_players, 4)[:, num_players//2:, :2]).view(bs, seq_len, -1).transpose(0, 1) # s x b x encode_dim        

        coords_encoded = torch.cat([team_1_coords_encoded, team_2_coords_encoded], dim=-1) # s x b x (2 * encode_dim)

        rnn_input = self.pos_encoder(coords_encoded)  # s x b x (2 * encode_dim)

        src_mask = self.generate_square_subsequent_mask(seq_len)
        if torch.cuda.is_available():
            src_mask = src_mask.cuda()

        output = self.transformer_encoder(rnn_input).transpose(0, 1)  # b x s x (2 * encode_dim)
        
        first_ball_out = self.first_ball_decoder(output) * pitch_size  # b x s x 2
        
        
        if self.training:
            random_mask = (torch.cuda.FloatTensor(bs, seq_len).uniform_() > 0.9).unsqueeze(-1).to(device)
            masked_target = target * random_mask
        else:
            masked_target = target
        
        output = torch.cat([first_ball_out, coords, masked_target], dim=-1)
        
        #output = self.transformer_decoder(rnn_input, output, src_mask).transpose(0, 1)
        output, _ = self.lstm(output)
        
        poss_output = self.poss_decoder(output) # b x s x 3
        
        out = self.ball_decoder(output) # b x s x 2
        
        
        if tags is not None:
            log_likelihood, sequence_of_tags = self.crf(poss_output, tags), self.crf.decode(poss_output)
            return log_likelihood, sequence_of_tags, out * pitch_size, first_ball_out# * pitch_size
        else:
            sequence_of_tags = self.crf.decode(poss_output)
            return sequence_of_tags, out * pitch_size, first_ball_out#  * pitch_size
    

class PossTransformerLSTM(nn.Module):
    def __init__(self, lstm_input=128, hidden_dim=128, num_layers=6, ball_trace_given=True, mode='player'):
        super().__init__()
        
        if mode == 'player':
            if ball_trace_given:
                output_dim = 26
            else:
                output_dim = 27
        else:
            output_dim = 3
            

        self.feature_dim = lstm_input
        self.rnn_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = 8

        self.dropout = 0.1

        self.set_tf = SetTransformer(in_dimension=4, out_dimension=self.rnn_dim // 2)

        self.pos_encoder = PositionalEncoding(self.rnn_dim, self.dropout)

        encoder_layers = TransformerEncoderLayer(self.rnn_dim, self.num_heads, self.rnn_dim * 4, self.dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, self.num_layers)
        
        decoder_layer = TransformerDecoderLayer(d_model=self.rnn_dim, nhead=self.num_heads)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=self.num_layers)

        
        self.rnn_cell = nn.LSTMCell(
            input_size=self.feature_dim,
            hidden_size=self.rnn_dim,
        )

        self.poss_decoder = nn.Linear(self.rnn_dim, output_dim)
        self.ball_decoder = nn.Linear(self.rnn_dim, 2)
        
        self.crf = CRF(num_tags=3, batch_first=True)

        self.init_weights()

        self.progress = []

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        self.poss_decoder.bias.data.zero_()
        self.poss_decoder.weight.data.uniform_(-initrange, initrange)
        
        self.ball_decoder.bias.data.zero_()
        self.ball_decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, coords, tags=None):

        bs = x.size(0)
        seq_len = x.size(1)

        if torch.cuda.is_available():
            coords = coords.cuda()

        num_players = 22

        team_1_coords_encoded = self.set_tf(coords.view(-1, num_players, 4)[:, :num_players//2, :]).view(bs, seq_len, -1).transpose(0, 1)
        team_2_coords_encoded = self.set_tf(coords.view(-1, num_players, 4)[:, num_players//2:, :]).view(bs, seq_len, -1).transpose(0, 1)

        coords_encoded = torch.cat([team_1_coords_encoded, team_2_coords_encoded], dim=-1)

        rnn_input = self.pos_encoder(coords_encoded)

        src_mask = self.generate_square_subsequent_mask(seq_len)
        if torch.cuda.is_available():
            src_mask = src_mask.cuda()

        output = self.transformer_encoder(rnn_input)
        
        output = self.transformer_decoder(rnn_input, output, src_mask).transpose(0, 1)
        
        poss_output = self.poss_decoder(output)
        
        out = self.ball_decoder(output)
        
        
        hx = torch.zeros(bs, self.rnn_dim)
        cx = torch.zeros(bs, self.rnn_dim)

        if torch.cuda.is_available():
            hx = hx.cuda()
            cx = cx.cuda()

        rnn_output = []

        coords = coords.transpose(0, 1)
        
        first_ball_output = out.transpose(0, 1)
        poss_output_argmax = torch.argmax(poss_output.transpose(0, 1), dim=-1)

        for i in range(seq_len):
                        
            pred_trace = torch.unsqueeze(first_ball_output[i], dim=1)
            player_x = coords[i, :, 0:44:2]  # seq_len x bs x 22
            player_y = coords[i, :, 1:44:2]  # seq_len x bs x 22
            player_trace = torch.stack([player_x, player_y], dim=-1)  # seq_len x bs x 22 x 2

            ball_dists = torch.linalg.norm(pred_trace - player_trace, dim=-1)
            
            closest_index = torch.min(ball_dists, dim=-1).indices
            
            print(closest_index.shape)

            rnn_input = torch.cat([coords[i], coords_encoded[i], closest_dist, step_out], dim=-1)

            hx, cx = self.rnn_cell(rnn_input, (hx, cx))

            step_out = self.linear(hx) * pitch_size  # batch_size x 2

            rnn_output.append(step_out)


        out = torch.stack(rnn_output, dim=0).transpose(0, 1)
        
        device = coords.device.type
        pitch_size = torch.tensor(x.shape[-2:]).to(device)
        
        if tags is not None:
            log_likelihood, sequence_of_tags = self.crf(out, tags), self.crf.decode(out)
            return log_likelihood, sequence_of_tags, out * pitch_size
        else:
            sequence_of_tags = self.crf.decode(out)
            return sequence_of_tags, out * pitch_size
    
        #return poss_output, out * pitch_size
    
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
