# Network for binary classification
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_output):
        scores = self.attention(lstm_output)  # (batch_size, seq_length, 1)
        attention_weights = F.softmax(scores, dim=1)  # (batch_size, seq_length, 1)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)  # (batch_size, hidden_dim)
        return context_vector, attention_weights

class TimeSeriesAttnLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=1, dropout=0, bidirectional=False, seq_length=49):
        super(TimeSeriesAttnLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True, bidirectional=bidirectional)
        self.attention = Attention(hidden_dim * 2 if bidirectional else hidden_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_length, hidden_dim)
        context_vector, _ = self.attention(lstm_out)  # (batch_size, hidden_dim)
        return context_vector
    
class TimeSeriesLSTM(nn.Module):
    def __init__(self, seq_length=49, input_dim=1, hidden_dim=64, num_layers=1, dropout=0):
        super(TimeSeriesLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # LSTM output: (batch_size, seq_length, hidden_dim)
        lstm_out, _ = self.lstm(x)
        # Global Average Pooling
        lstm_out = lstm_out.permute(0, 2, 1)  # (batch_size, hidden_dim, seq_length)
        pooled_output = self.global_avg_pool(lstm_out).squeeze(-1)  # (batch_size, hidden_dim)
        return pooled_output


class TimeSeriesTransformer(nn.Module):
    def __init__(self, seq_length, d_model=64, nhead=8, num_encoder_layers=3, dim_feedforward=256, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.embedding = nn.Linear(1, d_model)
        self.position_encoding = nn.Parameter(torch.zeros(1, seq_length, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # Global Average Pooling

    def forward(self, src):
        src = src.unsqueeze(-1)  # (batch_size, seq_length, 1)
        src = self.embedding(src) + self.position_encoding
        output = self.transformer_encoder(src)
        output = output.permute(0, 2, 1)  # (batch_size, d_model, seq_length)
        output = self.global_avg_pool(output).squeeze(-1)  # (batch_size, d_model)
        return output


class AttenLstmPosSpaceScale(nn.Module):
    """
    Experiment 2024071801 (Added attention module)
    """
    def __init__(self, dropout_rate=0.1):
        super(AttenLstmPosSpaceScale, self).__init__()
        self.lstm1 = TimeSeriesAttnLSTM(hidden_dim=64)  # Output dim 64
        self.lstm2 = TimeSeriesAttnLSTM(hidden_dim=64)  # Output dim 64
        self.fc1 = nn.Linear(128 + 500, 1024)  # Total dim of 4 LSTM outputs and pos_feature
        self.lkrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, 256)
        self.fc7 = nn.Linear(256, 64)
        self.fc8 = nn.Linear(64, 1)
        self.pos_fc1 = nn.Linear(4, 100)
        self.pos_fc2 = nn.Linear(100, 200)
        self.pos_fc3 = nn.Linear(200, 500)

    def forward(self, x):
        ts_features1 = self.lstm1(x[:, 4:53].unsqueeze(-1))
        ts_features2 = self.lstm2(x[:, 53:102].unsqueeze(-1))
        first_four = x[:, :4]  # First 4 columns
        pos_feature = self.pos_fc1(first_four)
        pos_feature = self.relu(pos_feature)
        pos_feature = self.pos_fc2(pos_feature)
        pos_feature = self.relu(pos_feature)
        pos_feature = self.pos_fc3(pos_feature)
        combined_features = torch.cat([ts_features1, ts_features2, pos_feature], dim=1)
        x = self.fc1(combined_features)
        x = self.lkrelu(x)
        x = self.fc2(x)
        x = self.lkrelu(x)
        x = self.fc3(x)
        x = self.lkrelu(x)
        x = self.fc4(x)
        x = self.lkrelu(x)
        x = self.fc5(x)
        x = self.lkrelu(x)
        x = self.fc6(x)
        x = self.lkrelu(x)
        x = self.dropout(x)  
        x = self.fc7(x)
        x = self.lkrelu(x)
        x = self.dropout(x) 
        x = self.fc8(x)
        return x
