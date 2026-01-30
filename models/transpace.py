import torch
import torch.nn as nn
import torch.nn.functional as F

# 注意力机制实现
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_output):
        scores = self.attention(lstm_output)  # (batch_size, seq_length, 1)
        attention_weights = F.softmax(scores, dim=1)  # (batch_size, seq_length, 1)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)  # (batch_size, hidden_dim)
        return context_vector, attention_weights

# TimeSeriesLSTM实现
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
        # LSTM的输出是 (batch_size, seq_length, hidden_dim)
        lstm_out, _ = self.lstm(x)
        # 全局平均池化
        lstm_out = lstm_out.permute(0, 2, 1)  # 调整维度为(batch_size, hidden_dim, seq_length)
        pooled_output = self.global_avg_pool(lstm_out).squeeze(-1)  # (batch_size, hidden_dim)
        return pooled_output


class TimeSeriesTransformer(nn.Module):
    def __init__(self, seq_length, d_model=64, nhead=8, num_encoder_layers=3, dim_feedforward=256, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.embedding = nn.Linear(1, d_model)
        self.position_encoding = nn.Parameter(torch.zeros(1, seq_length, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # 全局平均池化

    def forward(self, src):
        src = src.unsqueeze(-1)  # 确保输入为(batch_size, seq_length, 1)
        src = self.embedding(src) + self.position_encoding
        output = self.transformer_encoder(src)
        output = output.permute(0, 2, 1)  # 调整维度以适应池化层 (batch_size, d_model, seq_length)
        output = self.global_avg_pool(output).squeeze(-1)  # (batch_size, d_model)
        return output
    
class TranPosSpace(nn.Module):
    """
    试验编号  transformer_kanpos_3407
    """
    def __init__(self, dropout_rate = 0.1):
        super(TranPosSpace, self).__init__()
        self.transformer1 = TimeSeriesTransformer(seq_length=49)  # 用于第5-53列
        self.transformer2 = TimeSeriesTransformer(seq_length=49)  # 用于第54-102列
        self.fc1 = nn.Linear(128 + 500, 1024)  # Transformer的两个输出和前4列数据
        self.lkrelu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, 256)
        self.fc7 = nn.Linear(256, 64)
        self.fc8 = nn.Linear(64, 1)
        self.pos_fc1 = nn.Linear(4,100)
        self.pos_fc2 = nn.Linear(100,200)
        self.pos_fc3 = nn.Linear(200,500)

    def forward(self, x):
        # 假设x已经是适当的格式
        ts_features1 = self.transformer1(x[:, 4:53])  # 提取第5-53列特征
        ts_features2 = self.transformer2(x[:, 53:102])  # 提取第54-102列特征
        first_four = x[:, :4]  # 前4列
        pos_feature = self.pos_fc1(first_four)
        pos_feature = self.lkrelu(pos_feature)
        pos_feature = self.pos_fc2(pos_feature)
        pos_feature = self.lkrelu(pos_feature)
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
        x = self.dropout(x)  # 应用 Dropout
        x = self.fc7(x)
        x = self.lkrelu(x)
        x = self.dropout(x)  # 应用 Dropout
        x = self.fc8(x)
        return x
    
class KranPosSpace(nn.Module):
    def __init__(self, dropout_rate = 0.1):
        """
        试验编号  2024062702， 
        """
        super(KranPosSpace, self).__init__()
        self.transformer1 = TimeSeriesTransformer(seq_length=49)  # 用于第5-53列
        self.transformer2 = TimeSeriesTransformer(seq_length=49)  # 用于第54-102列
        self.kan1 = KAN([4, 64, 500])
        self.kan2 = KAN([128 + 500, 64, 1])

    def forward(self, x):
        # 假设x已经是适当的格式
        ts_features1 = self.transformer1(x[:, 4:53])  # 提取第5-53列特征
        ts_features2 = self.transformer2(x[:, 53:102])  # 提取第54-102列特征
        first_four = x[:, :4]  # 前4列
        pos_feature = self.kan1(first_four)
        combined_features = torch.cat([ts_features1, ts_features2, pos_feature], dim=1)
        x = self.kan2(combined_features)
        return x


class LstmPosSpace(nn.Module):
    """
    试验编号 2024062702
    """
    def __init__(self, dropout_rate=0.1):
        super(LstmPosSpace, self).__init__()
        self.lstm1 = TimeSeriesLSTM(hidden_dim=64)  # 输出维度64
        self.lstm2 = TimeSeriesLSTM(hidden_dim=64)  # 输出维度64
        self.pos_fc1 = nn.Linear(4, 100)
        self.pos_fc2 = nn.Linear(100, 200)
        self.pos_fc3 = nn.Linear(200, 500)
        self.fc1 = nn.Linear(128 + 500, 1024)
        self.layers = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        ts_features1 = self.lstm1(x[:, 4:53].unsqueeze(-1))  # 确保维度正确
        ts_features2 = self.lstm2(x[:, 53:102].unsqueeze(-1))  # 确保维度正确
        first_four = x[:, :4]
        pos_feature = self.pos_fc1(first_four)
        pos_feature = nn.LeakyReLU()(pos_feature)
        pos_feature = self.pos_fc2(pos_feature)
        pos_feature = nn.LeakyReLU()(pos_feature)
        pos_feature = self.pos_fc3(pos_feature)
        combined_features = torch.cat([ts_features1, ts_features2, pos_feature], dim=1)
        x = self.fc1(combined_features)
        return self.layers(x)
    
class KstmPosSpace(nn.Module):
    """
    试验编号 2024062703
    """
    def __init__(self, dropout_rate=0.1):
        super(KstmPosSpace, self).__init__()
        self.lstm1 = TimeSeriesLSTM(hidden_dim=64)  # 输出维度64
        self.lstm2 = TimeSeriesLSTM(hidden_dim=64)  # 输出维度64
        self.kan1 = KAN([4, 64, 500])
        self.kan2 = KAN([128 + 500, 64, 1])
        self.fc1 = nn.Linear(128 + 500, 1024)
    
    def forward(self, x):
        ts_features1 = self.lstm1(x[:, 4:53].unsqueeze(-1))  # 确保维度正确
        ts_features2 = self.lstm2(x[:, 53:102].unsqueeze(-1))  # 确保维度正确
        first_four = x[:, :4]
        pos_feature = self.kan1(first_four)
        combined_features = torch.cat([ts_features1, ts_features2, pos_feature], dim=1)
        x = self.kan2(combined_features)
        return x
    
class TranPosSpaceScale(nn.Module):
    """
    试验编号 2024070401     多尺度数据
    """
    def __init__(self, dropout_rate = 0.1):
        super(TranPosSpaceScale, self).__init__()
        # 新数据 ，高分辨率的时间序列
        self.transformer1 = TimeSeriesTransformer(seq_length=49)  # 用于第5-53列
        self.transformer2 = TimeSeriesTransformer(seq_length=49)  # 用于第54-102列
        # 旧数据，低分辨率的时间序列
        self.transformer3 = TimeSeriesTransformer(seq_length=61)  # 用于第5-65列
        self.transformer4 = TimeSeriesTransformer(seq_length=61)  # 用于第66-126列

        self.fc1 = nn.Linear(128 + 128 + 500, 1024)  # Transformer的两个输出和前4列数据
        self.lkrelu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, 256)
        self.fc7 = nn.Linear(256, 64)
        self.fc8 = nn.Linear(64, 1)
        self.pos_fc1 = nn.Linear(4,100)
        self.pos_fc2 = nn.Linear(100,200)
        self.pos_fc3 = nn.Linear(200,500)

    def forward(self, x):
        # 假设x已经是适当的格式
        ts_features1 = self.transformer1(x[:, 4:53])  # 提取第5-53列特征
        ts_features2 = self.transformer2(x[:, 53:102])  # 提取第54-102列特征
        ts_features3 = self.transformer3(x[:, 102:163])  # 提取第5-53列特征
        ts_features4 = self.transformer4(x[:, 163:])  # 提取第54-102列特征
        first_four = x[:, :4]  # 前4列
        pos_feature = self.pos_fc1(first_four)
        pos_feature = self.lkrelu(pos_feature)
        pos_feature = self.pos_fc2(pos_feature)
        pos_feature = self.lkrelu(pos_feature)
        pos_feature = self.pos_fc3(pos_feature)
        combined_features = torch.cat([ts_features1, ts_features2, ts_features3, ts_features4, pos_feature], dim=1)
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
        x = self.dropout(x)  # 应用 Dropout
        x = self.fc7(x)
        x = self.lkrelu(x)
        x = self.dropout(x)  # 应用 Dropout
        x = self.fc8(x)
        return x
    
class LstmPosSpaceScale(nn.Module):
    """
    试验编号 2024070402(多尺度数据+ LSTM)  2024070701(数据做了均衡化处理)  2024071101（另一种均衡化处理方案）
    """
    def __init__(self, dropout_rate = 0.1):
        super(LstmPosSpaceScale, self).__init__()
        self.lstm1 = TimeSeriesLSTM(hidden_dim=64)  # 输出维度64
        self.lstm2 = TimeSeriesLSTM(hidden_dim=64)  # 输出维度64
        # 低分辨率的时间序列
        self.lstm3 = TimeSeriesLSTM(hidden_dim=64, seq_length=61)  # 输出维度64
        self.lstm5 = TimeSeriesLSTM(hidden_dim=64, seq_length=61)  # 输出维度64
        self.fc1 = nn.Linear(128 + 128 + 500, 1024)  # Transformer的两个输出和前4列数据
        self.lkrelu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, 256)
        self.fc7 = nn.Linear(256, 64)
        self.fc8 = nn.Linear(64, 1)
        self.pos_fc1 = nn.Linear(4,100)
        self.pos_fc2 = nn.Linear(100,200)
        self.pos_fc3 = nn.Linear(200,500)

    def forward(self, x):
        # 假设x已经是适当的格式
        ts_features1 = self.lstm1(x[:, 4:53].unsqueeze(-1))  # 确保维度正确
        ts_features2 = self.lstm2(x[:, 53:102].unsqueeze(-1))  # 确保维度正确
        ts_features3 = self.lstm3(x[:, 102:163].unsqueeze(-1))  # 确保维度正确
        ts_features4 = self.lstm5(x[:, 163:].unsqueeze(-1))  # 确保维度正确
        first_four = x[:, :4]  # 前4列
        pos_feature = self.pos_fc1(first_four)
        pos_feature = self.lkrelu(pos_feature)
        pos_feature = self.pos_fc2(pos_feature)
        pos_feature = self.lkrelu(pos_feature)
        pos_feature = self.pos_fc3(pos_feature)
        combined_features = torch.cat([ts_features1, ts_features2, ts_features3, ts_features4, pos_feature], dim=1)
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
    
class KstmScalePosSpace(nn.Module):
    """
    试验编号 2024070403   LSTM + KAN + 多尺度数据
    """
    def __init__(self, dropout_rate=0.1):
        super(KstmScalePosSpace, self).__init__()
        self.lstm1 = TimeSeriesLSTM(hidden_dim=64)  # 输出维度64
        self.lstm2 = TimeSeriesLSTM(hidden_dim=64)  # 输出维度64
        # 低分辨率的时间序列
        self.lstm3 = TimeSeriesLSTM(hidden_dim=64, seq_length=61)  # 输出维度64
        self.lstm4 = TimeSeriesLSTM(hidden_dim=64, seq_length=61)  # 输出维度64
        self.kan1 = KAN([4, 64, 500])
        self.kan2 = KAN([128 + 128 + 500, 64, 1])
        self.fc1 = nn.Linear(128 + 500, 1024)
    
    def forward(self, x):
        ts_features1 = self.lstm1(x[:, 4:53].unsqueeze(-1))  # 确保维度正确
        ts_features2 = self.lstm2(x[:, 53:102].unsqueeze(-1))  # 确保维度正确
        ts_features3 = self.lstm3(x[:, 102:163].unsqueeze(-1))  # 确保维度正确
        ts_features4 = self.lstm4(x[:, 163:].unsqueeze(-1))  # 确保维度正确
        first_four = x[:, :4]
        pos_feature = self.kan1(first_four)
        combined_features = torch.cat([ts_features1, ts_features2, ts_features3, ts_features4, pos_feature], dim=1)
        x = self.kan2(combined_features)
        return x

# 主模型
class AttenLstmPosSpaceScale(nn.Module):
    """
    试验编号 2024071801(添加attention模块)
    """
    def __init__(self, dropout_rate=0.1):
        super(AttenLstmPosSpaceScale, self).__init__()
        self.lstm1 = TimeSeriesAttnLSTM(hidden_dim=64)  # 输出维度64
        self.lstm2 = TimeSeriesAttnLSTM(hidden_dim=64)  # 输出维度64
        self.lstm3 = TimeSeriesAttnLSTM(hidden_dim=64, seq_length=61)  # 输出维度64
        self.lstm4 = TimeSeriesAttnLSTM(hidden_dim=64, seq_length=61)  # 输出维度64
        self.fc1 = nn.Linear(256 + 500, 1024)  # 4个LSTM输出和pos_feature的总维度
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
        # 假设 x 已经是适当的格式
        ts_features1 = self.lstm1(x[:, 4:53].unsqueeze(-1))  # (batch_size, 49, 1)
        ts_features2 = self.lstm2(x[:, 53:102].unsqueeze(-1))  # (batch_size, 49, 1)
        ts_features3 = self.lstm3(x[:, 102:163].unsqueeze(-1))  # (batch_size, 61, 1)
        ts_features4 = self.lstm4(x[:, 163:].unsqueeze(-1))  # (batch_size, 61, 1)
        first_four = x[:, :4]  # 前4列
        pos_feature = self.pos_fc1(first_four)
        pos_feature = self.relu(pos_feature)
        pos_feature = self.pos_fc2(pos_feature)
        pos_feature = self.relu(pos_feature)
        pos_feature = self.pos_fc3(pos_feature)
        combined_features = torch.cat([ts_features1, ts_features2, ts_features3, ts_features4, pos_feature], dim=1)
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
    
class AttenKANLstmPosSpaceScale(nn.Module):
    """
    试验编号 2024071802(添加attention模块，结合KAN)
    """
    def __init__(self, dropout_rate=0.1):
        super(AttenKANLstmPosSpaceScale, self).__init__()
        self.lstm1 = TimeSeriesAttnLSTM(hidden_dim=64)  # 输出维度64
        self.lstm2 = TimeSeriesAttnLSTM(hidden_dim=64)  # 输出维度64
        self.lstm3 = TimeSeriesAttnLSTM(hidden_dim=64, seq_length=61)  # 输出维度64
        self.lstm4 = TimeSeriesAttnLSTM(hidden_dim=64, seq_length=61)  # 输出维度64
        self.kan1 = KAN([4, 64, 500])
        self.kan2 = KAN([128 + 128 + 500, 64, 1])
        self.fc1 = nn.Linear(128 + 500, 1024)

    def forward(self, x):
        # 假设 x 已经是适当的格式
        ts_features1 = self.lstm1(x[:, 4:53].unsqueeze(-1))  # (batch_size, 49, 1)
        ts_features2 = self.lstm2(x[:, 53:102].unsqueeze(-1))  # (batch_size, 49, 1)
        ts_features3 = self.lstm3(x[:, 102:163].unsqueeze(-1))  # (batch_size, 61, 1)
        ts_features4 = self.lstm4(x[:, 163:].unsqueeze(-1))  # (batch_size, 61, 1)
        first_four = x[:, :4]
        pos_feature = self.kan1(first_four)
        combined_features = torch.cat([ts_features1, ts_features2, ts_features3, ts_features4, pos_feature], dim=1)
        x = self.kan2(combined_features)
        return x