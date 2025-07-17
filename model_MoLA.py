# MoLA: Multimodal Recursive Neural Network for Molecular Property Prediction
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GINConv, global_mean_pool, BatchNorm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DNM(nn.Module):
    def __init__(self, input_size, out_size, M, device='cpu'):
        super(DNM, self).__init__()
        self.M = M
        self.input_size = input_size
        self.out_size = out_size

        m = torch.rand([out_size, M]).to(device)
        torch.nn.init.uniform_(m, a=-10.0, b=10.0)

        self.params = nn.ParameterDict({
            'm': nn.Parameter(m),
        })
        self.mlp = nn.Linear(input_size, out_size * M)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.mlp(x)
        x = self.act(x)
        x = x.view(*x.shape[:-1], self.out_size, self.M)
        A = torch.tanh(self.params['m'])
        O = torch.sum(x * A, -1)  # dendritic aggregation
        return O

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.input_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

class Encoder(nn.Module):
    """
    多模态 Encoder，包含一层 FirstLayer 和 (num_layers-1) 层相同的 MultimodalLayer。
    forward 返回 shape 为 [num_layers*3, batch_size, hidden_dim] 的融合输出序列。
    """
    def __init__(self, graph_dim, fp_dim, sm_vocab_size, hidden_dim, num_layers):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        
        # 第一层：接收原始输入维度
        self.first_layer = nn.ModuleDict({
            'graph_conv': GINConv(
                nn.Sequential(
                    nn.Linear(graph_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
            ),
            'fp_mlp': DNM(fp_dim + 768, hidden_dim, M=20),
            'sm_embedding': nn.Embedding(sm_vocab_size, hidden_dim, padding_idx=0),
            'sm_transformer': nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8),
                num_layers=1
            ),
            'dropout': nn.Dropout(0.3),
        })
        
        # 后续层：维度均为 hidden_dim
        self.layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.layers.append(nn.ModuleDict({
                'graph_conv': GINConv(
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim)
                    )
                ),
                'fp_mlp': DNM(hidden_dim, hidden_dim, M=20),
                'sm_transformer': nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8),
                    num_layers=1
                ),
                'dropout': nn.Dropout(0.3),
            }))
    
    def forward(self, x, edge_index, fp, fp2, sm, batch):
        fused_out_list = []
        
        # 第一层处理
        l0 = self.first_layer
        graph_x = F.relu(l0['graph_conv'](x, edge_index))
        graph_feat = global_mean_pool(graph_x, batch)
        
        batch_size = batch.max().item() + 1
        fp_flat = fp.view(batch_size, -1)
        fp2_flat = fp2.view(batch_size, -1)
        fp_in = torch.cat([fp_flat, fp2_flat], dim=1)
        fp_feat = F.relu(l0['fp_mlp'](fp_in))
        
        sm_embed = l0['sm_embedding'](sm)              # [B, L, H]
        sm_x = l0['sm_transformer'](sm_embed)         # [B, L, H]
        sm_x = F.relu(sm_x)
        sm_feat = sm_x.mean(dim=1)                    # [B, H]
        
        # dropout
        graph_x = l0['dropout'](graph_x)
        fp_feat = l0['dropout'](fp_feat)
        sm_x = l0['dropout'](sm_x)
        
        fused_out_list.append(torch.stack([graph_feat, fp_feat, sm_feat], dim=0))
        
        # 后续层递归处理
        h_x, h_fp, h_sm = graph_x, fp_feat, sm_x
        for layer in self.layers:
            gx = F.relu(layer['graph_conv'](h_x, edge_index))
            gf = global_mean_pool(gx, batch)
            
            ff = F.relu(layer['fp_mlp'](h_fp))
            
            sx = layer['sm_transformer'](h_sm)
            sx = F.relu(sx)
            sf = sx.mean(dim=1)
            
            # dropout
            gx = layer['dropout'](gx)
            ff = layer['dropout'](ff)
            sx = layer['dropout'](sx)
            
            fused_out_list.append(torch.stack([gf, ff, sf], dim=0))
            h_x, h_fp, h_sm = gx, ff, sx
        
        # 将所有层的输出在第 0 维拼接
        # 结果 shape: [num_layers*3, batch_size, hidden_dim]
        return torch.cat(fused_out_list, dim=0)


class MoLA(nn.Module):
    """
    MoLA 主干模型：先通过 Encoder 提取各层模态融合特征，再做跨层多头注意力、加权融合并输出最终预测。
    """
    def __init__(self, graph_dim, fp_dim, sm_vocab_size, hidden_dim, output_dim, num_layers):
        super(MoLA, self).__init__()
        self.encoder = Encoder(graph_dim, fp_dim, sm_vocab_size, hidden_dim, num_layers)
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8)
        
        # 融合后输出层
        self.out_layer_final = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4 * hidden_dim, output_dim),
        )
        # 每个通道（层×3）加权参数
        self.layer_weights = nn.Parameter(torch.ones(num_layers * 3, 1, 1))
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        fp, fp2, sm, batch = data.fp, data.fp2, data.sm, data.batch
        
        # 得到所有层的融合输出 [T, B, H]，T = num_layers * 3
        fused_all = self.encoder(x, edge_index, fp, fp2, sm, batch)
        
        # 跨层多头注意力
        attn_out, attn_map = self.cross_attention(fused_all, fused_all, fused_all)
        # 加权求和融合
        fused_feat = (attn_out * self.layer_weights).sum(dim=0)  # [B, H]
        
        # 最终输出
        fused_out_final = self.out_layer_final(fused_feat)       # [B, output_dim]
        
        # 返回值顺序与 MultimodalRecursiveNet 完全一致：
        # [ fused_feature, layer_weights.view(-1,1,1), attn_map, fused_out_final ]
        return [
            fused_feat,
            self.layer_weights.view(-1, 1, 1),
            attn_map,
            fused_out_final
        ]