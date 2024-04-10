import torch
import torch.nn as nn
import torch.nn.functional as F

def calc_mean_std(x, eps=1e-8):
        """
        calculating channel-wise instance mean and standard variance
        x: shape of (N,C,*)
        """
        mean = torch.mean(x.flatten(2), dim=-1, keepdim=True) # size of (N, C, 1)
        std = torch.std(x.flatten(2), dim=-1, keepdim=True) + eps # size of (N, C, 1)

        return mean, std

def cal_adain_style_loss(x, y):
    """
    style loss in one layer

    Args:
        x, y: feature maps of size [N, C, H, W]
    """
    x_mean, x_std = calc_mean_std(x)
    y_mean, y_std = calc_mean_std(y)

    return nn.functional.mse_loss(x_mean, y_mean) \
         + nn.functional.mse_loss(x_std, y_std)

def cal_mse_content_loss(x, y):
    return nn.functional.mse_loss(x, y)

class LearnableIN(nn.Module):
    '''
    Input: (N, C, L) or (C, L)
    '''
    def __init__(self, dim=256):
        super().__init__()
        self.IN = torch.nn.InstanceNorm1d(dim, momentum=1e-4, track_running_stats=True)

    def forward(self, x):
        if x.size()[-1] <= 1:
            return x
        return self.IN(x)

class SimpleLinearStylizer(nn.Module):
    def __init__(self, input_dim=256, embed_dim=32, n_layers=3) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        self.IN = LearnableIN(input_dim)

        self.q_embed = nn.Conv1d(input_dim, embed_dim, 1)
        self.k_embed = nn.Conv1d(input_dim, embed_dim, 1)
        self.v_embed = nn.Conv1d(input_dim, embed_dim, 1)

        self.unzipper = nn.Conv1d(embed_dim, input_dim, 1, bias=0)

        self.mapping = Mapping(512, 512, 1024, 6)
        self.mapping.load_state_dict(torch.load('./configs/MLP.pth'), strict=False)

        s_net = []
        for i in range(n_layers - 1):
            out_dim = max(embed_dim, input_dim // 2)
            s_net.append(
                nn.Sequential(
                    nn.Conv1d(input_dim, out_dim, 1),
                    nn.ReLU(inplace=True),
                )
            )
            input_dim = out_dim
        s_net.append(nn.Conv1d(input_dim, embed_dim, 1))
        self.s_net = nn.Sequential(*s_net)
        self.s_net_1 = nn.Sequential(*s_net)

        self.s_fc = nn.Linear(embed_dim ** 2, embed_dim ** 2)
        self.s_fc_1 = nn.Linear(embed_dim ** 2, embed_dim ** 2)


    def _vectorized_covariance(self, x):
        cov = torch.bmm(x, x.transpose(2, 1)) / x.size(-1)
        cov = cov.flatten(1)
        return cov

    def get_content_matrix(self, c):
        '''
        Args:
            c: content feature [N,input_dim,S]
        Return:
            mat: [N,S,embed_dim,embed_dim]
        '''
        normalized_c = self.IN(c)

        q_embed = self.q_embed(normalized_c)
        k_embed = self.k_embed(normalized_c)
        
        c_cov = q_embed.transpose(1,2).unsqueeze(3) * k_embed.transpose(1,2).unsqueeze(2) # [N,S,embed_dim,embed_dim]
        attn = torch.softmax(c_cov, -1) # [N,S,embed_dim,embed_dim]

        return attn, normalized_c

    def transform_content_3D(self, c):
        '''
        Args:
            c: content feature [N,input_dim,S]
        Return:
            transformed_c: [N,embed_dim,S]
        '''
        attn, normalized_c = self.get_content_matrix(c) # [N,S,embed_dim,embed_dim]
        c = self.v_embed(normalized_c) # [N,embed_dim,S]
        c = c.transpose(1,2).unsqueeze(3) # [N,S,embed_dim,1]
        c = torch.matmul(attn, c).squeeze(3) # [N,S,embed_dim]

        return c.transpose(1,2)

    def get_style_mean_std_matrix(self, s):
        '''
        Args:
            s: style feature [N,input_dim,S]

        Return:
            mat: [N,embed_dim,embed_dim]
        '''
        s_mean = s.mean(-1, keepdim=True)
        s_std = s.std(-1, keepdim=True)
        s = s - s_mean

        s_embed = self.s_net(s)
        s_cov = self._vectorized_covariance(s_embed)
        s_cov = self.s_fc(s_cov)
        s_mat = s_cov.reshape(-1, self.embed_dim, self.embed_dim)

        return s_mean, s_std, s_mat

    def get_clip_mean_std_matrix(self, f):
        feature, feature_1 = self.mapping(f)
        s_mean, s_std = feature.contiguous().unsqueeze(-1).chunk(2, 1)
        # s_std, s_mean = feature.contiguous().unsqueeze(-1).chunk(2, 1)

        s = feature_1[..., None] - s_mean

        s_embed = self.s_net_1(s)
        s_cov = self._vectorized_covariance(s_embed)
        s_cov = self.s_fc_1(s_cov)
        s_mat = s_cov.reshape(-1, self.embed_dim, self.embed_dim)

        return s_mean, s_std, s_mat

    def transfer_style_2D(self, s_mean_std_mat, c, acc_map):
        '''
        Agrs:
            c: content feature map after volume rendering [N,embed_dim,S]
            s_mat: style matrix [N,embed_dim,embed_dim]
            acc_map: [S]
            
            s_mean = [N,input_dim,1]
            s_std = [N,input_dim,1]
        '''
        s_mean, s_std, s_mat = s_mean_std_mat

        cs = torch.bmm(s_mat, c) # [N,embed_dim,S]
        cs = self.unzipper(cs) # [N,input_dim,S]

        cs = cs * s_std + s_mean * acc_map[None,None,...]

        return cs

class Mapping(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers) -> None:
        super(Mapping, self).__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.last_fc = nn.Linear(output_dim, input_dim)
        self.feature_fc = nn.Linear(hidden_dim, output_dim // 4)

    def forward(self, x):
        x1 = None
        for i, layer in enumerate(self.layers):
            if i == self.num_layers - 1:
                x1 = x
            x = F.relu(layer(x.float()))
        x = self.last_fc(x)
        x1 = self.feature_fc(x1)
        return x, x1