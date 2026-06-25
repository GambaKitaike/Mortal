import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from typing import *
from functools import partial
from itertools import permutations
from libriichi.consts import obs_shape, oracle_obs_shape, ACTION_SPACE, GRP_SIZE

class ChannelAttention(nn.Module):
    def __init__(self, channels, ratio=16, actv_builder=nn.ReLU, bias=True):
        super().__init__()
        self.shared_mlp = nn.Sequential(
            nn.Linear(channels, channels // ratio, bias=bias),
            actv_builder(),
            nn.Linear(channels // ratio, channels, bias=bias),
        )
        if bias:
            for mod in self.modules():
                if isinstance(mod, nn.Linear):
                    nn.init.constant_(mod.bias, 0)

    def forward(self, x: Tensor):
        avg_out = self.shared_mlp(x.mean(-1))
        max_out = self.shared_mlp(x.amax(-1))
        weight = (avg_out + max_out).sigmoid()
        x = weight.unsqueeze(-1) * x
        return x

class ResBlock(nn.Module):
    def __init__(
        self,
        channels,
        *,
        norm_builder = nn.Identity,
        actv_builder = nn.ReLU,
        pre_actv = False,
    ):
        super().__init__()
        self.pre_actv = pre_actv

        if pre_actv:
            self.res_unit = nn.Sequential(
                norm_builder(),
                actv_builder(),
                nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
                norm_builder(),
                actv_builder(),
                nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
            )
        else:
            self.res_unit = nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
                norm_builder(),
                actv_builder(),
                nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
                norm_builder(),
            )
            self.actv = actv_builder()
        self.ca = ChannelAttention(channels, actv_builder=actv_builder, bias=True)

    def forward(self, x):
        out = self.res_unit(x)
        out = self.ca(out)
        out = out + x
        if not self.pre_actv:
            out = self.actv(out)
        return out

class ResNet(nn.Module):
    def __init__(
        self,
        in_channels,
        conv_channels,
        num_blocks,
        *,
        norm_builder = nn.Identity,
        actv_builder = nn.ReLU,
        pre_actv = False,
    ):
        super().__init__()

        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResBlock(
                conv_channels,
                norm_builder = norm_builder,
                actv_builder = actv_builder,
                pre_actv = pre_actv,
            ))

        layers = [nn.Conv1d(in_channels, conv_channels, kernel_size=3, padding=1, bias=False)]
        if pre_actv:
            layers += [*blocks, norm_builder(), actv_builder()]
        else:
            layers += [norm_builder(), actv_builder(), *blocks]
        layers += [
            nn.Conv1d(conv_channels, 32, kernel_size=3, padding=1),
            actv_builder(),
            nn.Flatten(),
            nn.Linear(32 * 34, 1024),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class Brain(nn.Module):
    def __init__(self, *, conv_channels, num_blocks, is_oracle=False, version=1):
        super().__init__()
        self.is_oracle = is_oracle
        self.version = version

        in_channels = obs_shape(version)[0]
        if is_oracle:
            in_channels += oracle_obs_shape(version)[0]

        norm_builder = partial(nn.BatchNorm1d, conv_channels, momentum=0.01)
        actv_builder = partial(nn.Mish, inplace=True)
        pre_actv = True

        match version:
            case 1:
                actv_builder = partial(nn.ReLU, inplace=True)
                pre_actv = False
                self.latent_net = nn.Sequential(
                    nn.Linear(1024, 512),
                    nn.ReLU(inplace=True),
                )
                self.mu_head = nn.Linear(512, 512)
                self.logsig_head = nn.Linear(512, 512)
            case 2:
                pass
            case 3 | 4:
                norm_builder = partial(nn.BatchNorm1d, conv_channels, momentum=0.01, eps=1e-3)
            case _:
                raise ValueError(f'Unexpected version {self.version}')

        self.encoder = ResNet(
            in_channels = in_channels,
            conv_channels = conv_channels,
            num_blocks = num_blocks,
            norm_builder = norm_builder,
            actv_builder = actv_builder,
            pre_actv = pre_actv,
        )
        self.actv = actv_builder()

        # always use EMA or CMA when True
        self._freeze_bn = False

    def forward(self, obs: Tensor, invisible_obs: Optional[Tensor] = None) -> Union[Tuple[Tensor, Tensor], Tensor]:
        if self.is_oracle:
            assert invisible_obs is not None
            obs = torch.cat((obs, invisible_obs), dim=1)
        phi = self.encoder(obs)

        match self.version:
            case 1:
                latent_out = self.latent_net(phi)
                mu = self.mu_head(latent_out)
                logsig = self.logsig_head(latent_out)
                return mu, logsig
            case 2 | 3 | 4:
                return self.actv(phi)
            case _:
                raise ValueError(f'Unexpected version {self.version}')

    def train(self, mode=True):
        super().train(mode)
        if self._freeze_bn:
            for mod in self.modules():
                if isinstance(mod, nn.BatchNorm1d):
                    mod.eval()
                    # I don't think this benefits
                    # module.requires_grad_(False)
        return self

    def reset_running_stats(self):
        for mod in self.modules():
            if isinstance(mod, nn.BatchNorm1d):
                mod.reset_running_stats()

    def freeze_bn(self, value: bool):
        self._freeze_bn = value
        return self.train(self.training)

class AuxNet(nn.Module):
    def __init__(self, dims=None):
        super().__init__()
        self.dims = dims
        self.net = nn.Linear(1024, sum(dims), bias=False)

    def forward(self, x):
        return self.net(x).split(self.dims, dim=-1)

def dueling_q(v, a, mask):
    a_sum = a.masked_fill(~mask, 0.).sum(-1, keepdim=True)
    mask_sum = mask.sum(-1, keepdim=True)
    a_mean = a_sum / mask_sum
    return (v + a - a_mean).masked_fill(~mask, -torch.inf)

def q_total(q_main, q_chip, beta_sel):
    return q_main + beta_sel * q_chip

class DQN(nn.Module):
    def __init__(self, *, version=1):
        super().__init__()
        self.version = version
        match version:
            case 1:
                self.v_head = nn.Linear(512, 1)
                self.a_head = nn.Linear(512, ACTION_SPACE)
                self.v_chip_head = nn.Linear(512, 1)
                self.a_chip_head = nn.Linear(512, ACTION_SPACE)
            case 2 | 3:
                hidden_size = 512 if version == 2 else 256
                self.v_head = nn.Sequential(
                    nn.Linear(1024, hidden_size),
                    nn.Mish(inplace=True),
                    nn.Linear(hidden_size, 1),
                )
                self.a_head = nn.Sequential(
                    nn.Linear(1024, hidden_size),
                    nn.Mish(inplace=True),
                    nn.Linear(hidden_size, ACTION_SPACE),
                )
                self.v_chip_head = nn.Sequential(
                    nn.Linear(1024, hidden_size),
                    nn.Mish(inplace=True),
                    nn.Linear(hidden_size, 1),
                )
                self.a_chip_head = nn.Sequential(
                    nn.Linear(1024, hidden_size),
                    nn.Mish(inplace=True),
                    nn.Linear(hidden_size, ACTION_SPACE),
                )
            case 4:
                self.net = nn.Linear(1024, 1 + ACTION_SPACE)
                nn.init.constant_(self.net.bias, 0)
                self.chip_net = nn.Linear(1024, 1 + ACTION_SPACE)
                nn.init.constant_(self.chip_net.bias, 0)

    def _main_heads(self, phi):
        if self.version == 4:
            return self.net(phi).split((1, ACTION_SPACE), dim=-1)
        return self.v_head(phi), self.a_head(phi)

    def _chip_heads(self, phi):
        if self.version == 4:
            return self.chip_net(phi).split((1, ACTION_SPACE), dim=-1)
        return self.v_chip_head(phi), self.a_chip_head(phi)

    def forward(self, phi, mask, *, return_q_chip=False):
        v, a = self._main_heads(phi)
        q = dueling_q(v, a, mask)
        if not return_q_chip:
            return q
        v_c, a_c = self._chip_heads(phi)
        q_chip = dueling_q(v_c, a_c, mask)
        return q, q_chip

    def forward_chip(self, phi, mask):
        v_c, a_c = self._chip_heads(phi)
        return dueling_q(v_c, a_c, mask)

    def chip_state_dict(self):
        if self.version == 4:
            return {'chip_net': self.chip_net.state_dict()}
        return {
            'v_chip_head': self.v_chip_head.state_dict(),
            'a_chip_head': self.a_chip_head.state_dict(),
        }

    def load_chip_state_dict(self, state_dict, *, strict=True):
        if self.version == 4:
            self.chip_net.load_state_dict(state_dict['chip_net'], strict=strict)
        else:
            self.v_chip_head.load_state_dict(state_dict['v_chip_head'], strict=strict)
            self.a_chip_head.load_state_dict(state_dict['a_chip_head'], strict=strict)

class ChipDQNTarget(nn.Module):
    """Target network for Q_chip heads only; phi comes from the online Brain."""

    def __init__(self, dqn: DQN):
        super().__init__()
        self.version = dqn.version
        match dqn.version:
            case 1:
                self.v_chip_head = nn.Linear(512, 1)
                self.a_chip_head = nn.Linear(512, ACTION_SPACE)
            case 2 | 3:
                hidden_size = 512 if dqn.version == 2 else 256
                self.v_chip_head = nn.Sequential(
                    nn.Linear(1024, hidden_size),
                    nn.Mish(inplace=True),
                    nn.Linear(hidden_size, 1),
                )
                self.a_chip_head = nn.Sequential(
                    nn.Linear(1024, hidden_size),
                    nn.Mish(inplace=True),
                    nn.Linear(hidden_size, ACTION_SPACE),
                )
            case 4:
                self.chip_net = nn.Linear(1024, 1 + ACTION_SPACE)
                nn.init.constant_(self.chip_net.bias, 0)

    def forward(self, phi, mask):
        if self.version == 4:
            v, a = self.chip_net(phi).split((1, ACTION_SPACE), dim=-1)
        else:
            v = self.v_chip_head(phi)
            a = self.a_chip_head(phi)
        return dueling_q(v, a, mask)

    def copy_from(self, dqn: DQN):
        if dqn.version == 4:
            self.chip_net.load_state_dict(dqn.chip_net.state_dict())
        else:
            self.v_chip_head.load_state_dict(dqn.v_chip_head.state_dict())
            self.a_chip_head.load_state_dict(dqn.a_chip_head.state_dict())

    def polyak_update(self, dqn: DQN, tau: float):
        with torch.no_grad():
            if dqn.version == 4:
                for tgt, src in zip(self.chip_net.parameters(), dqn.chip_net.parameters()):
                    tgt.lerp_(src, tau)
            else:
                for tgt, src in zip(self.v_chip_head.parameters(), dqn.v_chip_head.parameters()):
                    tgt.lerp_(src, tau)
                for tgt, src in zip(self.a_chip_head.parameters(), dqn.a_chip_head.parameters()):
                    tgt.lerp_(src, tau)

    def state_dict_for_save(self):
        if self.version == 4:
            return {'chip_net': self.chip_net.state_dict()}
        return {
            'v_chip_head': self.v_chip_head.state_dict(),
            'a_chip_head': self.a_chip_head.state_dict(),
        }

    def load_from_saved(self, state_dict):
        if self.version == 4:
            self.chip_net.load_state_dict(state_dict['chip_net'])
        else:
            self.v_chip_head.load_state_dict(state_dict['v_chip_head'])
            self.a_chip_head.load_state_dict(state_dict['a_chip_head'])

class GRP(nn.Module):
    def __init__(self, hidden_size=64, num_layers=2):
        super().__init__()
        self.rnn = nn.GRU(input_size=GRP_SIZE, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * num_layers, hidden_size * num_layers),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size * num_layers, 24),
        )
        for mod in self.modules():
            mod.to(torch.float64)

        # perms are the permutations of all possible rank-by-player result
        perms = torch.tensor(list(permutations(range(4))))
        perms_t = perms.transpose(0, 1)
        self.register_buffer('perms', perms)     # (24, 4)
        self.register_buffer('perms_t', perms_t) # (4, 24)

    # input: [grand_kyoku, honba, kyotaku, s[0], s[1], s[2], s[3]]
    # grand_kyoku: E1 = 0, S4 = 7, W4 = 11
    # s is 2.5 at E1
    # s[0] is score of player id 0
    def forward(self, inputs: List[Tensor]):
        lengths = torch.tensor([t.shape[0] for t in inputs], dtype=torch.int64)
        inputs = pad_sequence(inputs, batch_first=True)
        packed_inputs = pack_padded_sequence(inputs, lengths, batch_first=True, enforce_sorted=False)
        return self.forward_packed(packed_inputs)

    def forward_packed(self, packed_inputs):
        _, state = self.rnn(packed_inputs)
        state = state.transpose(0, 1).flatten(1)
        logits = self.fc(state)
        return logits

    # (N, 24) -> (N, player, rank_prob)
    def calc_matrix(self, logits: Tensor):
        batch_size = logits.shape[0]
        probs = logits.softmax(-1)
        matrix = torch.zeros(batch_size, 4, 4, dtype=probs.dtype)
        for player in range(4):
            for rank in range(4):
                cond = self.perms_t[player] == rank
                matrix[:, player, rank] = probs[:, cond].sum(-1)
        return matrix

    # (N, 4) -> (N)
    def get_label(self, rank_by_player: Tensor):
        batch_size = rank_by_player.shape[0]
        perms = self.perms.expand(batch_size, -1, -1).transpose(0, 1)
        mappings = (perms == rank_by_player).all(-1).nonzero()

        labels = torch.zeros(batch_size, dtype=torch.int64, device=mappings.device)
        labels[mappings[:, 1]] = mappings[:, 0]
        return labels
