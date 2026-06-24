import torch
import numpy as np

class RewardCalculator:
    def __init__(self, grp=None, pts=None, uniform_init=False, alpha=1.0, gamma_pt=1.0):
        self.device = torch.device('cpu')
        self.grp = grp.to(self.device).eval()
        self.uniform_init = uniform_init
        self.alpha = alpha
        self.gamma_pt = gamma_pt

        pts = pts or [3, 1, -1, -3]
        self.pts = torch.tensor(pts, dtype=torch.float64, device=self.device)

    def calc_grp(self, grp_feature):
        seq = list(map(
            lambda idx: torch.as_tensor(grp_feature[:idx+1], device=self.device),
            range(len(grp_feature)),
        ))

        with torch.inference_mode():
            logits = self.grp(seq)
        matrix = self.grp.calc_matrix(logits)
        return matrix

    def calc_rank_prob(self, player_id, grp_feature, rank_by_player):
        matrix = self.calc_grp(grp_feature)

        final_ranking = torch.zeros((1, 4), device=self.device)
        final_ranking[0, rank_by_player[player_id]] = 1.
        rank_prob = torch.cat((matrix[:, player_id], final_ranking))
        if self.uniform_init:
            rank_prob[0, :] = 1 / 4
        return rank_prob

    def calc_delta_pt(self, player_id, grp_feature, rank_by_player):
        rank_prob = self.calc_rank_prob(player_id, grp_feature, rank_by_player)
        exp_pts = rank_prob @ self.pts
        reward = exp_pts[1:] - exp_pts[:-1]
        return reward.cpu().numpy()

    def calc_delta_points(self, player_id, grp_feature, final_scores):
        seq = np.concatenate((grp_feature[:, 3 + player_id] * 1e4, [final_scores[player_id]]))
        delta_points = seq[1:] - seq[:-1]
        return delta_points

    def calc_delta_blend(self, player_id, grp_feature, rank_by_player, final_scores,
                         alpha=1.0, gamma_pt=1.0, chip_deltas=None, beta=0.0, chip_value=5.0,
                         aka_held=None, tenpai_end=None, won=None, dealt_in=None,
                         lambda_opp=0.0, noten_factor=0.0):
        sotensu = self.calc_delta_points(player_id, grp_feature, final_scores) / 1000.0
        juni = self.calc_delta_pt(player_id, grp_feature, rank_by_player)
        assert len(sotensu) == len(juni), f"length mismatch: sotensu={len(sotensu)}, juni={len(juni)}"
        reward = alpha * sotensu + gamma_pt * juni
        if chip_deltas is not None:
            assert len(chip_deltas) == len(reward), (
                f"length mismatch: chip_deltas={len(chip_deltas)}, reward={len(reward)}"
            )
            reward = reward + beta * chip_deltas * chip_value
        if (
            lambda_opp > 0
            and aka_held is not None
            and tenpai_end is not None
            and won is not None
            and dealt_in is not None
        ):
            n = len(reward)
            assert len(aka_held) == n, f"length mismatch: aka_held={len(aka_held)}, reward={n}"
            assert len(tenpai_end) == n, f"length mismatch: tenpai_end={len(tenpai_end)}, reward={n}"
            assert len(won) == n, f"length mismatch: won={len(won)}, reward={n}"
            assert len(dealt_in) == n, f"length mismatch: dealt_in={len(dealt_in)}, reward={n}"
            w = np.where(tenpai_end, 1.0, noten_factor)
            fire = (won == 0) & (dealt_in == 0) & (aka_held > 0)
            opp = -beta * lambda_opp * chip_value * aka_held * w * fire
            reward = reward + opp
        return reward
