# β=1 収支 / avg_rank サルベージ調査 (2026-07-01)

## 目的

β=1（正しいチップ経済: `chip_value=5.0` → **1枚=5000点**）run から、過去ログで収支・avg_rank を掘り起こせるかを判定する。新規学習・GPU 実験は行っていない。

## 換算前提（明示）

| 項目 | 値 |
|---|---|
| チップ換算 | 1枚 = 5000点（`chip_value=5.0` 千点単位） |
| ウマオカ | `[+35, +5, −15, −25]` 千点（= +35000 / +5000 / −15000 / −25000 点） |
| 持ち点 | 25000（Stat の `point = final_score − 25000` と一致） |
| 局収支 | `hora/ryukyoku deltas[player]` + `chip_delta × 5000` |
| 半荘収支 | 素点収支（`final_score − 25000`）+ チップ収支 + ウマオカ |

---

## Step 1: β=1 run / ckpt / ログの存在

### `/home/gamba/mahjong/runs/*/config.toml` 横断

`grep -rn "beta" runs/*/config.toml` の結果、**online 系 8 run はすべて `beta = 0.3`**。`beta = 1.0` の run config は **0 件**。

β=1 相当は Phase4 オフライン run のみ（config は `freeparlor/configs/phase4_chip_beta1_*.toml` および `runs/phase4_chip_beta1_64x10.toml`）。

### 捜索結果表

| run / 評価経路 | config beta | α / γ_pt | ckpt | test_play json.gz | train_play json.gz | 備考 |
|---|---:|---|---|---:|---:|---|
| `phase4/beta1_64x10` | **1.0** | 1.0 / 1.0 | ✓ (`mortal.pth`, `champion.pth`) | **0** | **0** | 疎通学習 400 step のみ |
| `phase4/beta1_192x40` | **1.0** | 1.0 / 1.0 | ✓ (`mortal.pth`, `model.pth`, `best.pth`) | **3000** | 0 | 本番 192×40・1 epoch |
| `phase4/sweep_eval/beta1/1v3` | 1.0 (eval) | — | — | — | — | **400** json.gz（自己対戦 sanity） |
| `phase4/eval/beta1/1v3` | 1.0 (eval) | — | — | — | — | 400 json.gz（sweep_eval と同一統計） |
| `online_main` 他 7 run | 0.3 | 1.0 / 1.0 | ✓ | 各 3000 | 各あり | **β=1 期間なし** |
| `phase4d/*` | 0.3 | 1.0 / 1.0 | ✓ | 各あり | — | Phase4d は β 縮小後 |

**結論 (Step 1):** β=1 の **ckpt は現存**。**json.gz ログは 2 系統** — (A) `beta1_192x40/test_play` 3000 半荘（mortal vs baseline×3）、(B) `sweep_eval/beta1/1v3` 400 半荘（champion=challenger 自己対戦）。`beta1_64x10` は ckpt のみでログなし。

### `freeparlor/docs/` の Phase4 β 導入期ドキュメント

| ファイル | β=1 言及 |
|---|---|
| `phase4_chip.md` | **主文档**。β スイープ表・打牌統計・avg_rank sanity (2.500)。**収支/PnL 数値なし** |
| `phase4_aka_conditional.md` | β=1.0 崩壊モデルの赤条件分解 |
| `next_steps_2.md` | 「β=1.0で崩壊」一行 |
| その他 chip/beta 言及 | online 系はすべて β=0.3 前提 |

---

## Step 2: β=1 ログからの収支再計算

再計算方法: `preprocess_chips.hora_chip_deltas` + mjai `deltas` を局単位で合算。`Stat.from_dir(..., 'mortal')` で avg_rank・打牌統計。test_play は seat ローテーション（4 split）があるため **各ファイルの `start_game.names` から mortal 席を特定**（player 0 固定は不可）。

### A. `beta1_192x40/test_play` — mortal vs baseline×3, n=3000 半荘

**avg_rank・打牌統計（当時未記録だった competitive 指標）**

| 指標 | β=1 (test_play) | β=0.3 (test_play, 参考) |
|---|---:|---:|
| avg_rank | **1.246** | 1.009 |
| avg_pt (uma 千点) | +28.91 | +34.73 |
| 和了率 | 14.03% | 36.40% |
| 副露率 | 4.39% | 7.77% |
| 流局率 | 84.36% | 62.31% |
| 副露和了率 | 0.14% | 2.26% |
| 平均和了打点 | 10416 | 8038 |

**収支（mortal 席, チップ込み）**

| 指標 | 値 |
|---|---:|
| 平均局収支（素点+チップ） | **+3191 点/局** |
| 平均半荘収支・素点のみ | +14240 |
| 平均半荘収支・チップのみ | +13928 |
| 平均半荘収支・素点+チップ+ウマ | **+57075** |
| chip 実現率（chip≠0 局で hora かつ chip>0） | 97.31% |

**副露局 vs 門前局（割に合わない鳴き仮説）**

| 区分 | 平均局収支 | 構成比 |
|---|---:|---:|
| 副露局 | **+284** | 4.39% |
| 門前局 | **+3324** | 95.61% |

副露局の結末内訳（mortal 副露局 n=1253）: 和了 40 / 流局 1194 / 放銃 2 / 他家和了 17

### B. `sweep_eval/beta1/1v3` — 自己対戦 sanity, n=400 半荘

`phase4_chip.md` に記録済みの打牌統計と一致（fuuro 0.51%, ryukyoku 62.63%, agari 9.39%）。

| 指標 | 値 | 解釈 |
|---|---:|---|
| avg_rank | 2.500 | 自己対戦のため **competitive 指標として無意味** |
| avg_pt | 0.00 | 同上 |
| 平均半荘収支（full） | −176 | ゼロ和に近い（seat ローテーション込み） |
| 副露局 avg / 門前局 avg | −680 / +193 | n 極小（副露局 20 局）|

---

## 結論

### β=1 の収支 / avg_rank は判明したか？

**Yes（部分）**

| 指標 | サルベージ可否 | 数値 |
|---|---|---|
| avg_rank（competitive） | **可** | **1.246**（test_play n=3000 vs baseline） |
| avg_rank（self-play sanity） | 可（既存 doc とも一致） | 2.500（sweep n=400） |
| 半荘収支（チップ+ウマ込み） | **可** | **+57075 / 半荘**（test_play vs baseline） |
| 局収支・副露 vs 門前 | **可** | 副露 +284 vs 門前 +3324 |

**No（欠落）**

- β=1 期の **online run ログは皆無**（online は最初から β=0.3）。
- `beta1_64x10` はログなし。
- 当時の判断材料として記録されていたのは **sweep self-play の打牌統計のみ**（`phase4_chip.md`）。**vs baseline の avg_rank / 収支は未記録だったが、ログから後付け再計算可能**だった。

### 跳ねた副露が収支を悪化させたか？（1行）

**vs baseline では avg_rank 1.246・半荘 +57075 と強く、収支面の「崩壊」は見えない**；一方 **副露局の平均局収支 (+284) は門前 (+3324) を大きく下回り**、稀な鳴きは経済合理性に対して取りこぼしがある（ただし副露率 4.39% と極小）。

### 代替候補（β=1 ログが無かった場合の案 — 参考）

今回は test_play 3000 半荘が現存したため不要。万一無かった場合:

1. **β=0.3 run から β=1 収支を近似不可** — β が報酬スケールと学習済み方策の両方を変えるため、打牌分布も収支も非線形。
2. **新規 run 1 本**（β=1 ckpt + vs baseline test_play 3000 半荘）が最短。GPU 1 回の評価のみで足りる（学習不要、既存 `beta1_192x40` ckpt 使用）。

---

## 実行コマンド（再現用・読み取りのみ）

```bash
# 打牌統計
/home/gamba/miniconda3/envs/mortal/bin/python -c "
import sys; sys.path.insert(0,'/home/gamba/mahjong/Mortal/mortal')
from libriichi.stat import Stat
s=Stat.from_dir('/home/gamba/mahjong/runs/phase4/beta1_192x40/test_play','mortal',True)
print('avg_rank', s.avg_rank, 'fuuro', s.fuuro_rate, 'agari', s.agari_rate)
"

# PnL 再計算は preprocess_chips + hora_chip_deltas を各 json.gz に適用（本調査で ad-hoc 実行）
```

## sanity

- [x] Step 1 を net で確認してから Step 2 へ進んだ
- [x] 新規学習なし・既存スクリプト未変更
- [x] 存在しない数値は埋めていない（online β=1、beta1_64x10 ログ等）
