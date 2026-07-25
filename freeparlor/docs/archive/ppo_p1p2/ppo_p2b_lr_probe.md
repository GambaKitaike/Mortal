# PPO P2b — lr プローブ (2026-07-03)

**仮説:** fuuro 崩壊は critic 未学習下の advantage ノイズによるレア行動ラチェットで、
lr=3e-4 が加速主因。

**単一変数:** `ppo.lr` 3e-4 → **2e-5** のみ。他は【3】再走と同一
(client×3, 400 step, inline test_play 無効, c_ent=0.01)。

**run dir:** `/home/gamba/mahjong/runs/ppo/smoke_p2b/`  
**ブランチ:** `ppo-migration`

---

## 1. Run 結果

| 項目 | 値 |
|---|---|
| 学習 | 2026-07-03 01:21–02:28、tmux `ppo_p2b_lr` |
| OOM / mismatch / chip | なし / 0 / 0 |
| clip @ step400 epoch4 | 0.072 |
| eval | 2026-07-03 03:30–04:00、100半荘 seed [10000,10100) |

---

## 2. control eval 比較表

| 指標 | init | step400 (lr=3e-4) | step400 (lr=2e-5) |
|---:|---:|---:|---:|
| fuuro | **17.29%** | **0.00%** | **0.00%** |
| riichi | 23.65% | 14.91% | 8.47% |
| agari | 20.55% | 14.84% | 18.93% |
| houjuu | 12.18% | 7.87% | 10.79% |
| avg_rank | 2.5000 | 2.5000 | 2.5000 |

- init: `ppo_p2_diag.md` §1（mortal_init.pth）
- lr=3e-4: `ppo_p2_smoke.md` §3 eval_sanity (2026-07-02)
- lr=2e-5: `eval_sanity_step400_lr2e5.log`（checkpoint `smoke_p2b/mortal.pth`）

---

## 3. π(鳴き|可能) / π(立直|可能) 時系列（lr=2e-5）

`ppo_diag.jsonl` `event=action_mass`（400 step 各1行）。

| 指標 | step 0 / 初 | step 399 / 終 |
|---|---:|---:|
| π(鳴き\|可能) | 0.140 | 0.100 |
| π(立直\|可能) | — (n=0) | 0.042 |

step 1 時点 π(立直\|可能)=0.526 → step 399 で 0.042（**−92%**）。

π(鳴き\|可能): 400 step 中 **減少 208 / 増加 191**（単調減衰ではないが net −29%）。
lr=3e-4 側の action_mass 計装は未実施（【3】再走前）→ **lr 間の減衰速度比較は不可**。

---

## 4. 結論

> **Q1: lr 2e-5 で fuuro 崩壊は止まったか？（fuuro が init 水準の半分以上残存 = Yes）**

**No.** step400 eval fuuro=**0.00%**（init 17.29% の 0%）。半分以上（≥8.6%）には届かず。
lr 15× 低減でも eval 副露率は lr=3e-4 と同様ゼロ。

> **Q2: π(鳴き\|可能) の時系列はラチェット（単調減衰）を示したか？ lr で減衰速度は変わったか？**

**部分的 Yes / lr 比較は未確定。**

- lr=2e-5 でも π(鳴き\|可能) は 0.140→0.100、π(立直\|可能) は 0.526→0.042 と
  **レア行動 mass の net 減衰**は観測される（ラチェット仮説と整合）。
- 厳密な単調減衰ではない（step 間で増減混在）。
- lr=3e-4 の action_mass 未計装のため、**減衰速度の lr 依存は本 run では判定不能**。

**総評:** lr 単独では fuuro 崩壊を止められない。方策 mass 減衰は lr=2e-5 でも進行 —
  仮説の「lr が加速主因」は棄却方向（低 lr でも eval fuuro=0）。

---

## 5. 再現

```bash
bash freeparlor/scripts/run_ppo_p2b_lr_probe.sh

MORTAL_CFG=/home/gamba/mahjong/runs/ppo/smoke_p2b/config.toml \
EVAL_LABEL=step400_lr2e5 \
EVAL_CHECKPOINT=/home/gamba/mahjong/runs/ppo/smoke_p2b/mortal.pth \
bash freeparlor/scripts/run_eval_ppo_smoke_sanity.sh

conda run -n mortal python freeparlor/scripts/summarize_p2b_action_mass.py \
  /home/gamba/mahjong/runs/ppo/smoke_p2b/logs/ppo_diag.jsonl
```
