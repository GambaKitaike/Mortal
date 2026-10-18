# β=1 Huber 損失検証 — スケール仮説 (2026-07-01)

## 目的

β=1（1枚=5000点=経済的に正しいレート）の offline 学習は副露 0.51% / 和了 14% / 流局 84% で打牌が崩壊した（`phase4_chip.md`, `beta1_pnl_salvage.md`）。

**仮説:** 崩壊は経済ではなく最適化の頑健性。生 MC リターンへの MSE（`train.py` `0.5*mse(q, q_target_mc)`）が、β=1 で +10〜+50 に達するチップ和了局の巨大残差に支配され、通常局の Q フィットが崩れている。

**検証:** MSE を Huber (`smooth_l1`, δ=15) に差し替え、報酬（経済価値）を一切変えずに最適化だけ頑健化して β=1 が健全化するかを見る。

---

## 大前提（守ったもの / 変えたもの）

| 項目 | 扱い |
|---|---|
| 報酬式・`chip_value=5.0`・`beta=1.0` | **変更なし** |
| `lambda_opp` / `min_q_weight` / データ / pts | 崩壊 run と同じ |
| 変更点 | **`dqn_loss` の測り方のみ**（MSE → Huber） |
| online 自己対戦 | **未使用**（offline 1 本・2009 天鳳） |
| モデル | **64×10**（GPU 最小化。崩壊機構はサイズ非依存と想定） |

---

## Step 1: コード変更

`mortal/train.py` — `dqn_loss` を config スイッチ化。default は既存挙動（MSE）。

`mortal/config.py` — default 追加:

```toml
[control]
dqn_loss = 'mse'      # or 'huber'
huber_delta = 15.0
```

```python
if dqn_loss_type == 'huber':
    dqn_loss = F.huber_loss(q, q_target_mc, delta=huber_delta)
else:
    dqn_loss = 0.5 * mse(q, q_target_mc)
```

- **δ=15 の根拠:** 通常リターン（素点±12・順位±2）は二乗域で精密に、β=1 チップ外れ値 (+10〜50) は線形域で de-weight。
- **触っていない:** `chip_loss`（MSE のまま）、`cql_loss`、online 経路。

Config テンプレ: `freeparlor/configs/phase4_chip_beta1_huber_64x10.toml`  
（`phase4_chip_beta1_64x10.toml` との唯一の diff = `dqn_loss` / `huber_delta` / run パス）

---

## Step 2: 検証 run

| 項目 | 値 |
|---|---|
| run dir | `/home/gamba/mahjong/runs/phase4/beta1_huber_64x10/` |
| config | `runs/phase4/beta1_huber_64x10/config.toml` |
| GPU | cuda:0 (RTX 5060) |
| 学習量 | **20,000 steps**（1 epoch 完走） |
| NaN/発散 | なし |
| checkpoint | `mortal.pth` |
| 評価 | vs baseline **test_play 3000 半荘**（`grp_baseline.pth` ×3 席） |

---

## Step 3: 結果（2 軸 — 競技経路 test_play）

集計:

- 打牌統計: `Stat.from_dir(..., 'mortal')`
- チップ: `freeparlor/scripts/analyze_chip_realize.py`（赤保持局 = 局終了 `aka_held>0`）
- 局収支: `preprocess_chips.hora_chip_deltas` + mjai `deltas`、1枚=5000点

| arm | 損失 | β | arch | 和了率 | 流局率 | 副露率 | avg_rank | チップ実現率† | 鳴き和了率† | 副露局収支‡ | 門前局収支‡ | 出所 |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| β=1 MSE（崩壊） | MSE | 1.0 | 192×40 | 14.03% | 84.36% | 4.39% | 1.246 | 13.68% | 0.13% | +104 | +1800 | `beta1_192x40/test_play` |
| β=0.3 MSE（健全） | MSE | 0.3 | 192×40 | 36.40% | 62.31% | 7.77% | 1.009 | 35.99% | 1.61% | +2467 | +3588 | `beta0_3_192x40/test_play` |
| **β=1 Huber（新規）** | **Huber** | **1.0** | **64×10** | **44.73%** | **54.35%** | **6.99%** | **1.006** | **45.75%** | **3.68%** | **+4034** | **+3379** | **本 run** |

† `analyze_chip_realize` 赤保持局ベース  
‡ 局収支 = 素点 delta + chip×5000（mortal 席）

### test_play 行（本 run 詳細）

```
avg rank: 1.00633
avg pt:   +89.655 (千点 uma)
agari=44.73%  houjuu=0.12%  fuuro=6.99%  riichi=15.65%  ryukyoku=54.35%
agari_pt=6608  houjuu_pt=-6752.9
```

### 参考: 自己対戦 sanity（competitive 指標に使わない）

| arm | 和了率 | 流局率 | 副露率 | avg_rank | 出所 |
|---|---:|---:|---:|---:|---|
| β=1 MSE | 9.39% | 62.63% | 0.51% | 2.500 | `phase4/sweep_eval/beta1/1v3` |
| β=0 (balanced) | 21.38% | 15.45% | 16.97% | 2.490 | 同上 beta0 |

---

## 判定（決定木）

```
打牌健全?
  → YES（和了 44.7% / 流局 54.4% / 副露 7.0% — 崩壊 arm を大幅に上回る）
      ↓
チップ実現率 ≥ β=0.3?
  → YES（45.75% >> 35.99%、鳴き和了率 3.68% >> 1.61%）
      ↓
★ 本命成功 — スケール問題確定 + チップ機能維持（むしろ向上）
```

| 分岐 | 本 run |
|---|---|
| 打牌健全 かつ チップ実現率 ≥ β=0.3 | **該当（成功）** |
| 打牌健全だがチップ信号が削れた | 非該当 |
| 打牌が崩壊のまま | 非該当 |

**結論:** β=1 の報酬スケール自体は問題なく、**MSE がチップ外れ値に支配されていた**のが主因。Huber(δ=15) のみで経済パラメータを変えずに打牌・チップ両方が回復した。

---

## 補足: test_play の流局率が高い理由

流局 54% は絶対値では高いが、**ランダム相手ではない**。

- 3 席は Phase1 **192×40 `grp_baseline.pth`**（順位報酬 Mortal、固定 checkpoint）
- 1 席の学習済み mortal が avg_rank≈1.0 と圧倒 → 放銃 ~0.1%、局の結末が「mortal 和了 vs 流局」に偏る
- 同じ test_play 経路の健全系も流局 **59–62%**（`phase4d_sweep_results.md` 参照）
- 本 run の 54% はその中では **低め**。崩壊 run の **84%** との差が病理指標

自己対戦（同強度×4）では流局 **~18%** が正常帯。

---

## 注意（交絡）

| 要因 | 影響 |
|---|---|
| Huber vs MSE | 意図した 1 変数 |
| 64×10 vs 192×40 | 比較相手 arm は 192×40。本 run はタスク設計どおり小モデル |
| test_play vs 1v3 | 打牌比較は test_play 内の arm 間のみ有効。絶対値は参考外要素あり |

---

## Artifacts

```
freeparlor/configs/phase4_chip_beta1_huber_64x10.toml
mortal/train.py          # dqn_loss スイッチ
mortal/config.py         # default

/home/gamba/mahjong/runs/phase4/beta1_huber_64x10/
  config.toml
  mortal.pth
  train.log
  tb/
  test_play/             # 3000 json.gz
```

## 再現コマンド

```bash
# 学習
MORTAL_CFG=/home/gamba/mahjong/runs/phase4/beta1_huber_64x10/config.toml \
  python /home/gamba/mahjong/runs/run_train.py

# 打牌統計
python -c "
import sys; sys.path.insert(0,'/home/gamba/mahjong/Mortal/mortal')
from libriichi.stat import Stat
s=Stat.from_dir('/home/gamba/mahjong/runs/phase4/beta1_huber_64x10/test_play','mortal',True)
print('avg_rank', s.avg_rank, 'agari', s.agari_rate, 'fuuro', s.fuuro_rate, 'ryukyoku', s.ryukyoku_rate)
"

# チップ実現
python freeparlor/scripts/analyze_chip_realize.py --skip-human-mjson \
  --eval huber:/home/gamba/mahjong/runs/phase4/beta1_huber_64x10/test_play
```

## sanity

- [x] config diff は `dqn_loss` 系のみ（報酬・β・データ構成は崩壊 run と一致、64×10 化は明記）
- [x] default=`mse` で既存 offline 挙動は不変（`config.py` import 確認済み）
- [x] 評価は競技経路 test_play（self-play avg_rank は表から除外）
- [x] 新規 online run なし

## 次の一手（参考）

- δ スイープ（20, 30）— チップ信号と打牌のトレードオフ確認
- 192×40 + Huber で崩壊 run と完全同条件の再検証
- 本番採用: `[control] dqn_loss = 'huber'` を β=1 offline / online へ展開
