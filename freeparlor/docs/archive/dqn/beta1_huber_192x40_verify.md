# β=1 Huber 192×40 — 交絡排除検証 (2026-07-02)

## 目的

`reward_design_teacherfree.md` §3 #2: **64×10 Huber run の小モデル交絡を排除**し、崩壊 arm（β=1 MSE 192×40）と同一条件で **`dqn_loss` のみ** Huber に差し替えたとき、スケール仮説が 192×40 でも成立するかを検証する。

**仮説（再掲）:** β=1 崩壊は経済パラメータではなく、生 MC リターンへの MSE がチップ外れ値残差に支配される最適化問題。Huber(δ=15) のみで打牌・チップが回復する。

---

## 大前提（守ったもの / 変えたもの）

| 項目 | 扱い |
|---|---|
| 報酬式・`chip_value=5.0`・`beta=1.0` | **変更なし** |
| `lambda_opp` / `min_q_weight=5` / データ / pts / arch | 崩壊 run（`beta1_192x40`）と同じ |
| `batch_size=128` / `num_epochs=1` / `max_steps=100000` / `save_every=400` / `test_every=20000` / `test_play.games=3000` | **変更なし** |
| 変更点 | **`dqn_loss='huber'`, `huber_delta=15.0` のみ** + run パス |
| online 自己対戦 | **未使用** |
| モデル | **192×40**（崩壊 arm と同 arch） |

Config: `freeparlor/configs/phase4_chip_beta1_huber_192x40.toml`  
Run dir: `/home/gamba/mahjong/runs/phase4/beta1_huber_192x40/`

---

## Step 2: 検証 run

| 項目 | 値 |
|---|---|
| GPU | cuda:0 (RTX 5060) |
| 学習量 | **20,000 steps**（offline 1 epoch 完走） |
| NaN/発散 | **なし** |
| checkpoint | `mortal.pth`（step 20,000） |
| 評価 | vs baseline **test_play 3000 半荘**（`grp_baseline.pth` ×3 席） |

test_play ログ（train.log より）:

```
avg rank: 1.006
test_play behavior: agari=39.81% houjuu=0.11% fuuro=7.88% riichi=21.03% ryukyoku=59.39%
```

---

## Step 3: 結果（競技経路 test_play）

集計:

- 打牌統計: `Stat.from_dir(..., 'mortal')`
- チップ: `freeparlor/scripts/analyze_chip_realize.py`（赤保持局 = 局終了 `aka_held>0`）
- 局収支‡: mjai `deltas` の**素点のみ**（`beta1_huber_verify.md` 表と同一定義）

| arm | 損失 | arch | 和了率 | 流局率 | 副露率 | avg_rank | チップ実現率† | 鳴き和了率† | 副露局収支‡ | 門前局収支‡ | 出所 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| β=1 MSE 192×40（崩壊・既存） | MSE | 192×40 | 14.03% | 84.36% | 4.39% | 1.246 | 13.68% | 0.13% | +104 | +1800 | `beta1_192x40/test_play` |
| β=1 Huber 64×10（既存） | Huber | 64×10 | 44.73% | 54.35% | 6.99% | 1.006 | 45.75% | 3.68% | +4034 | +3379 | `beta1_huber_64x10/test_play` |
| **β=1 Huber 192×40（本 run）** | **Huber** | **192×40** | **39.81%** | **59.39%** | **7.88%** | **1.006** | **40.90%** | **2.84%** | **+3207** | **+3246** | **本 run** |

† `analyze_chip_realize` 赤保持局ベース  
‡ 素点 delta のみ（mortal 席、副露/門前は鳴き有無で分割）

### 崩壊 arm からの改善幅（本 run）

| 指標 | MSE 崩壊 | Huber 192×40 | Δ |
|---|---:|---:|---:|
| 和了率 | 14.03% | 39.81% | **+25.8pp** |
| 流局率 | 84.36% | 59.39% | **−24.9pp** |
| avg_rank | 1.246 | 1.006 | **−0.24** |
| チップ実現率 | 13.68% | 40.90% | **+27.2pp** |

### 64×10 Huber との差（交絡チェック）

| 指標 | Huber 64×10 | Huber 192×40 | 解釈 |
|---|---:|---:|---|
| 和了率 | 44.73% | 39.81% | 192×40 やや低いが崩壊 arm を大幅に上回る |
| チップ実現率 | 45.75% | 40.90% | 同左 |
| avg_rank | 1.006 | 1.006 | **同等** |
| 鳴き和了率 | 3.68% | 2.84% | 64×10 が僅差で高い |

---

## 結論（Yes/No + 数値）

### Q1: 192×40 でも Huber は β=1 崩壊を回復させたか？

**Yes.**

- 和了 **39.81%** vs 崩壊 **14.03%**（+25.8pp）
- 流局 **59.39%** vs **84.36%**（−24.9pp）
- avg_rank **1.006** vs **1.246**
- チップ実現率 **40.90%** vs **13.68%**（+27.2pp）

MSE 崩壊の病理（流局偏重・和了枯れ）が、arch を変えず `dqn_loss` 差し替えのみで解消した。

### Q2: 回復幅は 64×10 と同水準か？ 小モデル交絡か？

**スケール仮説は 192×40 でも成立**（64×10 の効果は交絡ではなかった）。

- avg_rank は両 Huber arm で **1.006** と一致 → 競技性能の回復は arch 非依存。
- 和了/チップ実現は 64×10 が **+4〜5pp** 僅差で上（44.73% / 45.75% vs 39.81% / 40.90%）。これは「小モデルが崩壊を和らげていた」方向ではなく、**大モデル側がやや控えめ** — 容量差・最適化差の範囲で、崩壊救済の主因が 64×10 サイズにあるとは言えない。

### Q3: 副露局収支 > 門前局収支 の逆転は 192×40 でも維持されるか？

**No（維持されない）。**

| arm | 副露局収支‡ | 門前局収支‡ | 逆転 |
|---|---:|---:|---|
| MSE 崩壊 | +104 | +1800 | 門前 ≫ 副露 |
| Huber 64×10 | +4034 | +3379 | **副露 > 門前** |
| **Huber 192×40** | **+3207** | **+3246** | **ほぼ同値（門前 +39）** |

64×10 で見えた「鳴き局の相対優位」は 192×40 では再現せず、両局種とも高収支（+3200 点台）に収束。Huber 回復自体は arch 非依存だが、**副露優位パターンは 64×10 固有**の可能性がある。

---

## 補足: 64×10 との学習量非対称

Huber 64×10 は `batch_size=256`、Huber 192×40 は `batch_size=128`。両者 20k step のため、実サンプル更新量は **5.12M vs 2.56M**（192×40 が半分）。加えて 192×40 はパラメータ約 9 倍。

よって **Q1**（Huber vs MSE @192×40、両者 batch128・20k・予算一致）は無傷だが、**Q2** の「大モデルやや控えめ（−4〜5pp）」と **Q3** の「副露>門前 逆転が消えた」は **cross-arch かつ学習予算非対称の交絡**を含む。

### 結論の格

| 問い | 格 | 内容 |
|---|---|---|
| Q1 | **確定** | 192×40 でも Huber は MSE 崩壊を回復 |
| Q2 | **部分確定** | スケール仮説成立は確定（回復は 192×40 でも実在）。4–5pp gap の arch 帰属は**保留** |
| Q3 | **事実のみ** | 「192×40 で副露≈門前（パリティ）」は事実。64×10 の逆転が arch 固有か学習量差かは本実験では分離不能 → **「逆転は再現せず」を確定 claim にはしない** |

### 設計含意

どちらの解釈でも設計判断は不変：**副露が門前と赤字でない＝経済的に鳴きを潰す必要はない**。Q3 確定のための再燃焼はコスト対効果が低く**非推奨**。

---

## sanity

- [x] **config diff は `dqn_loss` 系 + run パスのみ**

```bash
diff -u freeparlor/configs/phase4_chip_beta1_192x40.toml \
        freeparlor/configs/phase4_chip_beta1_huber_192x40.toml
# 追加: dqn_loss='huber', huber_delta=15.0
# 変更: beta1_192x40 → beta1_huber_192x40 パス
```

- [x] **崩壊 arm と一致:** `batch_size=128`, `min_q_weight=5`, `beta=1.0`, `conv_channels=192`, `num_blocks=40`, 2009 天鳳 glob — いずれも同一
- [x] **`lambda_opp`:** 両 config とも未指定（default 使用、同一）
- [x] **新規 online run なし**（`online = false`、offline 1 epoch のみ）
- [x] **NaN/発散なし**（train.log 確認）

---

## Artifacts

```
freeparlor/configs/phase4_chip_beta1_huber_192x40.toml
freeparlor/docs/beta1_huber_192x40_verify.md

/home/gamba/mahjong/runs/phase4/beta1_huber_192x40/
  config.toml
  mortal.pth
  train.log
  tb/
  test_play/             # 3000 json.gz
```

## 再現コマンド

```bash
# 学習（spawn ランチャ）
MORTAL_CFG=/home/gamba/mahjong/runs/phase4/beta1_huber_192x40/config.toml \
  python /home/gamba/mahjong/runs/run_train.py

# 打牌統計
python -c "
import sys; sys.path.insert(0,'/home/gamba/mahjong/Mortal/mortal')
from libriichi.stat import Stat
s=Stat.from_dir('/home/gamba/mahjong/runs/phase4/beta1_huber_192x40/test_play','mortal',True)
print('avg_rank', s.avg_rank, 'agari', s.agari_rate, 'fuuro', s.fuuro_rate, 'ryukyoku', s.ryukyoku_rate)
"

# チップ実現
python freeparlor/scripts/analyze_chip_realize.py --skip-human-mjson \
  --eval huber192:/home/gamba/mahjong/runs/phase4/beta1_huber_192x40/test_play
```
