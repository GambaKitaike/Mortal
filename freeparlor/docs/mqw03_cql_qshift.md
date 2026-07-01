# mqw03 CQL 鳴き Q 押し下げ検証

**日付:** 2026-07-01

**目的:** step6000→8000 の P(call|legal) 半減が、CQL による鳴き行動 Q の累積押し下げかを train ログから直接確認する。本線（教師非依存再設計）への方針は結果に依らず不変。

---

## Step 0: 記録所在調査

ベース: `/home/gamba/mahjong/runs/online_cql_mqw03/`

### 1. TensorBoard (`tb/`)

| 項目 | 結果 |
|---|---|
| events ファイル | あり（メイン `tb/events.out.tfevents.*` ×10、test_play サブディレクトリ多数） |
| `loss/cql_loss` | **記録なし**（tag 一覧に存在しない） |
| `loss/dqn_loss`, `loss/chip_loss`, `loss/next_rank_loss` | あり（save_every=400 毎） |
| Q ヒストグラム | `q_predicted`, `q_chip_predicted`, `q_target` あり（save_every=400 毎） |
| 鳴き vs 門前の Q 分離 scalar | **記録なし** |
| 鳴き頻度 proxy | `test_play/behavior/fuuro` あり（test_every=2000 毎） |

**`loss/cql_loss` が無い理由:** `train.py` は online モードでは CQL loss を計算するが TensorBoard への scalar 書き込みを `if not online:` でスキップしている（offline のみ記録）。

```346:347:mortal/train.py
                if not online:
                    writer.add_scalar('loss/cql_loss', stats['cql_loss'] / save_every, steps)
```

`q_predicted` 等のヒストグラムは **バッチ内でサンプルされた行動** の Q のみ（行動種別・鳴き/門前の分離なし）。

### 2. train 生ログ (`logs/`)

| ファイル | CQL / Q 統計 |
|---|---|
| `trainer.log` | step 進行ログのみ。**cql_loss / Q 統計の定期 print なし** |
| `extend.nohup` | test_play archive 時の behavior 一行（fuuro% 等）のみ |
| `client{0,1,2}.log`, `server.log` | 転送・接続ログのみ |

### 3. step 別チェックポイント

```
find .../online_cql_mqw03 -name "*.pth"
→ mortal.pth (steps=16000, 131MB)
→ mortal_warmstart.pth (steps=0)
```

| step | challenger/mortal ckpt |
|---:|---|
| 2000 | **なし** |
| 4000 | **なし** |
| 6000 | **なし**（extend 開始時点の ckpt は上書き消滅） |
| 8000 | **なし** |
| 10000–16000 | **なし** |

`config.toml`: `save_every=400`, `state_file=mortal.pth`（単一ファイル上書き）。step 別アーカイブ ckpt は `/home/gamba/mahjong` 配下にも存在せず。

### 4. 副次データ（本カットでは未使用）

`test_play_step{2000..16000}/` の各 json.gz には trainee 行動の `meta.q_values` / `meta.mask_bits` が記録されている。ただし Step B 要件（step6000 局面固定セット × 各 step ckpt 再推論）を満たせず、本カットでは深追いしない。

---

## 実施ステップ

| ステップ | 判定 | 理由 |
|---|---|---|
| **Step A**（既存記録抽出） | **部分実施** | TB に CQL loss・行動別 Q は無いが、loss 系・Q ヒストグラム（集約）・fuuro 率は抽出可能 |
| **Step B**（ckpt 再計算） | **未実施** | step 別 ckpt が存在しない |

---

## Step A 抽出結果

### loss 系（test_every 時点）

| step | dqn_loss | chip_loss | next_rank_loss | cql_loss |
|---:|---:|---:|---:|---|
| 2000 | 29.22 | 0.160 | 0.288 | **記録なし** |
| 4000 | 36.71 | 0.208 | 0.248 | **記録なし** |
| 6000 | 35.18 | 0.164 | 0.248 | **記録なし** |
| 8000 | 34.87 | 0.165 | 0.252 | **記録なし** |
| 10000 | 38.36 | 0.196 | 0.243 | **記録なし** |
| 12000 | 40.14 | 0.194 | 0.256 | **記録なし** |
| 14000 | 36.17 | 0.204 | 0.276 | **記録なし** |
| 16000 | 37.45 | 0.182 | 0.233 | **記録なし** |

6000→8000: dqn_loss 35.18→34.87（-0.31、大きな変化なし）。

### Q ヒストグラム平均（行動種別非分離・参考値）

バッチサンプル行動の Q_main（`q_predicted`）。**鳴き vs 門前の Δ ではない。**

| step | q_predicted | q_chip_predicted | q_target |
|---:|---:|---:|---:|
| 2000 | 7.119 | 0.610 | 8.048 |
| 4000 | 8.158 | 0.700 | 8.284 |
| 6000 | 7.886 | 0.661 | 7.868 |
| 8000 | 8.129 | 0.661 | 8.424 |
| 10000 | 8.270 | 0.709 | 8.221 |
| 12000 | 8.318 | 0.638 | 8.242 |
| 14000 | 7.742 | 0.640 | 7.813 |
| 16000 | 8.048 | 0.635 | 7.928 |

6000→8000: q_predicted **+0.243**（押し下げではなく微増）。q_chip は横ばい（0.661→0.661）。

### 鳴き vs 門前 Q（Δ = mean_Q(鳴き) − mean_Q(門前)）

| step | mean_Q(鳴き) | mean_Q(門前) | Δ |
|---:|---:|---:|---:|
| 全 step | **記録なし** | **記録なし** | **記録なし** |

### 状態証拠（行動出力・参考）

| step | test_play fuuro% | P(call\|legal) ※mqw03_collapse_diag |
|---:|---:|---:|
| 6000 | 48.90% | 30.42% |
| 8000 | 26.89% | 15.37% |
| Δ | **-22.01pp** | **-15.06pp** |

※ P(call|legal) は [mqw03_collapse_diag.md](mqw03_collapse_diag.md) の B1 集計。

---

## 崩落区間 step6000→8000（強調）

| 指標 | step6000 | step8000 | Δ | 直接 Q 証拠 |
|---|---:|---:|---:|---|
| P(call\|legal) | 30.42% | 15.37% | -15.06pp | 状態証拠のみ |
| test_play fuuro% | 48.90% | 26.89% | -22.01pp | 状態証拠のみ |
| mean_Q(鳴き) − mean_Q(門前) | — | — | — | **記録なし** |
| cql_loss | — | — | — | **記録なし** |
| q_predicted（集約） | 7.886 | 8.129 | +0.243 | 行動種別非分離のため仮説検証に不十分 |

---

## 結論

**「鳴き行動の Q が step で相対的に押し下げられている」の直接確認: No（記録不足）**

根拠:
1. **CQL loss 時系列なし** — online モードで TB 未記録、生ログにも無し。
2. **鳴き(38–42) vs 門前の Q 分離記録なし** — TB ヒストグラムはバッチサンプル行動の集約 Q のみ。
3. **step 別 ckpt なし** — Step B（固定局面 × ckpt 再推論）不可。6000/8000 時点の重みを復元できない。
4. 利用可能な集約 Q（q_predicted）は 6000→8000 で押し下げを示さないが、検証対象の Δ（鳴き−門前）ではないため、CQL 累積抑圧仮説の否定・肯定どちらにもならない。

**間接的に整合する状態証拠:** P(call|legal) 半減・fuuro% 急落（[mqw03_collapse_diag.md](mqw03_collapse_diag.md)）。機構仮説「CQL が鳴きを OOD 扱い Q を押し下げ」は**直接証拠なく未確認**。

**方針:** 本線（教師データ非依存の再設計）へ進む判断は不変。
