# online 無差別鳴き — 診断A/B 総合報告

**日付:** 2026-06-27  
**背景:** online step26000 で副露50%・赤選択性ゼロ。step2000（beta_sel=0）時点で既に副露61%＝Q_chip 以前に Q_main が鳴き暴走している疑い。  
**方針:** 本番再開なし。lambda_opp と cql を**別々**に短時間（step2000）診断。

---

## 背景数値（参照）

| run | step | 副露率 | 備考 |
|---|---:|---:|---|
| phase4d lo=0.3 warmstart | — | **17.59%** | 1v3 eval |
| online_main | 2,000 | **61.25%** | beta_sel=0, cql=0, lambda_opp=0.3 |
| online_main | 26,000 | **56.63%** | beta_sel=0.3 |

出典: `freeparlor/docs/online_fuuro50_aka_selectivity.md`

---

## 診断設計

| | 診断A | 診断B |
|---|---|---|
| **変更点** | lambda_opp **0.0** | cql **enable_online=true**, min_q_weight=**1** |
| **固定** | cql=0 | lambda_opp=0.3 |
| run dir | `runs/online_diag_a/` | `runs/online_diag_b/` |
| 起動 | warmstart step0 | 同一 |
| 停止 | step2000 test_play | 同一 |
| 詳細 | `online_diag_a_lambda_opp_zero.md` | `online_diag_b_cql_weak.md` |

---

## 結果一覧 ★

### step2000 test_play 副露率（beta_sel ≈ 0）

| run | lambda_opp | cql (online) | min_q_weight | **副露率** | 立直率 |
|---|---:|---|---:|---:|---:|
| phase4d warmstart | 0.3 ※ | あり | 5 | **17.59%** | 23.49% |
| online_main | 0.3 | **0** | — | **61.25%** | — |
| **診断A** | **0.0** | 0 | — | **83.81%** | 7.95% |
| **診断B** | 0.3 | **あり** | 1 | **13.61%** | 34.18% |

※ warmstart ckpt は offline 学習時 lambda_opp=0.3。online 診断の warmstart 復元時は optimizer 除外。

### step2000 その他 behavior

| run | 和了 | 放銃 | 流局 |
|---:|---:|---:|---:|
| 診断A | 21.13% | 15.76% | 16.36% |
| 診断B | 20.73% | 14.53% | 17.96% |
| phase4d warmstart | 20.63% | 13.26% | 18.35% |

---

## 判定

### 診断A — lambda_opp

| 事前仮説 | 結果 |
|---|---|
| lambda_opp=0 なら 61% に跳ねない → 主因 lambda_opp | **棄却** |
| lambda_opp=0 でも跳ねる → 主因は別 | **支持** |

- lambda_opp=0 でも副露 **83.81%**（61% **超**）。online 鳴き暴走の**主因ではない**。
- online 本番から lambda_opp を外す必要性は、本診断では**支持されない**。

### 診断B — cql

| 事前仮説 | 結果 |
|---|---|
| cql 弱導入で副露抑制 → 主因 cql_loss=0 | **支持** |
| 抑制されない | 棄却 |

- min_q_weight=1 の online CQL で副露 **13.61%** → warmstart 17.59% と**同程度の健全域**。
- offline で効いていた CQL 的保守性を online でも（弱く）残すと、step2000 時点の無差別鳴きは**抑制される**。

### 総合

| 要因 | 切り分け結果 |
|---|---|
| **lambda_opp=0.3** | 主因**ではない**（A: 外しても悪化） |
| **cql_loss=0** | **主因候補**（B: 弱 CQL で 13.61% に復帰） |
| Q_main online TD 単体 | cql 無しでは暴走、cql 有りでは抑制 → **cql 経路が支配的** |

**online 本番再開前の示唆:** `enable_online=true` + `min_q_weight` のスイープ（1 は有効、5 は未検証）を検討。lambda_opp 変更は優先度低。

---

## 診断A 83.81% について（保留事項）

診断A step2000 は online_main（61%）より高く、立直 7.95% と極端。意図せず継続した step4000 では 61.15%（立直 20.77%）。

| 可能性 | 内容 |
|---|---|
| 一過性異常 | test_play タイミング・ckpt 状態の問題 |
| lambda_opp=0 副作用 | opp 項除去で Q 学習が別経路に崩れた |
| 再現未確認 | 単発 run のため要再検証 |

**切り分け結論への影響:** 「lambda_opp=0 でも跳ねる」は 83.81% でも 61% でも成立。**主因が lambda_opp でない**という方向性は変わらない。

---

## インフラ

| 項目 | 内容 |
|---|---|
| 起動 | `/home/gamba/mahjong/runs/run_online_diag.sh <run_dir> 2000` |
| 診断A | cleanup 不備で step6000 まで継続（修正前） |
| 診断B | cleanup 正常（`pkill mortal/train.py` 追加後） |
| コード | `mortal/train.py` — `[cql] enable_online` フラグ追加 |

---

## 未実施

| 項目 | 理由 |
|---|---|
| 本番 online 再開 | 原因特定優先（ユーザー指示） |
| min_q_weight スイープ（1 vs 5） | 診断B は 1 のみ |
| 診断A step2000 再現 | 異常値確認用・任意 |
| 赤条件別副露（診断A/B） | 本報告は全体副露のみ |

---

## 集計メタ

| 項目 | パス |
|---|---|
| 診断A config / ログ | `runs/online_diag_a/` |
| 診断B config / ログ | `runs/online_diag_b/` |
| 診断A 詳細 | `freeparlor/docs/online_diag_a_lambda_opp_zero.md` |
| 診断B 詳細 | `freeparlor/docs/online_diag_b_cql_weak.md` |
| 赤選択性・時系列 | `freeparlor/docs/online_fuuro50_aka_selectivity.md` |
| 本番設定 | `freeparlor/docs/online_main_report.md` |

**実施内容:** 診断A・B 完了。複数変数の同時変更なし。本番長時間再開なし。
