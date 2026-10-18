# 赤を活かした鳴き和了（target 正例）の希少性

**日付:** 2026-06-30

## 実行条件

- コマンド: `python count_aka_call_hora.py --log-dir /home/gamba/mahjong/runs/online_diag_b/train_play/client0 --version 4`
- `--log-dir`: `/home/gamba/mahjong/runs/online_diag_b/train_play/client0`
- 対象ファイル数: 800
- 総 game 数: 3198
- スキップ game 数（at_kyoku 連番前提違反）: 2
- 総局数: 28672

## 集計表（局単位）

| 区分 | 局数 | 全局比(%) | chip局比(%) | 1ファイル平均 | min | median | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1. 全局数 | 28672 | 100.0000 | — | 35.84 | 8 | 36.0 | 72 |
| 2. trainee和了 | 3352 | 11.6908 | 58.3667 | 4.19 | 0 | 4.0 | 10 |
| 3. trainee和了 ∧ 副露あり | 165 | 0.5755 | 2.8731 | 0.21 | 0 | 0.0 | 3 |
| 4. trainee和了 ∧ 副露 ∧ 赤 (target) | 20 | 0.0698 | 0.3483 | 0.03 | 0 | 0.0 | 2 |
| 5. trainee和了 ∧ 立直 ∧ 赤 (参考) | 369 | 1.2870 | 6.4252 | 0.46 | 0 | 0.0 | 7 |
| 6. chip signal (r_chip≠0) | 5743 | 20.0300 | 100.0000 | 7.18 | 0 | 6.0 | 20 |

### 正例(4) ファイル単位ヒストグラム

| 件数/ファイル | ファイル数 |
|---|---|
| 0 | 781 |
| 1 | 18 |
| 2 | 1 |

0件ファイル率: 97.6250%

## 判定根拠

### 鳴きアクション ID（GameplayLoader / libriichi gameplay.rs v4）

- 副露判定: trainee action ∈ [38, 39, 40, 41, 42]
- 立直判定: trainee action == 37 (reach (riichi))
- 全ログ move 集計: 37=reach (riichi) (4196 moves), 38=chi_low (415 moves), 39=chi_mid (10157 moves), 40=chi_high (4273 moves), 41=pon (20647 moves), 42=kan_select (daiminkan / kakan / ankan) (2796 moves)

※ action 42 は daiminkan に加え kakan/ankan も同 ID。副露は chi/pon/明カン中心だが、loader 上は 42 を副露相当として含む（暗槓/加槓のみで和了した局は稀な誤包含）。

### 赤判定（n_aka > 0）

- 経路: 生ログを `PlayerState` で再生 → trainee hora 時に `agari_detail(is_ron, ura)` → `detail.num_aka`
- 根拠: `preprocess_chips.chip_base(detail)` が `detail.num_aka` を chip 枚数に含めるのと同経路
- hora 集約: `collect_hora_by_kyoku` 相当を `collect_kyoku_event_data` に統合（`get_hora_chip_delta` 使用）
- chip signal との対応: `load_kyoku_hora_r_chip` + `assign_r_chip_to_trainee_final_moves` で r_chip≠0

### cross-check

- 期待 chip signal 局数（kyoku_length_dist.md）: 5743
- 今回 (6): 5743（差 0）→ OK
- 包含関係: (2)≥(3)≥(4): 3352≥165≥20 → OK

## 結論

正例(4) = 全局の **0.0698%** ・1ファイル平均 **0.03件** ・0件ファイル率 **97.6250%**。
希少性が主因と言えるか: **Yes** — 正例(4)は全局の0.0698%・1ファイル平均0.03件・0件ファイル率97.6250%と極めて薄く、batch 内で平均化されやすい。
