# Next Steps / 引き継ぎメモ (更新: 2026-06-29)

## 0. このプロジェクトの位置づけ ★最初に読む

**最終目標:** 商用利用できるフリー雀荘特化麻雀AI。
**本プロジェクトの役割:** その前段調査。「チップあり麻雀で勝てるAIを作るには、報酬を
どう設計すればいいか」を、Mortal フレームワーク上で実験的に解明する。

つまり本リポジトリは**完成品ではなく調査基盤**。各 Phase は「報酬設計の知見」を
積む実験であり、Phase5 までで「offline の限界」と「online の可能性と障壁」が判明した。

---

## 1. プロジェクト一行要約

Mortal（天鳳＝着順特化）の報酬を、フリー雀荘ルール（素点 + ウマオカ + チップ）向けに
再設計し、チップ最適な打牌（特に赤を活かす鳴き）を学習させられるか検証する。

```
reward = α·(素点/1000) + β·(チップ枚数 × 5.0) + γ·順位点 + opp(取りこぼし)
        + [online] β_sel · Q_chip(per-move TD value)
```

---

## 2. 完了済み Phase 一覧

| Phase | 内容 | 結論 |
|---|---|---|
| 0 | 環境構築 WSL2/RTX5060/Rust+pyo3 | — |
| 1 | 素Mortal再現(192×40, 天鳳2009) | 打牌が人間水準 |
| 2 | 素点+ウマオカ報酬 | 報酬設計で打牌は定量的に動く（副露−13.4pp） |
| 3 | α:γ スイープ | **offline天井を発見**：素点重視でも副露増えず |
| 4 | チップβ導入(libriichi改修) | 健全域 β≤0.3。β=1.0で崩壊 |
| 4c | 赤条件別 人間vsAI | 人間は赤持ちで副露+2.81pp、AIは逆(−1.39)。強C棄却 |
| 4d | 取りこぼしプローブ(per-kyoku opp) | §3 |
| **5** | **online自己対戦 + per-move Q_chip TD** | **§4 ★今回の本丸** |

---

## 3. Phase 4d 結論（offline の限界確定）

- 取りこぼし報酬(lambda_opp)は副露・チップ実現を動かすレバーとして機能（lo=0.3で健全）。
- だが頭打ち：赤保持局の**鳴き和了率が人間8.98%に対しAI3%台**で届かない。
- 正体＝**赤を持っても鳴きで活かせない**（立直では活かせる）。これがPhase3天井の具体形。
- 原因＝CQLが「赤で鳴く」をOOD(教師データに無い)として抑制。**offlineでは原理的に超えられない**。
- → online自己対戦が必要、という結論でPhase5へ。

---

## 4. Phase 5 結論 ★最重要（online調査の到達点）

### 5.1 実装したもの（3層TDパイプライン・全層検証済み）
- **層1**: arena の hora イベントに `chip_delta[4]` を埋め込み（Rust改修、非ゼロ2530件でcross-check済み）
- **層2**: dataloader で TD transition `(s,a,r_chip,s',done)` を構成。r_chip は
  「局のtrainee最終move」に集約（hora=運搬役、多ロン合算対応）
- **層3**: dueling第2ヘッド `Q_chip`、`Q_total = Q_main + β_sel·Q_chip`、
  n-step(3) TD、target net(Polyak τ=0.005, chipヘッドのみ)、β_sel warmup
- **設計の肝**: Q_main(rank+素点+順位のMC+GRP)は不変更。チップだけTD化して分離。
  → 検証済みの土台を壊さず、チップ価値だけper-moveで学べる構造。

### 5.2 判明したこと（★商用化で最重要の知見）

**(a) online TDは offline天井に"触れる"。**
- 最良スパイクで鳴き和了率6.73%(offline天井3.03%の倍)、chip実現24%(人間21.75%超)、
  副露Δ+0.58〜+2.92(人間+2.81方向＝赤を選択的に鳴く)を記録。
- **「赤を選択的に鳴いてチップ化する」behaviorは、瞬間的には確かに出現する。**

**(b) だが現設計では安定保持できない（振動する）。**
- 6.73%(step6000)は**振動の山**。延長(step8000-16000)で平均3.16%・副露Δ平均−3.45ppに収束。
- どのCQL強度でも安定した「選択性×天井超え」は出ず、好成績は全て再現しない単発スパイク。

**(c) CQL強度スイープの結論（min_q_weight）:**

| min_q_weight | 鳴き和了率(安定) | 副露Δ(安定) | 評価 |
|---|---|---|---|
| 0 (online_main) | 4-6%(振動) | −2.74 | 無差別鳴き(副露61%暴走) |
| 0.3 | 3.16%(振動) | −3.45 | 量も選択性も安定せず |
| 0.5 | 2.75% | −1.00 | 中途半端 |
| 1.0 | 0.09% | — | 鳴き全潰れ |

→ **CQL強度という単一軸では安定解が無い**ことが確定。

### 5.3 振動の原因仮説（Phase6で潰す対象・優先順）

1. **n_step=3の局末集約 × 逆流距離不足（最有力）**：チップは局末にしか立たず、
   序盤の鳴き判断まで価値が届くのにbootstrap反復回数頼み。届く前に方策が動く＝振動。
2. **自己対戦の非定常性**：相手=自分の最新コピー。方策が動くと学習対象分布も動く。
   → 過去ckpt混合プールで緩和(標準手法)。
3. **CQL強度と鳴き圧の綱引き**：cql強→鳴き潰れ、弱→無差別。今回スイープ済み、
   この軸単独では解けないと判明。

### 5.4 健全だった点（崩壊ではない）
- 全Phase5通じて avg_rank≈2.5、放銃率15-18%(baseline13.26%から大崩れなし)、loss発散なし。
- 「打牌は壊れていない、が目的behaviorが安定しない」状態。失敗ではなく**未到達**。

---

## 5. 商用化に向けたロードマップ（Phase6以降の選択肢）

### 短期（既存設計の安定化）
- **Phase6a: n_step延長(3→5,7)** ★最優先。振動の最有力犯人を構造的に潰す。
- **Phase6b: 相手プール混合**（過去ckpt mix）で自己対戦の非定常性を緩和。
- これらで「選択性×天井超え」が安定保持できるかを検証。

### 中長期（商用版の根本設計・要判断）
- 現設計は「天鳳特化Mortalへのチップ報酬後付け」。商用フリー雀荘AIとして本気なら：
  - **学習データ**：2009天鳳はチップ無しルール＝チップ最適の手本が原理的に無い。
    チップありルールの棋譜 or 完全自己対戦からの学習を検討。
  - **Q_chip設計**：後付けヘッドでなく、チップ最適化をfirst-classにした構造の再設計。
  - **ルール網羅**：実フリー雀荘の多様なチップ規定(店ルール差)への対応。
- これは大きな分岐。Phase6a/bの結果を見て、「既存改良で行けるか/再設計が要るか」を判断。

### 評価軸（商用品質の定義・要明確化）
- 現状は「鳴き和了率」「副露Δ」「chip実現率」で見ているが、商用なら最終的には
  **「チップありルールでの収支(期待チップ込み和了点)」**で測るべき。
  指標を打牌統計から収支ベースへ移す検討。

---

## 6. 再開時に最初に読むファイル
1. 本ファイル §0-4（プロジェクト全体像と Phase5 到達点）
2. freeparlor/docs/phase4d_chip_realize.md（offline天井の正体）
3. freeparlor/docs/online_r_chip_layer{1,2,3}.md（TD実装の詳細）
4. freeparlor/docs/online_cql_min_q_weight_sweep.md（Phase5の振動データ）
5. freeparlor/docs/online_diag_{a,b}_*.md（無差別鳴みの原因切り分け）

---

## 7. 環境メモ
- **run dir 再利用禁止**（証拠上書き防止）。各実験は新規 run dir を使い、
  名前に日時 suffix を付ける（例 `smoke_p2c_20260703_0541`）。既存 dir への
  上書き・再実行は禁止。
- 作業箱 `wsl -d mahjong` のみ。ユーザー小文字 `gamba`(`/home/gamba`)。
- リポジトリ `/home/gamba/mahjong/Mortal`。`conda activate mortal`。
- **WSL2 メモリ**: `.wslconfig` `memory=24GB` / `swap=16GB`（32GB 禁止・ホスト総量32GB）。
  変更後は `wsl --shutdown` → 再起動 → `nvidia-smi` で GPU 確認。
- 学習は `runs/` の spawn ランチャ経由（WSL2 CUDA fork回避）。
- libriichi改修後: `cargo build -p libriichi --lib --release`
  → `cp target/release/deps/libriichi.so mortal/libriichi.so`。
- **GPU ワークロードは常に1系統**: 学習（server/trainer/client）と eval（OneVsThree 自己対戦）
  の**同時実行禁止**。OOM 回避のため、eval 完走後に学習を起動する（逆も同様）。
- **診断/スモーク config**: inline `test_play` 無効 = `control.test_every > ppo.max_steps`
  （trainer 終了時の arena 400 局を走らせない）。評価は `run_eval_ppo_smoke_sanity.sh` で別途。
- **tmux 必須**: 学習・eval とも `run_ppo_p2_smoke.sh` / `run_eval_ppo_smoke_sanity.sh` が
  自動で tmux セッション起動（Cursor 切断と分離）。`tmux attach -t <session>` で監視。
- **online 3プロセス構成**: server×1 / trainer×1 / client×3。RTX5060単機で生成律速
  解消済み（client3並列でdrain空待ち≈0）。warmup2000≈50分、step/秒≈0.67。
- **インフラ注意**: online は stale プロセス混入・drain競合の事故が頻発した。
  各run起動前に pkill + `ss -tlnp | grep 5000` で残党ゼロ確認、buffer/drainクリア必須。
  **eval_sanity も同様**（`run_eval_ppo_smoke_sanity.sh` が起動前チェックを実施）。
  server/client 不要の単独プロセスだが、残存 server が GPU/ポートを占有するとハングする。
- 比較は**打牌統計**で（自己対戦avg_rankは常に≈2.5）。
  ※学習時test_play(grp_baseline3席)はavg_rank≈1.0、これは参考外。

## 8. 主要パラメータ現状値
- offline健全: β≤0.3, lambda_opp=0.3
- online: enable_cql_online=true 推奨(=falseだと無差別鳴き暴走), min_q_weight 未確定
  (0.3が最良だが不安定), chip_n_step=3(延長検討), chip_target_tau=0.005,
  chip_weight=1.0, beta_sel_max=0.3(warmup2000/ramp2000)