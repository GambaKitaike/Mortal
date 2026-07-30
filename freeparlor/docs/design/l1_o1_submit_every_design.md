# L1 O1（`submit_every` 引き下げ）設計書 — 最適化衛生の単一変数対照実験

**日付:** 2026-07-30
**ステータス:** **事前登録案（DRAFT）**。§10 の裁定事項に Gamba の裁定が付いた時点で §11 を
確定させ、その commit を事前登録とする。**それまで発進しない**
**起草:** 実装エージェント（Claude Code）。設計判断・採否は Gamba
**位置づけ:** `../ops/policy_session_0b_decisions_20260729.md` §2 が定めた次の軸
（**L1 の O1 → O3**）の O1。同 §2 は「発進前に design md で判定条件を事前登録すること」を
要求しており、本書がそれに当たる（`robust_selfplay_ppo_design.md` は DRAFT で判定条件を持たない）
**根拠文書:**
- `../reports/ppo_optimization_health_20260725.md` — L1 の一次証拠（8 run 横断）
- `robust_selfplay_ppo_design.md` §2 / §2a — 候補 O1–O5 と推奨順序
- `../ops/policy_session_0b_decisions_20260729.md` §1（seed 方針）/ §2（次の軸）
- `../reports/early_damage_probe_result_20260729.md` §9 — run 間ばらつきが効果量と同程度
- `anchored_ppo_design.md` §6（判定条件の書式）/ §5-a1（ゲート設計の教訓）

> **本書は §1a・§1b・§1c に新規実測を含む**（本書のために測定した。判定非関与）。
> 計測器は `freeparlor/scripts/analyze_staleness_decomposition.py`（新規・read-only）。

---

## 0. 目的（改善策ではなく因果の検定）

0b 裁定 §2 の留保をそのまま引く:

> L1 が基礎劣化の原因である証拠は無い（`ppo_optimization_health_20260725.md` §5）。
> O1/O3 はその因果を検定するための対照実験であって、**改善策として登録するのではない**。

検定したい命題は1つだけ:

> **H(O1): off-policy staleness の削減は、PPO が init の基礎技能（放銃回避）を
> 劣化させる度合いを小さくする。**

## 1. 介入定義（単一変数）

`[control] submit_every = 50 → 10`。**config 1行。学習コード・libriichi・eval 経路すべて無変更。**

trainer は `steps % submit_every == 0` のたびに param を server へ submit し
`trainer_param_version` を +1 する（`mortal/train_ppo.py:605-608`）。client は session の
先頭で server の最新 version を1回だけ取得する（`mortal/client.py:217-234`）。
したがって `submit_every` は「client が受け取れるパラメータの鮮度の上限」を決める。

### 1a. この変数が実際に動かす量（**新規実測**。O1 の効果に上限を与える）

`batch_lag` から staleness を **optimizer step 単位で復元**した（復元法は
`analyze_staleness_decomposition.py` の docstring。version v のスナップショット =
step `(v-1)*submit_every`、resume run は base 補正）。

| run | staleness mean | = transit | + 量子化 | S=10 の予測 |
|---|---:|---:|---:|---:|
| stage1_20260706_020120_resume（step10000–16000） | 91.5 | 67.0 | 24.5 | **71.5（−21.9%）** |
| stage3_20260712_033403 | 92.0 | 67.5 | 24.5 | **72.0（−21.7%）** |
| anchor_k_20260727_000805 | 93.4 | 68.9 | 24.5 | **73.4（−21.4%）** |
| anchor_k_b04_20260729_150631 | 94.4 | 69.9 | 24.5 | **74.4（−21.2%）** |

- **量子化成分は 4 run すべてで 24.5 = (submit_every−1)/2 に厳密一致**（分解の妥当性の陽性対照）。
- **transit 成分（67–70 step）は `submit_every` では動かない。** 実体は
  「client は 1 session = **20 半荘**（`[train_play.clientN] games = 20`）を同一 param で
  打ち切り、3 client 分が drain キューに並ぶ」構造で、drain 1 世代 ≈ 20.9 traj
  （b04: 765 世代 / 16000 step）と整合する。
- ⇒ **O1 は staleness を 1/5 にする介入ではない。動かせるのは全体の約 26% しかない
  量子化成分だけで、実効は −21〜22%**（92.5 → 72.5 step 前後）。

> ⚠ **この事実は本書のために測って初めて分かった。** `robust_selfplay_ppo_design.md` §2a の
> 「staleness の直接削減」という記述と `ppo_optimization_health_20260725.md` §3b の
> 「lag 1 単位 = 50 optimizer step」は誤りではないが、**lag（param_version 単位）だけを見て
> 「submit_every を 1/5 にすれば staleness も 1/5」と読むと 5 倍の過大評価になる。**
> transit を直接叩く別の変数（1 session の半荘数）については §10 裁定事項2 に提案として分離した。

### 1b. clip_fraction がどれだけ動くかは**事前予測しない**

staleness → clip_fraction の量級を較正しようとしたが、**既存データからは較正できない**。
同一 run 内で lag 別に clip@epoch1 を比べると、**符号が run 間で一致しない**:

| run | clip@e1 (lag=1) | clip@e1 (lag=2) | 差 | z |
|---|---:|---:|---:|---:|
| stage3_20260712_033403 | 0.2151 | 0.2103 | **−0.0049** | −4.21 |
| anchor_k_20260727_000805 | 0.2153 | 0.2124 | **−0.0029** | −2.78 |
| anchor_k_b04_20260729_150631 | 0.2050 | 0.2072 | **+0.0022** | +2.14 |
| stage1_20260706_020120_resume | 0.2038 | 0.2009 | −0.0030 | −1.71 (n.s.) |

これは横断比較（同一 run 内で lag が違うバッチの比較）であって因果ではなく、
step トレンド・session 内位置・バッチ構成と交絡している。**よって「clip がここまで下がるはず」
という事前予測は置かない**（`anchored_ppo_design.md` §3 が β の事前較正を不採用にしたのと同じ理由。
較正情報が無いときに数値を置くのは Stage3 v1 の較正ミスと同型）。
clip@e1 は §3-3 の**副次（判定外・記録）**として扱う。

### 1c. スループットへの影響（**新規実測**、`robust_selfplay_ppo_design.md` §2a の宿題）

trainer.log の timestamp から submit 1 回のオーバーヘッドを実測:

| run | submit 1回 | s/step（実効） | S=10 の追加コスト |
|---|---:|---:|---:|
| stage3 | 0.626s | 5.041 | +13.4 分（+0.99%） |
| anchor_k | 0.490s | 5.633 | +10.5 分（+0.70%） |
| b04 | 0.523s | 5.485 | +11.2 分（+0.76%） |

- **許容範囲**（24h の run に対し +11〜13 分 = +1% 未満）。
- 実効 5.0–5.6 s/step に対し step 間隔の median は 1.2–1.4s であり、**trainer は
  データ待ちが支配**（client 律速）。submit の増分はその待ち時間に吸収される見込みで、
  上表は上限側の見積り。
- **1 optimizer step の意味は変わらない**（O2 と違い batch 束ねをしない）ので、
  **判定窓を step 基準のまま使える**（`robust_selfplay_ppo_design.md` §2a ⚠ の罠を踏まない）。
- `save_every = 2000` は 10 の倍数なので、`train_ppo.py:619-622` の追加 submit は発生しない
  （version と step の対応が保たれ、§4 のゲートが機械的に検査できる）。

## 2. 対照（baseline）の選択

**推奨: plain PPO（Stage1 相当 config、anchor なし）を base にする。**
= O1 の config は `freeparlor/configs/ppo_p3_stage1.toml` との diff が
**run パス + `submit_every` の1行のみ**。

理由（検出力）:

| base 候補 | 参照 run | 劣化の headroom（放銃差 vs init） | 計測器 |
|---|---|---:|---|
| **plain PPO（推奨）** | Stage1-16000 | **+3.06pp**（z=+3.93、n=400） | n=400（§7 で n=800 へ再測を提案） |
| Arm K（β=0.1） | anchor_k-16000 | +1.31pp（z=+2.51、n=800） | n=800（同一計測器） |

- `early_damage_probe_result_20260729.md` §9 の実測では、**同一 config の run 間ばらつきは
  ~1pp 級**（step2000 で +2.04 / +0.12 / +0.96pp）。Arm K を base にすると
  **検出すべき効果（≤1.31pp）がばらつきと同じ大きさになり、実験がほぼ無力化する。**
- plain PPO なら headroom が約 3 倍あり、かつ L2（アンカー）と混ざらないので
  「L1 単独の因果」という問いに正しく答える。
- 代償: 参照 run（Stage1）の 1v3 が n=400 で、計測器が判定用の n=800 と揃わない。
  → §7 の**参照脚の再測**（GPU ~1h）で解消できる。裁定事項1。

## 3. 判定条件（**裁定後に凍結。以後変更禁止**）

すべて **step16000 checkpoint**、既存ハーネス（grp_baseline 1v3）、**argmax / guard ON /
自然分布**、`analyze_fundamentals_1v3.py`。**anchor 系列（C / K / b04）と同一の計測器・
同一 seed・同一 n** にしてあるので、判定値はそのまま横並び比較できる。

1. **主判定1（基礎維持 = H(O1) の直接の読み出し）**:
   放銃率(ckpt) − 放銃率(init) の差が半荘クラスタ robust SE で **z < 2**。
   1v3 は **両脚とも n=800 半荘**（seeds [10000,10200)、4 半荘/seed、座席ローテは既存ハーネス準拠）。
2. **主判定2（経済適応の保持）**: チップ/半荘 (ckpt − init) が **+方向 かつ ≥1SE**。
   （anchor 系列と同一。O1 では「獲得」ではなく「潰していないこと」の確認に当たる）
3. **副次（判定外・記録のみ）**:
   - **staleness の実測**（`analyze_staleness_decomposition.py`）と **clip@epoch1**
     — 介入が機構レベルで効いたかの読み出し。§1b のとおり量級の事前予測は置かない
   - baseline 比: 放銃劣化幅の縮小（Δ_O1 − Δ_ref）。**cross-run 比較なので run 間ばらつきの
     留保が付く**（§8）
   - 和了率 / avg_rank / 順位分布 / 素点 / 順位点（`analyze_fundamentals_1v3.py` の基本＋拡張）
   - 2レンズ規律: sampled `action_mass`（立直 / 鳴き）
   - 0b 裁定 §3 議題1 の再検討条件に対応する参考副読:
     **有役テンパイ機会の見送り率**と**向聴を外した率**（`policy_quality_metrics_20260728.md` の指標。
     判定条件ではない）
   - メタ対決（ckpt vs init×3）+ ミラー較正脚
4. **分岐表と因果の読み**:

| | 判定1 ○（z<2） | 判定1 ✗（z≥2） |
|---|---|---|
| **判定2 ○** | **H(O1) 支持**（staleness が劣化に寄与）。§5 の 2 seed 目へ | 劣化幅の縮小を副次で確認 → 部分的寄与（Arm K 型）。2 seed 目の可否は §5 |
| **判定2 ✗** | 基礎は守れたがチップを失った → O1 は配備価値を持たない。因果の証拠としては残る | **H(O1) 不支持**（少なくとも量子化成分の削減では動かない） |

- 「判定1 ○」は**この 1 run では H(O1) の確証にならない**（§8 の留保）。
  0b §1 の逐次スクリーニングに従い **2 seed 目**で再現を見る。
- 「判定1 ✗ かつ劣化幅も baseline と同程度」の場合、否定されるのは
  **「量子化成分 24.5 step の削減で足りる」という仮説**だけであり、
  staleness 一般（transit 67–70 step を含む）は否定されない（§1a・§10 裁定事項2）。

### 3a. 再走規定

**再走しない。** `anchored_ppo_design.md` §6a のような機械的再走は置かない
（β のような量級パラメータが無く、`submit_every` は「下げる」方向に自明な次の値が無い）。
結果が「効果不確実」なら §5 の seed 規律に従って保留し、バックログへ戻す。

## 4. 発進ゲート（機械ゲートのみ。学習応答ゲートは置かない）

`anchored_ppo_design.md` §5-a1 の教訓（環境で偽陽性・偽陰性になる条件を合否にしない）に従い、
**機械的に保証される量のみを合否条件にする。**

**合否（1つでも落ちたら発進中止・run dir は保全）:**

1. 発進前 preflight: 残党チェック（実装完了報告の後にも）+ libriichi rebuild +
   `verify_ppo_p1.py` **全数 PASS**（本数はスクリプト出力が正）
2. `@step200`: `analyze_staleness_decomposition.py --gate --gate-window 1 200` が **PASS**
   = 窓の全レコードで `trainer_param_version == (step − base) // 10 + 1`
   （**`submit_every=10` が実際に効いていることの機械的検査**。50 のままなら必ず落ちる）
3. `@step200`: 監視6項目ゼロ（keymiss / orphan / fallback / chip / NaN / alive clients 3/3）
4. eval 経路への非漏洩: eval 構成 dump で `p_enrich=0 / call_bonus_b=0 / anchor_prob=0 /
   kl_beta=0`（既存検定 17d。`submit_every` は訓練配管のパラメータで engine 構成に入らないため
   新規 assert は不要 — この点は §8 で明示）

**INFO（報告するが合否条件にしない）:**

- 窓 step 201–500 の staleness mean。**予測 ~72**、参照 run の同窓は 89.5 / 96.2 / 102.8。
  合否にしないのは、transit が run 間で 65–78 step とばらつき、閾値を置くと
  §5-a1 と同型の環境依存な偽判定を生むため
- clip@epoch1 の推移（§1b）

## 5. seed 方針（0b 裁定 §1 の**最初の適用**。§1a が要求する2つの定義）

### 5a. 「別 seed」の操作的定義（**実測に基づく確定案**）

> **「別 seed」= 同一 config・新規 run dir で再走すること。config の変更は不要。**

根拠（一次ソース確認済み）:

- `TrainPlayer.__init__` は **`self.train_key = secrets.randbits(64)`**（`mortal/player.py:217`）。
  訓練 rollout の山・配牌はこの key に依存し、**client プロセスごと・run ごとに OS エントロピーから
  独立に引かれる**。`train_seed` は 10000 固定だが key が違えば別系列。
- したがって **同一 config の 2 本目は、何もしなくても独立な seed の run になる**
  （= `early_damage_probe_result_20260729.md` §9 の replicate 2本が実際にそうだった）。
- 対して **eval は決定論的**（`EVAL_SEED_BASE=10000` / `EVAL_SEED_KEY=0x2000` 固定、
  `eval_grp_baseline_1v3.py:138-140`）。init 脚の全指標 diff = 0.000 が実証済み
  （`fundamentals_significance_pass_20260725.md` §2）。**訓練だけが非決定論**という非対称を確認した。

⚠ **副作用（負債として記録。裁定事項3）**: `train_key` は **どこにもログされていない**
（`mortal/` 全体を grep 済み、run の client ログにも 0 件）。よって
(a) 2 本の run が実際に異なる key だったことは**事後検証できない**、
(b) run の完全な再現は**原理的に不可能**。
**提案**: client 起動時に `train_key` を INFO ログ 1 行で出す（観測のみ・学習に影響しない）。
O1 の run に含めるかは 1 変数規律の解釈なので Gamba 裁定。

### 5b. 「同じ方向」の操作的定義（**2 seed 目を走らせる前に固定する**）

0b §1a は (a) 同じ象限 か (b) 主判定の符号一致 かを選ぶよう要求している。

> **推奨（実装エージェント案・0b §1a の推奨どおり）: (b) 主判定の符号一致。有意性は問わない。**
>
> 2 seed が次の**両方**で一致したとき「同じ方向」= **採用候補**:
>
> - **S1 = sign( Δ放銃(ckpt−init) − Δ放銃_ref )** … `Δ放銃_ref` は §11 で凍結する
>   baseline の登録値（負 = O1 が劣化を縮めた方向）
> - **S2 = sign( Δチップ(ckpt−init) )**
>
> どちらか一方でも食い違えば **「効果不確実」として保留**（捨てない。バックログへ戻す）。

(a) 同じ象限 を採らない理由: 象限は閾値（z=2 / 1SE）の跨ぎで決まるので、
2 本の点推定がほぼ同じでも閾値の両側に落ちれば「食い違い」になる。
run 間ばらつきが効果量と同程度（0b §1 の前提事実）である以上、
**閾値跨ぎの一致を要求するのは検出力的に厳しすぎ、逐次スクリーニングの意図
（GPU 予算を倍にせずに再現性を担保）と矛盾する。**
2 本目は**再現性の確認**であって独立の判定ではない、という位置づけを符号一致が正しく表す。

**採否は Gamba（裁定事項4）。** ここで固定しないまま 2 本目を走らせることは禁止
（post-hoc goalpost になる）。

### 5c. 2 seed 目を走らせる条件

0b §1 のとおり **1 本目が §3 の判定を満たしたときのみ**。
「満たした」の操作的定義は **判定1 ○（z<2）**とする（判定2 は経済側の確認であり、
H(O1) の主張は判定1 が担う）。判定1 ✗ の場合、2 本目は走らせず結果を記録して次の軸（O3）へ。

## 6. 実装（裁定後・別ブランチ）

- ブランチ: **`l1-o1-submit-every`**（`ppo-migration` から分岐。`1 branch = 1 variable`）。
  本設計書は `ppo-migration` に置く（documentation は変数ではない）
- config: `freeparlor/configs/ppo_l1_o1.toml`
  = `ppo_p3_stage1.toml` の run パス置換 + `[control] submit_every = 10` のみ
  （**diff は run パス + 1 行**。凍結後に diff を貼って報告する）
- launcher: 既存 `run_ppo_p3_stage1_inner.sh` を env で再利用（新規ロジック禁止）。
  `MONITOR_HOURS` 既定 48h のまま（実績 22–25h）
- run 命名: `l1_o1_<日時>`（日時 suffix・再利用禁止）
- 検定: 新規 assert は不要（§8）。既存 `verify_ppo_p1.py` 全数 PASS が条件
- **学習コード（`mortal/`）と libriichi は一切触らない**

## 7. eval バッテリー（§3 の計測器）

`run_eval_anchor_c.sh` を RUN_DIR 差し替えで再利用（Arm K / b04 と同じ運用）:
argmax 6ckpt + grp_baseline 1v3 **n=800 両脚** + ミラー較正脚 + メタ対決。

**追加提案（裁定事項1）: 参照脚の再測。**
`stage1_20260706_020120_resume/checkpoints/step_016000.pth` を **n=800・seeds [10000,10200)** で
1 脚だけ測り直し、baseline の登録値 `Δ放銃_ref` を判定と同一の計測器に揃える。

- コスト: 1v3 1 脚 ≈ **GPU 1 時間**（init 脚は決定論的同一なので再利用可 —
  `fundamentals_significance_pass_20260725.md` §2）
- 効果: §2 の唯一の弱点（n=400 vs n=800）が消え、§5b の S1 が同一計測器上の差になる
- 実施時期: **O1 発進前**（後から測ると「baseline を選び直した」に見える）

## 8. 留保（結果の読みに必ず添える）

1. **run 間ばらつき（最重要）**: `early_damage_probe_result_20260729.md` §9 のとおり、
   同一 config の run 間ばらつきは効果量と同程度。**この 1 run で判定1 の閾値を跨ぐか否かを
   決める検出力は限定的**。だから §5 の逐次スクリーニングを掛ける。
   cross-run 比較（baseline 比）はさらに弱い。
2. **O1 が動かすのは staleness の約 26%（量子化成分）だけ**（§1a）。
   判定1 ✗ でも「staleness 仮説の棄却」ではない。
3. **eval SE は保守的**（`check_cluster_se.py`。seed クラスタ + 両脚対応で 0.77–0.81 倍）。
   判定値は現行 SE で出す。結論は「少なくともこの強さ」と読む。
4. **L2/L3 との交互作用は測らない**（base は plain PPO）。アンカー下で L1 を直したらどうなるかは
   別の実験。
5. **`submit_every` は engine 構成に入らない**（`dump_engine_config` の対象は
   `p_enrich` / `call_bonus_b` / `anchor_prob` / `kl_beta`）。理由は
   **訓練配管のパラメータであって eval 経路に存在しないから**
   （eval は server/client 経由でパラメータを受け取らず、checkpoint を直接ロードする）。
   よって「訓練側の介入は eval 経路に漏らさない」規律に対する新規 assert は不要 —
   ただしこの判断自体を本書に明記して、後任が「assert 漏れ」と誤読しないようにする。

## 9. コスト

| 項目 | 見積り |
|---|---|
| 設計・事前登録（本書 + 計測器） | 完了（GPU 不要） |
| 参照脚の再測（§7、裁定事項1） | GPU ~1h |
| run 1本（16k steps） | GPU ~23–25h（実績 22.4–25.0h）+ **+1% 未満**（§1c）+ drain ~230–370GB |
| eval バッテリー | GPU 数時間 |
| 2 seed 目（判定1 ○ のときのみ） | 上記の run + eval を 1 回 |

## 10. 裁定事項（Gamba。**すべて発進前**）

1. **baseline を plain PPO（Stage1 相当）にするか、Arm K にするか。**
   実装側推奨 = **plain PPO**（§2 の検出力）。併せて §7 の**参照脚 n=800 再測**（GPU 1h）の可否
2. **transit 成分（67–70 step、staleness の 74%）を叩く変数を O1 の後に置くか。**
   `[train_play.clientN] games = 20` を下げる案（例 20→5）は config 1 値で
   staleness を ~20 step 台まで落とせる見込みだが、**別の変数**であり
   (a) drain の世代数・IO が 4 倍、(b) client 1 session あたりの
   `train_seed` 前進量が変わる、(c) O1 と併用すると 2 変数になる。
   **本書では実装しない。提案として記録する**（`robust_selfplay_ppo_design.md` の
   L1 候補に O6 として追加するかも含めて裁定）
3. **`train_key` の INFO ログ 1 行を O1 の run に含めるか**（§5a の副作用）。
   観測のみ・学習に影響しないが、厳密には 1 行のコード追加
4. **§5b の「同じ方向」= 主判定の符号一致（推奨）で確定してよいか**
5. **§5c の「2 seed 目に進む条件 = 判定1 ○」で確定してよいか**

## 11. 凍結記録（**Gamba 裁定後に埋める。埋めた commit を事前登録とする**）

- [ ] baseline = ?（plain PPO / Arm K）→ `Δ放銃_ref` = ? pp（出典 run と n を明記）
- [ ] 参照脚 n=800 再測の実施 = ?（する / しない）
- [ ] 判定条件 = §3 本文（判定1: 放銃差 z<2 / 判定2: チップ +方向かつ ≥1SE / 1v3 両脚 n=800・
      seeds [10000,10200)）。**再走なし**（§3a）
- [ ] 判定窓 = step 8000–16000（従来踏襲）／判定は step16000 checkpoint
- [ ] 発進ゲート = §4（機械のみ。staleness の絶対値は INFO）
- [ ] 「別 seed」= 同一 config・新規 run dir（§5a）
- [ ] 「同じ方向」= ?（(b) 主判定の符号一致 / (a) 同じ象限）
- [ ] 2 seed 目に進む条件 = ?（判定1 ○ / その他）
- [ ] `train_key` ログ追加 = ?（する / しない）
- [ ] run 命名 = `l1_o1_<日時>` / ブランチ = `l1-o1-submit-every` / config = `ppo_l1_o1.toml`
