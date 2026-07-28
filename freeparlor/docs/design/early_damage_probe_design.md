# 初期損傷プローブ設計書 — step 0–2000 の内部形状を測る短 run

**日付:** 2026-07-28
**ステータス:** **事前登録（診断・判定非関与）。** 本書 commit をもって設計を固定し、
run 中の変更を禁じる。ただし本 run は**どの仮説の判定にも用いない** —
`anchored_ppo_design.md` §6/§6a の判定条件・再走規定には一切触れない
**起草:** 実装エージェント（Claude Code）。**採否・発進は Gamba 裁定**
**根拠文書:** `../reports/anchor_checkpoint_trajectory_20260728.md`（損傷の時間分布）、
`../reports/ppo_optimization_health_20260725.md`（L1 の実測）、
`../ops/session_handover_20260728.md` §1b

---

## 0. 問い

`anchor_checkpoint_trajectory_20260728.md` が確定させたのは:

| 量 | init | step2000（全体の 12.5%） | step16000 |
|---|---:|---:|---:|
| 放銃率（Arm K） | 11.85% | **13.89%（z=+3.91）** | 13.16%（z=+2.51） |
| 降りの中断率（Arm K） | 23.78% | **31.78%** | 33.75% |
| 立直和了 割合（Arm K） | 54.32% | **81.24%** | 83.35% |
| チップ /SE（Arm K） | — | **+2.85SE（全 ckpt 中最大）** | +2.18SE |

**損傷もチップ獲得も最初の 2000 step でほぼ完了している。** 残り 14000 step は
維持（K）か喪失（C）かの期間でしかない。

しかし **step 0–2000 の内部で何が起きたかは測れない** — `save_every = 2000` なので
その区間に checkpoint が1つも存在しないためである（同書 §3-3 が残した唯一の宿題）。

本 run が答える問い:

> **損傷は滑らかに進行するのか、それともどこかで相転移するのか。**
> 立直マキシマリズムへの遷移（54% → 81%）と放銃劣化（11.85% → 13.89%）は
> 同時に起きるのか、順序があるのか。

**本書は仮説を判定しない。** 形状を測るだけである。判定条件を書かないのは
意図的で、「どんな形が出たら何を結論するか」を事前に固定できるだけの
先行知見が無いため（Stage3 v1 のゲート較正ミスと同じ理由 —
`anchored_ppo_design.md` §5）。

## 1. なぜ「16k run をもう1本」ではないのか

判定窓 step 8000–16000 という既存の設計は「収束後の性質」を測るためのもので、
**損傷が起きる区間（0–2000）を1点も観測しない**。本 run は 2000 step で止める:

- 2000 step ≈ **3.1h**（Arm K 実績 16000 step / 24.6h から線形換算）。
  16k run（≈2日）の 1/8 で、GPU 直列枠をほとんど食わない
- 損傷はこの区間で完了しているので、それ以降を回しても本問いには答えない

## 2. 単一変数 — 診断 checkpoint ストリーム

### 2a. `save_every` を下げてはいけない（**この設計書の要点**）

素朴な案は `save_every = 2000 → 100` だが、**これは2変数を同時に動かす**。

`OpponentPool.list_checkpoints()`（`mortal/opponent_pool.py:41-45`）は
run dir の `checkpoints/step_*.pth` を **glob して相手プールを構成する**。
一方 `save_checkpoint()`（`mortal/train_ppo.py:158-163`）は同じディレクトリへ書く。
したがって `save_every` は **測定の粒度と相手分布の両方**を支配している:

| | save_every = 2000（現行） | save_every = 100 にした場合 |
|---|---|---|
| step 2000 時点の pool 内 checkpoint 数 | 1（step_000000 のみ） | **21** |
| `latest`（確率 0.5） | step_000000 = init | **直近 100 step 前の自分** |
| `past_k = 5` の中身 | 空（fallback） | **直近 500 step の自分5点** |

つまり素朴案は「観測を細かくした run」ではなく「**相手が急に自分の直近版になった
別の学習系**」になる。Arm K との比較が成立せず、測りたい形状も別物になる。

### 2b. 採る方式 — pool が見ないディレクトリへ書く

`[control]` に **`diag_save_every`**（既定 **0 = OFF**）を追加し、非ゼロのとき
`checkpoints_diag/step_%06d.pth` へ**追加で**保存する。

- **`checkpoints/` には一切書かない** → `OpponentPool` の glob 対象は不変。
  相手分布は Arm K とビット同一の構成のまま
- **既定 0 では `save_checkpoint` の呼び出し自体が増えない** = 既存経路ビット不変
  （`p_enrich` / `call_bonus_b` / `anchor_prob` / `kl_beta` と同じ「設計された OFF」）
- 学習ループ・損失・報酬・rollout には一切触れない。**観測のみ**の変更であり、
  `robust_selfplay_ppo_design.md` §6c の「観測のみ・自動停止なし」と同じ規律に従う

### 2c. 基底 config は Arm K

`freeparlor/configs/ppo_anchor_k.toml` を基底にする（`kl_beta = 0.1` のまま）。理由:

1. **比較対象が実在する**。軌跡測定は C と K について n=800 で取ってあり、
   本 run の step_002000 は **Arm K の step2000（放銃 z=+3.91 / チップ +2.85SE）を
   統計的に再現するはず**。再現しなければ「同じ系ではない」ことになり、
   run 自体を疑う陽性対照になる（決定論的一致は期待しない — RNG ストリームが違う）
2. K は現行で最良の arm であり、§6a の kl_beta=0.4 再走が発火しうる先でもある
3. 損傷は C（z=+4.53）と K（z=+3.91）でこの区間ほぼ同じなので、
   アンカーの有無はこの窓では支配的でない = K を選んでも一般性をあまり失わない

**`ppo_anchor_k.toml` との diff は run パス + `max_steps` + `diag_save_every` のみ。**

## 3. run 構成（凍結）

| 項目 | 値 |
|---|---|
| 基底 config | `freeparlor/configs/ppo_early_probe.toml`（`ppo_anchor_k.toml` から派生） |
| init | `beta1_huber_192x40/mortal.pth`（全 stage と同一） |
| `kl_beta` | **0.1**（Arm K と同一。単一変数を保つため変えない） |
| `anchor_prob` | キー不在 = OFF（Arm K と同一） |
| `max_steps` | **2000** |
| `save_every` | **2000**（変更しない。pool 汚染を避けるため） |
| `diag_save_every` | **100** → step 100,200,…,2000 の **20 点** |
| run dir | `runs/ppo/early_probe_<日時>`（日時 suffix・再利用禁止） |
| 監視 | 既存4項目 + バックログ11 で追加した2項目（全て 0 を期待） |

## 4. eval 設計（凍結）

判定に使わないので n は判定標準（800）より小さくてよいが、**形状を見るには点数が要る**。

| 項目 | 値 |
|---|---|
| 対象 checkpoint | step 200, 400, …, 2000 の **10 点**（100 刻みの半分を間引き） |
| ハーネス | 既存 `grp_baseline` 1v3（argmax / guard ON / 自然分布）— 判定と同一経路 |
| n | **400 半荘 / 脚**、seeds **[10000, 10100)**（判定 seed 範囲 [10000,10200) の前半） |
| baseline 脚 | init（決定論的に同一なので既存の init 脚を再利用してよい） |
| 指標 | `analyze_fundamentals_1v3.py`（放銃・和了・avg_rank + 拡張）、
`analyze_freeparlor_pnl_1v3.py`（チップ）、`analyze_genbutsu_discipline.py`（降りの中断率）、
`diagnose_agari_composition.py`（立直/ダマ/副露の構成比） |

**n=400 の SE は判定標準（n=800）の約 1.41 倍**である。本 run の数値を判定文書の
数値と並べるときは必ずこれを明記する（`anchor_checkpoint_trajectory_20260728.md` の
値は n=800、本 run は n=400 で、**SE が違う表を混ぜない**）。

### 4a. 測定点の amendment（2026-07-28、発進後・インシデント起因）

**終端を step 2000 → step 1900 に変更する。** 測定点数（10）は変えない:
**step 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 1900**。

理由は `early_probe_20260728_202533` の完走時に **step_002000.pth が書き込み途中で
切り詰められた**こと（108.8MB / 正常 130.7MB、`torch.load` が失敗）。原因は launcher の
cleanup 修正（2026-07-28、バックログ4 追跡調査）で trainer の実ワーカーを
直接 reap するようにした結果、監視ループが完走を検知してから cleanup までの間に
SIGTERM が保存処理に重なったこと。**残り 19 点（step 100–1900）は全て健全**
（`torch.load` 成功・`steps` フィールド一致）。

**この amendment が測定の趣旨を損なわない根拠:**

- 本 run の目的は **step 0–2000 の内部形状**であり、終端の1点ではない
- **step2000 は既に Arm K で n=800 で測られている**
  （放銃 13.89% / z=+3.91、チップ +2.85SE — `anchor_checkpoint_trajectory_20260728.md`）。
  本 run の n=400 より**精度の高い測定が既にある**ため、終端の情報価値は元々低い
- §5-1 の陽性対照（Arm K の step2000 再現）は step_001900 で行い、
  **100 step ぶん手前である点を明記して読む**

launcher は同 commit で修正済み（完走検知後に最終 checkpoint の保存ログを待ってから
cleanup する。隔離レプリカで確認/未確認の両経路を実演）。**再走はしない** —
上記のとおり終端1点の損失は本測定の問いに影響しないため、3.1h の GPU を
再投入する価値がない。

## 5. 陽性対照・健全性チェック（発進後に確認する項目）

1. **step_002000 が Arm K の step2000 を再現するか** — 放銃 z、チップ /SE、
   立直和了割合、降りの中断率。n が違うので点推定の一致ではなく
   **SE 圏内での整合**を見る。大きく外れたら run 構成を疑う
2. **相手プールが Arm K と同じ挙動か** — pool draw ログで、
   step 2000 到達まで `latest` が常に `step_000000`（= init）であること。
   `checkpoints_diag/` が glob されていないことの実証になる
3. **`kl_anchor` イベント** — 全バッチ kl_beta=0.1、step0 = 0.0 ちょうど（陽性対照）、
   非有限 0 件（`anchored_ppo_design.md` §5-a1 と同じ確認）
4. 監視6項目すべて 0

## 6. 何が分かれば何と言えるか（討議用・判定条件ではない）

| 観測される形 | 読み（**仮説であって判定ではない**） |
|---|---|
| 放銃劣化が step 200–400 で大半完了 | 損傷は「学習の蓄積」ではなく**初期の急激な方策シフト**。L1（staleness / clip）の初期挙動が関与する余地が大きい |
| 放銃劣化が 0–2000 で線形 | 損傷は蓄積型。区間を細かく見る意味は薄く、L1 の初期特異性という筋も弱まる |
| 立直シフトが放銃劣化に**先行** | 「立直に寄った結果として押し引きが壊れた」という因果順序の傍証 |
| 放銃劣化が立直シフトに**先行** | 逆。押し引きが先に壊れ、立直特化はその後の適応 |

**いずれも因果の証明にはならない**（単一 run・介入なし）。次の実験の設計材料である。

## 7. コスト

| 項目 | 見積り |
|---|---|
| 実装（`diag_save_every` + config + launcher + 検定） | 小（0.5日以内） |
| run | GPU **≈3.1h** |
| drain | 2000 step 分 ≈ **30–45GB**（16k run の 1/8。現在空き 909GB） |
| 診断 checkpoint | 20 × ~118MB ≈ **2.4GB** |
| eval | 10 ckpt × n=400 1v3。GPU **数時間**（判定バッテリーより軽い） |

**合計で GPU 直列枠を 1 日弱**。16k run 1本（≈2日 + drain 230–370GB）の半分以下。

## 8. 検定の拡張

既存 `verify_ppo_p1.py` に1本追加する:

**(21) diag checkpoint ストリームの OFF 恒等性と非汚染**
  - (a) `diag_save_every` 不在 / 0 で `save_checkpoint` の書き込み先集合が従来と一致
    （`checkpoints_diag/` が作られないこと）
  - (b) `diag_save_every > 0` のとき `checkpoints_diag/` にのみ増分が出て、
    `checkpoints/` の内容が変わらないこと
  - (c) **`OpponentPool.list_checkpoints()` が `checkpoints_diag/` を拾わないこと**
    （本設計の中核。合成ディレクトリで assert する）

## 9. 凍結記録（本 commit = 事前登録）

- [x] 基底 = `ppo_anchor_k.toml`（kl_beta=0.1、anchor_prob キー不在）
- [x] `max_steps = 2000` / `save_every = 2000`（不変）/ `diag_save_every = 100`
- [x] eval = 10 ckpt（step 200 刻み）× 1v3 n=400 seeds [10000,10100)
- [x] **判定条件は置かない**（診断・判定非関与）。本 run の結果で
      `anchored_ppo_design.md` の判定・再走規定を変更しない
- [x] run 命名 = `early_probe_<日時>`

**未確定（Gamba 裁定事項）:** 発進の可否と、GPU 直列枠における §6a の
kl_beta=0.4 再走との優先順位。
