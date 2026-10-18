# 差し戻しタスク: anchor 系列 Arm K の NaN 勾配修正 + 検定(20)強化 + pool_draw 競合

**日付:** 2026-07-25（監督3段検証の結果、`a20517b`/`1cb6097` の Arm K を差し戻し）
**宛先:** 実装エージェント（Cursor Composer / Sonnet）
**正となる仕様:** `freeparlor/docs/design/anchored_ppo_design.md`（**凍結済み・変更禁止**）
**前タスク:** `anchor_impl_task_20260725.md`（Arm C は合格。本書は Arm K の欠陥修正のみ）

## 0. 監督検証の結論（何が起きたか）

**Arm C は合格。** OFF 時に `random.random()` を消費しない短絡、旧実装との RNG
ストリーム同一性検証（4シード）、anchor 不在の loud ValueError、レガシー
`mortal.pth` の読み込み、config の単一変数 diff、launcher の bash -n、
eval 3本への assert — いずれも監督側で独立再実行して確認済み。

**Arm K は発進すれば最初の optimizer step でネットワーク全体が NaN 化する。**
検定 (19)(20) は CPU で2回とも PASS するが、その検定が欠陥を検出できていない。

---

## 1. 【ブロッカー】`masked_kl_forward` が NaN 勾配を生む

### 症状（監督側の実測、実物の 192×40 ネット・kl_beta=0.1）

```
total loss finite : True          <- 前向きの損失は健全に見える
kl_ref value      : 0.0016564
trainable params with non-finite grad: 409/411
```

### 原因

`mortal/ppo.py` の `masked_kl_forward`:

```python
probs = masked_softmax(logits, mask)          # 非合法手は 0
logp = masked_log_softmax(logits, mask)       # 非合法手は -inf
logp_ref = masked_log_softmax(ref_logits, mask)  # 非合法手は -inf
contrib = probs * (logp - logp_ref)           # (-inf) - (-inf) = NaN -> 0 * NaN = NaN
return contrib.masked_fill(~mask, 0.0).sum(-1).mean()
```

`masked_fill` は**前向きの値だけ**を修復する。逆伝播では積の勾配
`d(contrib)/d(probs) = (logp - logp_ref)` が非合法手で NaN のままであり、
`grad_probs = 0 * NaN = NaN` が softmax backward を通って全パラメータへ伝播する。

### 修正（掛ける「前」にマスクする。監督側で等価性と NaN 解消を実測済み）

```python
diff = (logp - logp_ref).masked_fill(~mask, 0.0)
return (probs * diff).sum(-1).mean()
```

実測: 前向き値は現行と完全一致（`0.501216` vs `0.501216`、`allclose` PASS）、
`grad_nan=False`。**ただしこれは監督側が示す修正方向であって、同等以上に堅い
実装（合法手だけを gather して計算する等）を選んでも構わない** — 要件は
「前向き値が現行と一致し、勾配が全て有限であること」。

## 2. 【必須】検定 (20) の強化 — なぜ欠陥を通したか

現行の (20) は2箇所が弱い。**同じ穴を二度通さないため両方直すこと。**

- **(20)(c) が勾配の有限性を見ていない。** `assert p.grad is not None` は
  NaN 勾配でも成立する。→ **全 train パラメータについて
  `torch.isfinite(p.grad).all()` を assert** すること（`ref` 側の
  `grad is None` assert は現行のまま維持）。
- **(20)(b) が循環検証。** `expected` が実装と同じヘルパで同じ式を1行ずつ
  再現しているため、実装が何であれ一致してしまう。→ **実装ヘルパに依存しない
  独立の期待値**に差し替えること。例: 合法手2本だけの密なケースを手で組み、
  `p·log(p/q)` の閉形式（Python の `math.log` で計算した定数）と突き合わせる。
  `ref=pi -> KL=0`・単一合法手で有限、の2つは現行のまま良い（本物の検査）。
- 追加推奨: **非合法手を多く含む疎な mask**（実戦同様、46 中 5 合法）での
  勾配有限性ケースを明示的に入れる。今回の欠陥はまさにこの形状で出る。

## 3. 【軽微】pool_draw の `draw_kind` にスレッド競合の余地

`mortal/ppo_pool_engine.py` の `_ckpt_for_game` は `pool.sample()` の直後に
`pool.last_draw_kind` を**別途読み直す**。react_batch が並行に入ると
kind と checkpoint がずれ得る（GIL はバイトコード間で切り替わる）。
発進ゲートは `draw_kind` だけで採択率を数えているため、ゲート統計が汚れる。

修正の方向（どちらでも良い、実装者判断）:
- `sample()` が `(path, kind)` を返すようにして属性の読み直しをなくす、または
- `check_anchor_launch_gate.py` の Arm C 集計を race-free な `checkpoint` フィールド
  （anchor パスとの一致）でも数え、`draw_kind` 集計と一致することを assert する

## 3b. 【必須】発進ゲート Arm K を §5-a1 amendment に合わせる

設計 §5 原文の「全バッチ `kl_ref_mean > 0`」は**成立し得ない**（trainer_step 0 で
policy と ref は state_dict がビット同一 → 監督側実測で `kl_ref_mean` は厳密に 0.0）。
**Gamba 裁定により `anchored_ppo_design.md` §5-a1 として amendment 済み**。
`check_anchor_launch_gate.py` の `run_arm_k` を §5-a1 の3条件へ書き換えること
（設計書 §5-a1 が正。本書の要約と食い違ったら設計書に従う）:

1. 窓 W = `event='kl_anchor'` かつ `0 ≤ trainer_step ≤ 200`（W が空なら FAIL）。
   W の全レコードで `kl_beta` == 設定値
2. W の全レコードで `kl_ref_mean` が有限（NaN/inf なし）かつ **≥ 0**
   （現行の `v <= 0` で FAIL する実装を is-finite + `>= 0` に改める）
3. 後半窓 W₂ = `101 ≤ trainer_step ≤ 200` の**全レコード**で `kl_ref_mean` > 0
   （W₂ が空なら FAIL）
4. `trainer_step 0` の値は**合否に使わず INFO として出力**する
   （0.0 ちょうどなら ref が step0 方策と同一である陽性対照）

検証は合成 `ppo_diag.jsonl`（PASS 系1本 + FAIL 系: 窓空 / kl_beta 不一致 /
NaN 混入 / W₂ が全部 0 の張り付き、の各経路）を CPU で作って exit code を実演すること。

## 4. 触ってはいけないもの

- **Arm C の合格済み部分**（`opponent_pool.py` の anchor 分岐ロジック・config・
  launcher・検定(19)）は §3 の競合対応以外の変更禁止
- **`anchored_ppo_design.md` 本体は編集禁止**（凍結文書。§5-a1 は監督側で記録済み）
- libriichi、`client.py` rollout 経路、報酬3ストリーム、DRCA ハーネス4本、
  既存検定 (1)–(18) のロジック、進行中の DRCA 第3枠 run
- **GPU を使わないこと**（第3枠が占有中）。本タスクは全て CPU で完結する

## 5. 検証と報告

1. CPU で実施し証拠を貼る: 強化後の検定 (19)(20) の実行ログ、
   §3b のゲート合成データ実演（PASS/FAIL 各経路の exit code）、
   修正前後で `masked_kl_forward` の**前向き値が一致**することの実測、
   実物の 192×40 ネット + 疎 mask + kl_beta=0.1 で
   **全 train パラメータの勾配が有限**であることの実測
2. 検定は**反復実行**（最低2回）して結果を貼る（単発 PASS は採らない）
3. 変更していないものを明示列挙（§4 の各項目）
4. commit & push 後に `git ls-remote origin | grep ppo-migration` の出力を貼付
5. CLAUDE.md「現在の状態」の anchor 実装エントリを実態に合わせて更新
   （現エントリは Arm K を健全と記載しており誤り）
6. verify 全20本・400-step スモークは**やらない**（第3枠完走後の発進 preflight）
