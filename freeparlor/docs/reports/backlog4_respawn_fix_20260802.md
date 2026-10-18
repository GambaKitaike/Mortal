# バックログ4 の根治 — 完走 run の偽トレースバック（`ConnectionRefusedError`）

**ステータス: 修正・run-validation 完了（2026-08-02）。** ブランチ `fix-train-ppo-respawn`。

## 0. 症状

**完走した run の `trainer.log` が例外で終わる**（判定値・checkpoint は無傷）:

```
reached max_steps=16,000, stopping        ← 学習ループは正常終了
saved numbered checkpoint: step_016000.pth
... （子プロセスが再起動）
Traceback ... submit_param ... ConnectionRefusedError: [Errno 111] Connection refused
```

**Arm K（2026-07-27）→ L1 O1（08-01）→ L1 O3（08-02）と3 run 連続**で発生。
そのたびに「完走したのに失敗に見える」状態になり、確認コストを払っていた。

## 1. 原因は**2本の鎖**だった（従来は1本しか見えていなかった）

### 鎖1: `train_ppo.py` の `main()` が完走後に子を再 spawn

`main()` は本家 online DQN 由来の監督ループ:

```python
while True:
    child = Popen(cmd); 
    if (code := child.wait()) != 0: sys.exit(code)
    time.sleep(3)          # ← 完走(exit 0)でも 3 秒後にもう1つ spawn
```

`train_ppo()` は **完走（`max_steps` 到達）でも test_play 境界でも同じ `sys.exit(0)`** を返して
いたため、監督ループは両者を区別できなかった。

### 鎖2: cleanup の `kill $TRAINER_WATCHDOG_PID` が**無効だった**（今回発見）

```bash
trainer_watchdog() {
  (                       # ← 内側サブシェル
    while true; do start_trainer; ...; done
  )
}
trainer_watchdog &
TRAINER_WATCHDOG_PID=$!   # ← 外側の包みの PID
```

`trainer_watchdog &` は関数自体を既にサブシェルで走らせるため、`$!` が指すのは**外側の包み**。
ループを持つ内側の `( ... )` は cleanup の `kill` を生き延びていた。結果:

1. cleanup が trainer を `pkill` → trainer が **143（SIGTERM）** で終了
2. 生き残った watchdog が「143 ≠ 0 = 異常終了」と判断し **trainer を蘇生**
3. 蘇生した trainer が、cleanup に落とされる途中の server へ `submit_param`
   → **`ConnectionRefusedError`**

**一次証拠**: 実 run の `trainer_watchdog.log` に蘇生の記録が残っていた。

| run | trainer_watchdog.log |
|---|---|
| `l1_o1_20260731_012929` | `2026-08-01T05:21:48 trainer exited code=143, restart 1/3 this hour` |
| `l1_o3_20260801_150024` | `2026-08-02T15:23:12 trainer exited code=143, restart 1/3 this hour` |

## 2. 修正

### 2-1. `mortal/train_ppo.py` — 完走を終了コードで伝える

```python
TRAINING_COMPLETE_EXIT_CODE = 21

# train_ppo() 末尾
if online:
    if max_steps and steps >= max_steps:
        sys.exit(TRAINING_COMPLETE_EXIT_CODE)   # 完走 → 再起動させない
    if steps % test_every == 0:
        sys.exit(0)                              # test_play 境界 → 本家どおり作り直す

# main()
code = child.wait()
if code == TRAINING_COMPLETE_EXIT_CODE:
    return              # 外側は exit 0 → launcher の「exit 0 = 正常完走」経路に乗る
if code != 0:
    sys.exit(code)
time.sleep(3)
```

- **本家の再起動挙動（test_play 境界でのメモリ衛生的な作り直し）は保存する**
- `max_steps` 未設定の無限 online 学習も従来どおり（完走判定が発火しない）

### 2-2. `run_ppo_p3_stage1_inner.sh` — watchdog を実際に止められるようにする

- **内側サブシェル `( ... )` を外す**（`$!` = ループ本体になり cleanup の `kill` が効く）
- **`code == 143 || code == 130`（SIGTERM / SIGINT）は意図的な停止として再起動しない**
- **client watchdog にも同じ2点を適用**。実 run では未発火だが構造は同一で、
  「cleanup が client を落とした直後に server も落ちる」競合に助けられていただけ

## 3. run-validation（GPU 実機・2本）

`max_steps=20` / `test_every=10` の smoke で、**1本の run で両方の経路**を通した:

- step 10 = test_play 境界 → **子を作り直す**（本家挙動の保存を確認）
- step 20 = `max_steps` 到達 → **作り直さない**

| 検査 | smoke 1（`train_ppo.py` のみ修正） | **smoke 2（両方修正）** |
|---|---|---|
| run dir | `smoke_respawn_20260802_182624` | `smoke_respawn2_20260802_185038` |
| Traceback / ConnectionRefusedError | **2**（鎖2 が残っていた） | **0** |
| 子プロセス起動回数 | 2 | **1**（step 10 の境界のみ） |
| 完走後の再 spawn（鎖1） | **無し**（修正の効果を確認） | 無し |
| watchdog の蘇生（鎖2） | `code=143, restart 1/3` | **ログ自体が無い = 蘇生なし** |
| launcher 終了 | exit=0 | **exit=0** |
| 残党 / GPU | 0 / 0 | **0 / 0** |

smoke 1 は「鎖1 は直ったが鎖2 が残る」ことを示す**中間証拠として意味がある**ので記録に残す。

## 4. 留保

- 本 run-validation は `max_steps=20` の短 run。**16000 step の本走で再確認されるのは
  次の実験 run**（L1 O3 の次に走るもの）。そこで `trainer.log` が例外なしで終わることを
  発進報告に含める
- `TRAINING_COMPLETE_EXIT_CODE = 21` は他のどの終了経路とも衝突しない値を選んだ
  （Python の未捕捉例外 = 1、シグナル終了 = 128+n）
- **凍結中の run には影響しない**: O1 / O3 の checkpoint・eval 出力・判定値は本修正の前に
  確定しており、修正はそれらを変更しない
