# PPO P3 — pause/resume (run #7)

**更新:** 2026-07-05  
**対象:** run #7 `stage1_20260705_072900`（実 checkpoint は `stage1_20260705_053301` — config 生成時 RUN_DIR 不一致、学習本体は 053301 側）

---

## 1. train_ppo.py resume 経路確認

| 項目 | 実装 | 根拠 |
|---|---|---|
| **(a) optimizer state 復元** | **済** | `state_file` 存在時 `optimizer.load_state_dict(state['optimizer'])`（online かつ保存 config も online のとき） |
| **(b) global step 復元** | **済** | `steps = state['steps']` → 再開後最初の `train_on_trajectories` 完了で **10001** から継続（10000 に戻らない） |
| **(c) 相手プール ckpt dir** | **run dir 相対で継続** | `player.py`: `run_dir/checkpoints`（`train_play/*/log_dir` の親の親）。resume 時は **新 run dir に step_*.pth をコピー**すれば同一プール構成を再現 |

### コード参照

```63:71:mortal/train_ppo.py
    if os.path.isfile(state_file):
        state = torch.load(state_file, weights_only=True, map_location=device)
        mortal.load_state_dict(state['mortal'])
        actor_critic.load_state_dict(state['actor_critic'])
        if not online or state['config']['control']['online']:
            optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])
        steps = state['steps']
        logging.info(f'loaded checkpoint: steps={steps:,}')
```

```105:120:mortal/train_ppo.py
    def save_checkpoint():
        state = {
            'mortal': mortal.state_dict(),
            'actor_critic': actor_critic.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'steps': steps,
            ...
        }
        torch.save(state, state_file)
        numbered = ckpt_dir / f'step_{steps:06d}.pth'
```

```239:248:mortal/player.py
        run_dir = path.dirname(path.dirname(self.log_dir))
        ckpt_dir = path.join(run_dir, 'checkpoints')
        ...
        pool = OpponentPool(ckpt_dir, past_k=..., latest_prob=..., fallback_checkpoint=fallback)
```

### step_010000.pth 実測（2026-07-05）

```
steps=10000
keys: actor_critic, config, mortal, optimizer, scaler, steps, timestamp
optimizer state entries: 411
sample exp_avg norm: 38.05
```

---

## 2. コード変更（save/load のみ — 学習数式無変更）

**(a)(b) は既実装のため train_ppo.py パッチ不要。**

追加 diff:

| ファイル | 変更 |
|---|---|
| `freeparlor/scripts/verify_ppo_p1.py` | 検定 **(16)** 追加: save → load → 1 opt step → `exp_avg` が連続参照と一致 |
| `freeparlor/scripts/run_ppo_p3_resume.sh` | 新規: 新 run dir 作成、`step_010000` から `mortal.pth`、pool ckpt コピー、preflight |
| `freeparlor/scripts/run_ppo_p3_stage1_inner.sh` | preflight 表示を checks 1–16 に更新 |

### 検定 (16) 概要

1. **実 checkpoint** (`step_010000.pth`): load 後 `optimizer.state_dict()` の全 411 パラメータ `exp_avg` が保存値と一致
2. 1 step 後: 全パラメータの moment が変化（リセットされていないこと）
3. **合成 subprocess**: 小 Linear で save → load → 1 step、moment 変化を確認

---

## 3. pause 記録（run #7）

| 項目 | 値 |
|---|---|
| 計画停止 step | **10000** |
| 停止時刻 | **2026-07-05 18:49:19 JST**（`step_010000.pth` / `mortal.pth` 保存ログ） |
| 再開時刻 | **2026-07-06 02:40:07 JST**（step 10001、run7b: `stage1_20260706_020120_resume`） |
| checkpoint 源 | `/home/gamba/mahjong/runs/ppo/stage1_20260705_053301/checkpoints/step_010000.pth` |

### 判定窓 8000–16000 の集計方針

**run7a (8000–10000) + run7b (10000–16000)** を global step で連結。

- **step 10000 に運用上の継ぎ目あり**
- on-policy 連続性のみ一時リセット（新 run 起動時に client/trainer プロセス再生成）
- **optimizer・相手プール・config は継続**（`mortal.pth` + `checkpoints/` コピー）

---

## 4. 再開手順

```bash
conda activate mortal
# GPU 単独ルール: 既存 server/trainer/client を停止してから
bash /home/gamba/mahjong/Mortal/freeparlor/scripts/run_ppo_p3_resume.sh
# preflight 通過後、確認してから:
LAUNCH=1 RUN_DIR=/home/gamba/mahjong/runs/ppo/stage1_<日時>_resume \
  bash /home/gamba/mahjong/Mortal/freeparlor/scripts/run_ppo_p3_resume.sh
```

環境変数:

| 変数 | デフォルト | 説明 |
|---|---|---|
| `SOURCE_RUN` | `stage1_20260705_053301` | checkpoint 源 run dir |
| `RESUME_STEP` | `10000` | 再開 global step |
| `MAX_STEPS` | `16000` | 完走目標 |
| `LAUNCH` | `0` | `1` で preflight 後に inner.sh 起動 |

新 run dir 名: `stage1_<YYYYMMDD_HHMMSS>_resume`

---

## 5. 既知事項

- run #7 の **launch dir** (`072900`) と **state_file 実パス** (`053301`) が不一致。resume スクリプトは **053301** を SOURCE_RUN デフォルトにしている。
- run #7 本走は step 10000 計画停止前に step 11060 まで進行していた可能性あり。再開は **step_010000.pth** を正とする（計画停止 checkpoint）。
