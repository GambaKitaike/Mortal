# Online リプレイバッファ — データ生成・drain・ログ形式

## 3 プロセス構成

| プロセス | ファイル | 役割 |
|---|---|---|
| server | `mortal/server.py` | パラメータ配布・ログ受信・バッファ管理 |
| client (worker) | `mortal/client.py` | 対局生成 → ログ送信 |
| trainer | `mortal/train.py` | `drain()` でログ取得 → 学習 |

## フロー

```
client.py
  get_param ← server (mortal/dqn)
  TrainPlayer.train_play() → py_vs_py() → *.json.gz
  submit_replay → server buffer_dir

train.py (online)
  drain() → server が buffer_dir → drain_dir へ move
  GameplayLoader.load_gz_log_files(drain_dir/*)
  player_names=['trainee'] で trainee 席のみ学習
```

## リプレイバッファ生成

**入口:** `mortal/client.py`

1. `get_param` で最新 weight 取得
2. `TrainPlayer.train_play()`（`mortal/player.py`）
3. 生成ログを `submit_replay` で server へ送信

**対局:** `OneVsThree.py_vs_py(challenger=trainee, champion=baseline)`

- 設定: `[train_play.default]`（`games`, `log_dir`, `boltzmann_*` 等）
- プロファイル: 環境変数 `TRAIN_PLAY_PROFILE`

## server / drain()

**server:** `mortal/server.py`  
**trainer 側:** `mortal/common.py` の `drain()`

| RPC | 処理 |
|---|---|
| `submit_replay` | `buffer_dir/{submission_id}_{filename}` に raw bytes 保存 |
| `drain` | `buffer_dir/*` を `drain_dir/` へ `shutil.move` |
| `drain()` 返値 | `{'count', 'drain_dir'}` — trainer は `drain_dir` 内ファイルを列挙 |

**trainer 読込:** `mortal/train.py` `train_epoch()` 内

```python
dirname = drain()
file_list = [path.join(dirname, p) for p in os.listdir(dirname)]
# player_names = ['trainee']
```

設定: `[online.server]` の `buffer_dir`, `drain_dir`, `capacity`, `force_sequential`

## py_vs_py 出力形式

**実装:** `libriichi/src/arena/one_vs_three.rs`  
**シリアライズ:** `libriichi/src/arena/result.rs` `dump_json_log()`

| 項目 | 内容 |
|---|---|
| ファイル名 | `{seed}_{key}_{split}.json.gz`（split = a/b/c/d） |
| 圧縮 | gzip (NDJSON) |
| 1 seed あたり | 4 半荘（4 split） |
| `seed_count` | `games // 4`（`[train_play.default].games`） |

**ログ構造（1 行 1 イベント）:**

1. `start_game` — `names: [4]`, `seed: (nonce, key)`
2. 各局 mjai イベント（`start_kyoku`, `tsumo`, `dahai`, `hora`, `ryukyoku` 等）
3. `end_game`

**席名:** agent の `name()` → online では `trainee` / `baseline`（`MortalEngine(name=...)`）

**拡張:** 打牌イベントに `EventExt.meta`（`q_values`, `is_greedy`, `shanten` 等）が付く場合あり（`libriichi/src/mjai/event.rs`）

## 学習側パース

**Python:** `mortal/dataloader.py` `FileDatasetsIter.populate_buffer()`  
**Rust:** `libriichi/src/dataset/gameplay.rs` `GameplayLoader.load_gz_log_files()`

- gzip 解凍 → NDJSON パース → 各席 `Gameplay`（obs, actions, masks, kyoku_rewards 等）
- online 時 `player_names=['trainee']` で trainee 席のみ残す
- offline の `dataset/globs` と **同一 mjai 形式**

## 関連ファイル

| 役割 | パス |
|---|---|
| worker | `mortal/client.py`, `mortal/player.py` |
| server | `mortal/server.py` |
| drain / submit_param | `mortal/common.py` |
| trainer | `mortal/train.py` |
| dataloader | `mortal/dataloader.py` |
| 対局 | `libriichi/src/arena/one_vs_three.rs`, `game.rs`, `board.rs` |
| mjai 定義 | `libriichi/src/mjai/event.rs` |
| 設定例 | `mortal/config.example.toml` `[online]`, `[train_play.default]` |
