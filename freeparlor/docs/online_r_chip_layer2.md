# Online TD チップ報酬 — 層2: dataloader TD トランジション

**日付:** 2026-06-25  
**スコープ:** dataloader のみ（`train.py` / `model.py` / Rust 非改修）

## 目的

層1で hora イベントに載せた `meta.chip_delta` を読み取り、dataloader の各 entry 末尾に
TD トランジション `(s, a, r_chip, s', done)` 相当のフィールドを追加する。

```
層1  arena ログ生成時に meta.chip_delta を付与          ✅
層2  dataloader / TD transition へ配線                  ✅ 本タスク
層3  train.py / model.py 側の n-step 報酬合成・学習     未着手
```

## 前提

- 層1 hora に `chip_delta: [i8; 4]`（絶対席）が載る（非ゼロ 2530 件で検証済み）
- online は `player_names=['trainee']`。`chip_delta[player_id]` を trainee 視点の報酬とする
- 終端は**局内**: `at_kyoku` 跨ぎ or `dones[i]` で `done_chip=1`

## 設計判断

| 項目 | 決定 |
|---|---|
| 実装ルート | **ルートA**: Python 隣接ペアリング、Rust 非改修 |
| entry 構造 | 既存 6 要素（oracle 時 7 要素）の**順序・型・意味は不変**。末尾に 4 要素追加 |
| 追加フィールド | `next_obs`, `next_mask`, `done_chip`, `r_chip` |
| hora の役割 | **運搬役のみ**。`r_chip` を hora 直前エントリに置かない |
| offline ログ | `chip_delta` 無し → `r_chip=0`、クラッシュ禁止 |
| n-step | 層2 では per-move 配列＋隣接 next＋done のみ。n=3 集約は層3 |

### r_chip 帰属ルール（重要）

hora は「和了確定の瞬間＝決定後」に記録される。hora 直前エントリに `r_chip` を置くと、
ロン和了では他家 move に乗り trainee フィルタで消える／帰属がズレる。

**正しい帰属:** その局で trainee が打った**最後の自分の move**（和了／放銃に向けた最終決定）の
`r_chip` に、その局の `chip_delta[trainee]` を集約する。

1. ログを走査し、局ごとに hora の `chip_delta[player_id]` を合算（多ロンは sum）
2. 同一 `at_kyoku` 内の trainee 最終 move インデックスを特定
3. そのインデックスの `r_chip` のみ非ゼロ（他は 0）

| ケース | 帰属先 |
|---|---|
| ツモ和了 | trainee の最終 move（通常 action=43） |
| ロン和了 | trainee の最終 move（通常 action=43） |
| 放銃 | trainee の最終 move（通常放銃打牌） |
| 多ロン | 全 hora の `chip_delta[trainee]` 合算 → trainee 最終 move 1 箇所 |

### 隣接ペアリング（s' / done）

各 move `i` について:

```
has_next = (i+1 < game_size) and (at_kyoku[i+1] == at_kyoku[i]) and not dones[i]

has_next  → next_obs/mask = obs[i+1]/masks[i+1], done_chip = 0
else      → next_obs = zeros, next_mask = all-False, done_chip = 1
```

## entry 構造（変更後）

**非 oracle（6 + 4 = 10 要素）:**

| idx | フィールド | 既存/新規 |
|---|---|---|
| 0 | `obs` | 既存 |
| 1 | `action` | 既存 |
| 2 | `mask` | 既存 |
| 3 | `steps_to_done` | 既存 |
| 4 | `kyoku_reward` | 既存 |
| 5 | `player_rank` | 既存 |
| 6 | `next_obs` | **新規** |
| 7 | `next_mask` | **新規** |
| 8 | `done_chip` | **新規** (0 or 1) |
| 9 | `r_chip` | **新規** (float32) |

oracle 時は idx=1 に `invisible_obs` が挿入され、TD 4 要素は引き続き末尾。

## 変更ファイル

| ファイル | 内容 |
|---|---|
| `mortal/dataloader.py` | TD ヘルパー 3 関数 + `populate_buffer` 配線 |
| `freeparlor/scripts/verify_td_transitions.py` | 層2 検証ゲート 1–5（新規） |

### コア実装

**1. hora chip_delta の局内合算（運搬）**

```python
# mortal/dataloader.py
def load_kyoku_hora_r_chip(file_path, player_id):
    """Sum hora meta.chip_delta[player_id] per kyoku (transport only)."""
    per_kyoku = defaultdict(float)
    with open_log(file_path) as f:
        kyoku_idx = -1
        for line in f:
            ev = json.loads(line)
            if ev.get('type') == 'start_kyoku':
                kyoku_idx += 1
            elif ev.get('type') == 'hora' and kyoku_idx >= 0:
                chip_delta = get_hora_chip_delta(ev)
                if chip_delta is not None:
                    per_kyoku[kyoku_idx] += chip_delta[player_id]
    return dict(per_kyoku)
```

**2. trainee 最終 move への帰属**

```python
def assign_r_chip_to_trainee_final_moves(game_size, at_kyoku, kyoku_hora_r_chip):
    r_chip = np.zeros(game_size, dtype=np.float32)
    last_idx_by_kyoku = {}
    for i in range(game_size):
        last_idx_by_kyoku[at_kyoku[i]] = i
    for kyoku, idx in last_idx_by_kyoku.items():
        r_chip[idx] = kyoku_hora_r_chip.get(kyoku, 0.0)
    return r_chip
```

**3. 隣接 (s, a, r, s', done) ペアリング**

```python
def build_td_transitions(obs, masks, at_kyoku, dones):
    """n-step (n=3) aggregation is layer 3."""
    ...
```

**4. populate_buffer 配線**

```python
kyoku_hora_r_chip = load_kyoku_hora_r_chip(file_path, player_id)
r_chip = assign_r_chip_to_trainee_final_moves(game_size, at_kyoku, kyoku_hora_r_chip)
next_obs, next_masks, done_chip = build_td_transitions(obs, masks, at_kyoku, dones)
entry.extend([next_obs[i], next_masks[i], done_chip[i], r_chip[i]])
```

## 触っていないもの

- `mortal/train.py`, `mortal/model.py`
- `libriichi/` Rust 側（GameplayLoader 等）

## 検証

**スクリプト:** `freeparlor/scripts/verify_td_transitions.py`

```bash
cd mortal
MORTAL_CFG=config.toml PYTHONPATH=. \
  python ../freeparlor/scripts/verify_td_transitions.py \
  --log-dir /tmp/verify_arena_chip_delta_aibvnlx4 \
  --version 4
```

| オプション | 用途 |
|---|---|
| `--log-dir DIR` | 層1 検証済み arena ログ群を直接検証 |
| `--state-file PATH` | OneVsThree でログ生成してから検証 |
| `--max-files N` | ファイル数上限（0=全件） |
| `--offline-sample PATH` | tenhou `.mjson` に chip_delta を注入して smoke |

### Gate 1: r_chip vs hora 合算（trainee 最終 move）

**条件:** 層1 arena ログ 600 ファイル（2400 games、17784 hora 局）

| 項目 | 結果 |
|---|---|
| 不一致 | **0** |
| 判定 | **PASS** |

各局について `sum(hora chip_delta[player_id])` と trainee 最終 move の `r_chip` が一致。

### Gate 2: ロン放銃 — 帰属先トレース

**例:** `20000_0_a.json.gz`, kyoku 5, `player_id=0`

| 項目 | 値 |
|---|---|
| hora | `actor=1, target=0`（他家ロン）, `chip_delta[0]=-1` |
| trainee 最終 move | idx=**93**, action=**2**（打牌） |
| `r_chip` | **-1.0**（最終 move のみ非ゼロ） |

他家 move には載らず、放銃打牌（trainee 最終決定）に正しく帰属。

### Gate 3: 多ロン — trainee 視点合算

**例:** `20049_0_c.json.gz`, kyoku 1, `player_id=0`

| hora | `chip_delta[0]` |
|---|---|
| `actor=1, target=0` | -1 |
| `actor=3, target=0` | -3 |

| 項目 | 値 |
|---|---|
| 合算 expected | **-4** |
| trainee 最終 move | idx=**39**, action=**15** |
| `r_chip` | **-4.0** |

2 件の hora が trainee 最終 move 1 箇所に正しく合算。

### Gate 4: done_chip / next ゼロ化

| 項目 | 結果 |
|---|---|
| `at_kyoku` 跨ぎで `done_chip=1`、next ゼロ化 | **0 errors** |
| 判定 | **PASS** |

### Gate 5: 既存 6 列回帰

| 項目 | 結果 |
|---|---|
| obs / action / mask / steps / reward / rank の不一致 | **0 errors** |
| 判定 | **PASS** |

### Offline smoke

tenhou `.mjson`（`chip_delta` 無し）でもクラッシュせず `gate1_mismatches=0`。

**総合結果: PASS**

## 層3 への引き継ぎ

- entry 末尾 4 要素（`next_obs`, `next_mask`, `done_chip`, `r_chip`）を train 側で unpack
- n-step (n=3, 局内 truncate): `R = Σ γ^k r_chip[i+k]`, bootstrap = `i+n`（局末超えたら truncate, done 扱い）
- `r_chip` は既に trainee 最終 move に集約済み。層3 では n-step 合成と損失への組み込みのみ

## 関連

| ドキュメント / スクリプト | 内容 |
|---|---|
| `freeparlor/docs/online_r_chip_layer1.md` | 層1: arena hora chip_delta 埋め込み |
| `freeparlor/docs/online_replay_buffer.md` | online ログ形式・drain フロー |
| `freeparlor/scripts/verify_arena_chip_delta.py` | 層1 cross-check |
| `freeparlor/scripts/verify_td_transitions.py` | 層2 TD transition 検証 |
| `mortal/dataloader.py` | TD ヘルパー + populate_buffer |
