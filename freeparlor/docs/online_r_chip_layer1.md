# Online TD チップ報酬 — 層1: arena hora への chip_delta 埋め込み

**日付:** 2026-06-25  
**スコープ:** 生成のみ（学習・モデルは未変更）

## 目的

online 自己対戦ログ（`OneVsThree.py_vs_py` → `*.json.gz`）の各 **hora イベント** に、
その和了時点の即時チップ枚数（席ごとネット）を埋め込む。
後段（層2）の TD 学習で `r_chip` のソースになる。

```
層1 (本タスク)  arena ログ生成時に meta.chip_delta を付与
层2            dataloader / TD transition へ配線
层3            train.py / model.py 側の報酬合成
```

## 設計判断

| 項目 | 決定 |
|---|---|
| 載せ先 | **hora イベント** の `EventExt.meta`（打牌 move ではなく和了確定イベント） |
| フィールド | `chip_delta: [i8; 4]` — 4 席それぞれのネット枚数（和了者 +、支払者 −） |
| 視点 | 絶対席（trainee 視点ではない。dataloader 側で `player_id` を選ぶ） |
| 計算 | 既存 `agari_detail` + `preprocess_chips.py` と同一定義 |
| ryukyoku | `chip_delta` なし（`add_log_no_meta` のまま） |

打牌 meta（`q_values`, `shanten` 等）と同様、`EventExt` は `event` を flatten し **`meta` はネスト** してシリアライズされる。

### JSON 出力例

```json
{
  "type": "hora",
  "actor": 1,
  "target": 1,
  "deltas": [-1500, 4500, -1500, -1500],
  "ura_markers": [],
  "meta": {
    "chip_delta": [0, 0, 0, 0]
  }
}
```

## チップ計算（preprocess と同一）

`freeparlor/scripts/preprocess_chips.py` の定義を Rust 側 `AgariDetail` に移植:

```
chip_base = num_aka + num_ura + int(ippatsu) + (5 if yakuman >= 1 else 0)
```

| 和了形 | 配分 |
|---|---|
| ツモ | 和了者 +base×3、他 3 家各 −base |
| ロン | 和了者 +base、放銃者 −base |

## 変更ファイル

| ファイル | 内容 |
|---|---|
| `libriichi/src/mjai/event.rs` | `Metadata.chip_delta: Option<[i8; 4]>` 追加 |
| `libriichi/src/state/agari_detail.rs` | `chip_base()`, `chip_deltas(actor, target)` + 単体テスト |
| `libriichi/src/arena/board.rs` | `add_log_hora()` — hora ログ時に `agari_detail` → `chip_delta` を meta へ |

### Metadata 定義

```rust
// libriichi/src/mjai/event.rs
pub struct Metadata {
    // ... 既存フィールド ...
    /// Per-seat net chip count for this hora (winner +, losers -).
    pub chip_delta: Option<[i8; 4]>,
}
```

### hora への載せ方

```rust
// libriichi/src/arena/board.rs
fn add_log_hora(/* actor, target, deltas, ura_markers, is_ron, ura_indicators */) {
    let hora = Event::Hora { /* ... */ };
    let meta = self
        .hora_chip_delta(actor, target, is_ron, ura_indicators)
        .map(|chip_delta| Metadata {
            chip_delta: Some(chip_delta),
            ..Default::default()
        });
    self.add_log(EventExt { event: hora, meta });
}
```

`handle_hora` 内のロン（多ロン含む）・ツモ両方で `add_log_no_meta(hora)` を `add_log_hora(...)` に置換。
`agari_detail` 失敗時は `meta = None`（chip_delta 省略）。

## 触っていないもの

- `mortal/train.py`, `mortal/model.py`, 報酬合成
- `freeparlor/scripts/preprocess_chips.py`（参照のみ）

## 検証

**スクリプト:** `freeparlor/scripts/verify_arena_chip_delta.py`

```bash
PYTHONPATH=mortal python freeparlor/scripts/verify_arena_chip_delta.py \
  --mode mortal \
  --state-file /home/gamba/mahjong/runs/mortal.pth \
  --seed-start 20000 --seed-count 150 \
  --boltzmann-epsilon 0.5 --boltzmann-temp 1.0 \
  --device cuda:0 \
  --min-nonzero-hora 50
```

| オプション | 用途 |
|---|---|
| `--mode mortal` | 同一モデル self-play（`boltzmann_epsilon/temp` で和了率↑） |
| `--mode mortal_vs_agari` | モデル vs 常時 agari 受理 bot |
| `--agari-guard` | `enable_rule_based_agari_guard=True`（デフォルト False＝和了受理） |
| `--verify-only DIR` | 生成済みログの再検証のみ |

### Gate A: Mortal モデル — 非ゼロ hora 大量 cross-check（2026-06-25）

**生成条件:** `mortal.pth` self-play、`boltzmann_epsilon=0.5`, `boltzmann_temp=1.0`、
`enable_rule_based_agari_guard=False`、seed 20000–20149（150 seed × 4 split = **600 ログ**）

| 項目 | 結果 |
|---|---|
| hora 総数 | **4,488** |
| 非ゼロ `meta.chip_delta` hora | **2,530** |
| meta 欠落 | **0** |
| per-event 不一致 | **0** |
| per-kyoku 不一致（arena セマンティクス） | **0** |
| per-kyoku 不一致（`process_file()` そのまま） | 23（**多ロン局のみ**、下記） |

**検証方法（per-event）:** 各 hora 行について、ログ先頭から当該 hora 直前まで再生するが、
**hora イベントは `PlayerState.update` しない**（arena は hora を broadcast しないため）。
その状態で `agari_detail` → `hora_chip_deltas` を計算し `meta.chip_delta` と突合。

**多ロンと `process_file()` の差:** `preprocess_chips.process_file()` は hora を順次 `update` するため、
同一局 2 件目以降の hora で `agari_detail` が失敗し局集計が欠落する（Phase4 既知 edge case）。
arena の `meta.chip_delta` は各 hora 独立に正しく埋まる。22 ファイル・23 局が該当。

**非ゼロ hora の内訳（2,530 件）:**

| 要素 | 件数 | 例 |
|---|---|---|
| 赤 (aka) | 1,898 | `20000_0_a.json.gz` tsumo aka=2 → `[-2,6,-2,-2]` base=2 |
| 裏 (ura) | 784 | `20000_0_b.json.gz` tsumo ura=1 → `[3,-1,-1,-1]` |
| 一発 (ippatsu) | 564 | `20000_0_a.json.gz` ron → `[0,0,1,-1]` |
| 役満 (yakuman) | 1 | `20149_0_c.json.gz` ron → `[-5,0,5,0]` base=5 |

chip_base 分布: 1→1595, 2→649, 3→210, 4→64, 5→11, 6→1

**結果: PASS**

### Gate B: 初回スモーク（tsumogiri / agari bot、参考）

| 項目 | 結果 |
|---|---|
| 生成ログ | 1200 ファイル（agari 受理 bot、和了稀少） |
| hora 件数 | 4（すべて chip_base=0） |
| 不一致 | 0 |

### Gate C: ryukyoku

| 項目 | 結果 |
|---|---|
| ryukyoku 件数 | 14,400（Gate A ログ群） |
| `chip_delta` 付き | **0** |

### Rust 単体テスト

`libriichi/src/state/agari_detail.rs` に追加:

- `chip_deltas_ron` — base=4 (aka+ura+ippatsu), actor=2, target=1 → `[0, -4, 4, 0]`
- `chip_deltas_tsumo_yakuman` — base=5, actor=0 → `[15, -5, -5, -5]`

## ビルド

```bash
PYO3_PYTHON=/home/gamba/miniconda3/envs/mortal/bin/python \
  cargo build -p libriichi --lib --release
cp target/release/deps/libriichi.so mortal/libriichi.so
```

- ビルド成功（2026-06-25 確認）
- `mortal/libriichi.so` 更新済み、`from libriichi.arena import OneVsThree` import OK

## 層2 への引き継ぎ

→ **完了:** [`online_r_chip_layer2.md`](online_r_chip_layer2.md)

- dataloader / TD transition 側は **`ev["meta"]["chip_delta"]`**（hora 行）を読む
- hora は運搬役。`r_chip` は trainee 最終 move に局内合算で帰属（層2 実装済み）
- 層2 検証スクリプト: `freeparlor/scripts/verify_td_transitions.py`（tenhou ログへ chip_delta を注入する `inject_chip_delta_metadata` あり）

## 関連

| ドキュメント / スクリプト | 内容 |
|---|---|
| `freeparlor/docs/phase4_chip.md` | チップ規定・offline β 導入 |
| `freeparlor/docs/online_replay_buffer.md` | online ログ形式・drain フロー |
| `freeparlor/scripts/preprocess_chips.py` | chip_base / hora_chip_deltas 参照実装 |
| `freeparlor/scripts/verify_arena_chip_delta.py` | 層1 arena chip_delta cross-check |
| `freeparlor/docs/online_r_chip_layer2.md` | 層2 TD transition（dataloader 配線） |
| `freeparlor/scripts/verify_td_transitions.py` | 層2 TD transition 検証 |
