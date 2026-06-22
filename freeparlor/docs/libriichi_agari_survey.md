# libriichi 和了情報調査（チップ報酬 β 向け）

調査日: 2026-06-22  
調査対象: `/home/gamba/mahjong/Mortal/libriichi/src/`  
目的: 牌譜リプレイで和了時の「赤ドラ枚数・裏ドラ枚数・一発・役満・役・翻・符」が libriichi からどこまで取得できるかを確認する。

---

## 1. 点数計算・和了解析モジュール

### 1.1 モジュール構成

```
libriichi/src/algo/
├── agari.rs   # 役・翻・符の判定（和了アルゴリズム本体）
├── point.rs   # 符・翻 → 点数変換
├── shanten.rs
└── sp/        # 単騎期待値計算
```

### 1.2 `Agari` — 和了結果の Rust 内部表現

`libriichi/src/algo/agari.rs` 67–74 行:

```rust
pub enum Agari {
    Normal { fu: u8, han: u8 },
    Yakuman(u8),
}
```

| 項目 | 計算 | 保持形式 |
|---|---|---|
| 符 (fu) | ✅ | `Agari::Normal { fu, han }` |
| 翻 (han) | ✅ | 同上（役 + 状況 + ドラ合算後） |
| 役満 | ✅ | `Agari::Yakuman(u8)`（倍数のみ） |
| 個別役名 | △ 計算するが保持しない | なし |
| 赤ドラ枚数 | △ 計算に使用 | 独立フィールドなし |
| 裏ドラ枚数 | △ 計算に使用 | 独立フィールドなし |
| 一発 | △ 計算に使用 | `Agari` には残らない |

`search_yakus()`（212–213 行）は `DivWorker::search_yakus`（452 行〜）内で役ごとに `han` / `yakuman` を加算するだけで、**役名リストは返さない**。

### 1.3 `agari()` — ドラ・状況役を翻に合算

`libriichi/src/algo/agari.rs` 216–253 行:

- `additional_hans`: 立直・両立直・**一発**・門前清自摸和・槍槓・嶺上/海底/河底など（個別フラグではなく翻数として合算）
- `doras`: 表ドラ + 赤ドラ + 裏ドラを **1 つの u8 合計** として `han` に加算

### 1.4 `Point` — 点数のみ

`libriichi/src/algo/point.rs` 1–6 行:

```rust
pub struct Point {
    pub ron: i32,
    pub tsumo_ko: i32,
    pub tsumo_oya: i32,
}
```

符・翻・役・ドラ内訳は含まない。

### 1.5 `PlayerState::agari_points` — リプレイ時の実用エントリポイント

`libriichi/src/state/agent_helper.rs` 377–461 行:

- 一発: `self.at_ippatsu` を `additional_hans` に +1（400, 411 行）
- 赤ドラ: `winning_tile.is_aka()` で `final_doras_owned += 1`（427–429 行）
- 裏ドラ: `ura_indicators` から手牌/暗槓内の該当牌枚数を数え加算（432–442 行）
- **返却値は `Point`（点数）のみ**。`Agari { fu, han }` や内訳は 461 行で破棄される

---

## 2. 牌譜リプレイ / Python 公開 API

### 2.1 Python に公開されているクラス

| モジュール | クラス | 和了詳細 |
|---|---|---|
| `libriichi.state` | `PlayerState`, `ActionCandidate` | 間接的（後述） |
| `libriichi.dataset` | `GameplayLoader`, `Gameplay`, `Grp` | なし（学習用特徴量） |
| `libriichi.arena` | `OneVsThree`, `TwoVsTwo` | なし |
| `libriichi.stat` | `Stat` | 集計のみ |
| `libriichi.mjai` | `Bot` | なし |

`Agari`, `AgariCalculator`, `Point`, `agari_points` は **Python 未公開**（`state/mod.rs` 26–27 行: `PlayerState` と `ActionCandidate` のみ登録）。

### 2.2 `PlayerState` から取れるもの

公開 getter（`state/getter.rs`）:

- `tehai`, `akas_in_hand`, `chis/pons/minkans/ankans`
- `shanten`, `waits`, `last_cans`, `self_riichi_*`, `at_furiten` 等

**非公開**（Rust 内部のみ、`player_state.rs`）:

- `at_ippatsu`（113 行）
- `doras_owned`（129 行）
- `dora_indicators`
- `agari_points()`

### 2.3 `GameplayLoader`

牌譜を `PlayerState.update()` で再生し特徴量を生成するが、`Hora` 時の fu/han/ドラ等は記録しない（`dataset/gameplay.rs` 373–384 行: 和了アクションラベル `43` の判定のみ）。

### 2.4 mjai 牌譜ログから直接取れるもの

`libriichi/src/mjai/event.rs` 105–113 行:

```rust
Hora {
    actor: u8,
    target: u8,
    deltas: Option<[i32; 4]>,
    ura_markers: Option<Vec<Tile>>,
}
```

| 項目 | ログ JSON から |
|---|---|
| 和了者・放銃者 | ✅ |
| 点数移動 (`deltas`) | ✅ |
| 裏ドラ表示牌 (`ura_markers`) | ✅（立直和了時） |
| 符・翻・個別役 | ❌ |
| 赤ドラ枚数 | ❌ |
| 一発フラグ | ❌ |

### 2.5 リプレイ検証の参考実装（Rust CLI のみ）

`libriichi/src/bin/validate_logs.rs` 189–231 行:

1. `Hora` イベント到達時、**update 前**に `agari_points(is_ron, ura)` を呼ぶ
2. その後 `update_with_keep_cans(ev, true)` で状態更新

Python には `update_with_keep_cans` も `agari_points` も公開されていない。

### 2.6 `Stat` — 和了「詳細」ではなく集計

`libriichi/src/stat.rs` 364–366 行: 役満は **得点が役満ライン以上** で推定。個別和了の fu/han/ドラ内訳はなし。

---

## 3. 赤ドラ・一発・裏ドラの内部表現

### 3.1 赤ドラ（`tile.rs`）

- 牌 ID: `5mr`, `5pr`, `5sr`（17–18 行）
- `is_aka()` / `deaka()` / `akaize()`（68–91 行）
- `doras_owned` 更新時に `is_aka()` なら +1（`update.rs` 955–959, 758–773 行）
- 表ドラ・赤ドラは **`doras_owned` に合算** され、赤のみの枚数フィールドはない

### 3.2 一発（ippatsu）

- `PlayerState.at_ippatsu: bool`（`player_state.rs` 113 行）
- 立直成立で true（`update.rs` 677–685 行）
- 副露・槍槓等で false（58–59, 345, 446, 515, 560, 606, 638 行）
- 和了計算時は `additional_hans` に +1 として合算（`agent_helper.rs` 400, 411 行）

### 3.3 裏ドラ（uradora）

- 牌譜: `Event::Hora.ura_markers`（立直和了時のみ、`arena/board.rs` 423–427, 457–461 行）
- 計算: `agari_points` が `ura_indicators` から手牌/暗槓内の該当牌枚数を数え `final_doras_owned` に加算（432–442 行）
- **裏ドラ枚数の独立フィールドはない**（表 + 赤 + 裏が `doras` 1 本に合算）

---

## 4. 判定まとめ

### (a) Rust 側で計算済みか

**結論**: fu・合計 han・役満倍数は Rust 内で算出されるが、**赤/裏/表ドラ枚数・一発・個別役は独立して保持されない**。`agari_points` は最終的に `Point`（点数）だけ返す。

### (b) Python (pyo3) 公開状況

| 項目 | 公開 | 取得方法 |
|---|---|---|
| 符・翻・役満 | ❌ | — |
| 個別役 | ❌ | — |
| 一発 | ❌ | `at_ippatsu` 非公開 |
| 赤ドラ枚数 | △ | `akas_in_hand()` のみ（手牌の赤 5 可否）。副露赤・合計枚数は不可 |
| 裏ドラ | △ | mjai JSON の `ura_markers` を Python でパース |
| 和了点数 | △ | mjai JSON の `deltas`、または `Stat.from_log()` 集計 |
| 牌譜リプレイ | ✅ | `PlayerState.update(mjai_json)` |

### (c) 実装ルートの現実性

#### (c-1) Rust 小改修で Python 公開 — **推奨**

根拠:

1. 計算ロジックは完成済み（`AgariCalculator`, `PlayerState::agari_points`, `validate_logs.rs` のリプレイパターン）
2. 必要なのは **既存結果の返却形式拡張** が中心
   - 例: `calc_agari(is_ron, ura_markers) -> { fu, han, yakuman, ippatsu, num_aka, num_omote_dora, num_ura_dora, point }`
   - `agari_points` 461 行の `.point()` 前で `Agari` と内訳を返す
3. 改修規模は小さい（`getter.rs` / `agent_helper.rs` / pyo3 構造体追加）
4. `update_with_keep_cans` の Python 公開も検討余地あり（`update.rs` 28–31 行）

#### (c-2) Python 側で独自リプレイ — **非推奨**

根拠:

1. `agari_points` / `AgariCalculator` が Python から呼べない
2. `GameplayLoader` は和了詳細を出力しない
3. Python だけで取れるのは `ura_markers`・`deltas`・`akas_in_hand` 程度
4. 符・翻・役・ドラ内訳の再現には Rust ロジックの再実装か別 FFI が必要
5. `Stat.yakuman` は得点閾値ベース（364 行）で、和了単位の内訳には不向き

---

## 5. チップ報酬 β 向けの実装示唆

牌譜リプレイで和了ごとに報酬を付ける場合の現実的フロー:

```
1. PlayerState(player_id) を作成
2. イベントを順に update() — Hora の直前で止める
3. [要 Rust 改修] calc_agari(is_ron, hora["ura_markers"]) を呼ぶ
4. 返却値から β 用特徴量（赤枚数・裏枚数・一発・翻・符・役満）を取得
5. Hora イベントを update() して続行
```

現状の Python だけで可能なのは **裏ドラ表示牌の取得**（ログ JSON）と **合計得点**（`deltas`）まで。

符・翻・赤枚数・一発・個別役は **Rust 側の小改修が前提**。

個別「役」リストが必要な場合は、`search_yakus` 内の加算箇所（490 行〜）に役 ID 返却を追加する **中規模改修** が別途必要（現状は設計上保持しない）。

---

## 6. 主要ソース参照一覧

| ファイル | 行 | 内容 |
|---|---|---|
| `algo/agari.rs` | 67–74 | `Agari` enum 定義 |
| `algo/agari.rs` | 216–253 | `agari()` — ドラ・状況役合算 |
| `algo/point.rs` | 1–6 | `Point` 構造体 |
| `state/agent_helper.rs` | 377–461 | `agari_points()` |
| `state/player_state.rs` | 113, 129 | `at_ippatsu`, `doras_owned` |
| `state/getter.rs` | 42–48 | Python 公開 getter |
| `state/mod.rs` | 26–27 | Python 登録クラス |
| `state/update.rs` | 28–31, 677–685 | `update_with_keep_cans`, 一発設定 |
| `tile.rs` | 68–91 | 赤ドラ判定 |
| `mjai/event.rs` | 105–113 | `Hora` イベント |
| `bin/validate_logs.rs` | 189–231 | リプレイ検証パターン |
| `stat.rs` | 364–366 | 役満集計（得点閾値） |
| `dataset/gameplay.rs` | 373–384 | GameplayLoader（和了詳細なし） |
| `arena/board.rs` | 423–427, 457–461 | 裏ドラ表示牌のログ記録 |
