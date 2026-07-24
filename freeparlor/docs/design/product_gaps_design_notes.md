# 商品化設計ギャップ G1–G3 設計メモ（0b セッション向け検討材料）

**ステータス: DRAFT（2026-07-24、設計監督側起草・裁定非関与・実装未承認）。**
本書は `policy_session_0b_frame.md` §5.3 で列挙した設計ギャップ G1–G3 の技術的
検討材料であり、判定文書ではない。DRCA の凍結解釈条件（`drca_probe_design.md`
§5a/§4）に関与せず、進行中の DRCA 本測定（第2枠 a_init）にも触れない。
**本書のいかなる案も実装は 0b セッションでの Gamba 裁定後**（例外は §4 の
「DRCA 待ち時間に前倒し可能な準備」として明示した項のみ、それも着手前に承認要）。

前提となる商品性要件（2026-07-24 ヒアリング、`policy_session_0b_frame.md` §5.1）:
- 商品形態 = **サジェストツール（メイン）＋ AI 対戦機能**（対戦→牌譜→解析/
  サジェストのループが商品コア）
- 出力形式 = **上位N手 + 評価値**
- 強さ基準 = **人間（Gamba/専門家）との実戦**
- 経済プリセット軸 = **鳴き祝儀/面前祝儀・ウマオカ表・チップ**

---

## 0. 現行アーキテクチャの事実確認（2026-07-24 実地検証済み）

設計の土台として現物を確認した事実。以後の各節はこれに基づく。

| # | 事実 | 根拠 |
|---|---|---|
| F1 | 方策ヘッドは 46 行動の分布 π、価値ヘッドは**単一スカラー** V(s)（3ストリーム分解なし） | `mortal/model.py` PolicyHead / ValueHead（`Linear(→1)`、"Scalar state value"） |
| F2 | 正典3ストリーム（素点/GRP/チップ）は **GAE 前に単一通貨へ合算**される: `α·素点Δ + γ_pt·GRPΔ + β·chipΔ·chip_value` | `mortal/ppo.py:162`（`combine_rewards` 相当の純関数）、config キー `alpha/gamma_pt/beta/chip_value`（`ppo_stage3.toml` [ppo]） |
| F3 | チップ発生規則は Rust 側: `chip_base = 赤枚数 + 裏枚数 + 一発 + 役満×5`、ロンは放銃者負担・ツモは3人折半でなく各家 base 払い | `libriichi/src/state/agari_detail.rs:27-51`（`chip_base`/`chip_deltas`） |
| F4 | **祝儀の面前ゲートは暗黙的**: 裏・一発は立直（=面前）でしか発生し得ないため実質面前専用、**赤のみ鳴き手でも有効**。明示的な `is_menzen` 条件分岐はコード上に存在しない | 同上（chip_base に面前条件なし。裏/一発の立直依存は麻雀ルール由来） |
| F5 | ウマ（着順点）は `RewardCalculator(pts=...)` でパラメタ化済み、デフォルト `[3,1,-1,-3]`。Python 側・config 配線可能な位置にある | `mortal/reward_calculator.py:5,12` |
| F6 | engine インターフェースは `react_batch(obs, masks, invisible_obs, step_meta)` のバッチ形。既存のラッパー前例あり | `freeparlor/scripts/drca_common.py:170-202`（`RecordingPassthroughEngine`） |
| F7 | fork-by-replay（seed 決定論 + 台本再生 + 強制行動 + K rollouts）は検証済み資産。実測スループット ~109 rollouts/h/プロセス（1 rollout ≈ 33s）、3プロセス並列可 | DRCA 実装（`drca_run_probe.py` 系、監督3段検証合格）+ 第1枠実測 |

**注記（訂正）**: `policy_session_0b_frame.md` §5.3 G1 行の「critic V(s)(状態値・
3ストリーム)」は F1/F2 のとおり不正確（3ストリームは報酬合成段で消え、critic は
合算通貨の単一スカラー）。同行は本書 commit と同時に訂正。この事実は G1 に直接
効く: **サジェスト評価値を素点/着順/チップに分解して見せる能力は現行ネットに無い**。

---

## 1. G1 — per-action 評価器（上位N手 + 評価値）

### 1.1 問題

要件「上位N手 + 評価値」に対し、現行は π（行動の確率）と V(s)（状態の価値）のみ。
π の確率順位は「方策がよく選ぶ順」であって「行動の価値順」ではない（エントロピー
正則化・探索由来の質量を含む）。行動ごとの価値 Q(s,a) は直接には存在しない。

### 1.2 選択肢

**O1: π 上位N + 確率 + V(s) 文脈表示（訓練変更ゼロ）**
- 上位N手を π の確率つきで提示、局面全体の見通しとして V(s)（千点換算の合算通貨）
  を添える。実装は推論スクリプトのみ、学習系無変更。
- 限界: 評価値が「確率」であり千点単位の期待差ではない。2位以下の手の価値差は読めない
  （π が尖っている局面では 2位手の確率 ≈0 でも価値差は僅少かもしれない）。
- 位置づけ: **MVP。**最速で G2 パイプラインに載せられる。

**O2: オンデマンド反実仮想評価（DRCA 資産の転用）**
- 指定した decision point で fork-by-replay を回し、候補手それぞれを強制 →
  K rollouts → 各手の Q̂ を千点単位で提示。DRCA の採取・再生・強制行動・集計の
  検証済み資産（F7）をほぼそのまま使う。
- コスト実測ベース: N=3 候補 × K=8 = 24 rollouts ≈ 13分/decision（1プロセス）、
  3プロセス並列で ~4.5分。**live には不可、事後の牌譜解析（「この局面を深掘り」）
  には成立**。K を落とせば SE と引き換えに短縮可。
- **重要な制約: fork には山（seed）の知識が必要**。自前アリーナ（G2）で打たれた
  対局は seed 既知なので fork 可能。**外部牌譜（天鳳等）は山が未知なので厳密 fork
  不可**（観測整合な山の再サンプリングは別問題で難度が高い）。
  → 「深掘り解析は自前対戦の牌譜でのみ提供」という商品制約になり、
  **G2 の対戦機能を持つこと自体の追加根拠**になる。
- 位置づけ: **深掘りモード。O1 と二段構え**。

**O3: per-action Q/advantage ヘッドの増設（アーキ変更・新規訓練）**
- ActorCritic に行動次元のヘッドを足し、実測 return/GAE から蒸留して学習。
  live で全行動の価値を千点単位で出せる唯一の案。
- コスト: アーキ変更 + 16k 級 run 1本以上 + 検定拡張。凍結規律上、実施は完全に
  0b 裁定後。また「後付け Q_chip ヘッド不活性」の前例
  （`reward_design_teacherfree.md` 確定事項）があり、疎な行動で信号不足になる
  リスクは既知パターン。
- 位置づけ: **商品化本決まり後の投資案件**。MVP には含めない。

### 1.3 推奨

**O1（live）+ O2（深掘り）の二段構えを MVP とし、O3 は 0b で商品化が確定した場合の
投資判断に送る。** O1/O2 とも学習コード・libriichi 無変更で構成でき、既存資産の
転用率が高い。ストリーム分解表示（素点/チップ内訳）は O2 なら rollout 終端の
3ストリームを分けて集計するだけで可能（DRCA の d_chip 層別と同じ手法）— O1 では
不可能（F1/F2）、という非対称も二段構えを支持する。

---

## 2. G2 — AI 対戦機能（対人プレイ/牌譜/解析パイプライン)

### 2.1 商品コアとしての位置づけ

ヒアリング補足（2026-07-24）で確定: 対戦は牌譜生成源かつ強さ測定（人間実戦）の
ハーネスを兼ねる一級機能。パイプラインは
**対局 → mjai 牌譜 → (a) HTML ビューア閲覧 / (b) G1 解析・サジェスト**。
既存資産: self-play アリーナ（`one_vs_three` 系）、mjai ログ出力、
`mjai_log_to_html.py`（07-15 実装・検証済み）、log-viewer テンプレ。

### 2.2 主設計問題: 人間クライアントの挿入点

engine インターフェース（F6）は encode 済みテンソル `(obs, masks, invisible_obs)`
を受ける。人間に見せる盤面はテンソルからではなく mjai イベント列から描画したい。
候補は2つ:

**H1: HumanEngine（engine 層に挿入）**
- `react_batch` を実装した人間入力 engine を4席の1つに配置。既存アリーナを
  無改造で使える（`RecordingPassthroughEngine` と同じ挿入面）。
- 盤面描画は「ゲーム開始からの mjai イベントを並行タップして描画」が必要
  （obs テンソルの逆変換は非現実的）。アリーナがイベント列を人間側へ随時供給できるか
  が要調査点（ログは対局完了後書き出しの可能性）。
- 合法手は masks があるので UI 側の行動検証は不要（mask 内の選択のみ許す）。

**H2: mjai プロトコル層で接続（本家 mjai クライアント形式）**
- 本家 Mortal 系の mjai bot インターフェースに人間 UI を接続する形。libriichi の
  イベント駆動面を使うため描画は自然だが、現行の PPO 推論経路（ppo_engine）と
  アリーナ運用（チップ経済・one_vs_three 固定 split）との整合を新規に作る必要。

**判定は実装調査後**だが、暫定推奨は **H1**（既存アリーナ・経済・検定資産の転用率が
最も高く、Rust 表面積ゼロで済む見込み）。イベントタップの可否が調査の第一項目。

### 2.3 運用上の制約

- **GPU 1系統規律**: DRCA 測定中の対人対局は、(a) DRCA 完走待ち、または
  (b) CPU 推論（1 decision ずつなら 192×40 で実用レイテンシの見込み・要実測）の
  二択。焦って GPU を触らない。
- 対局の経済設定は G3 のプリセットと一致させる（サジェストと対局で経済が違うと
  評価値の意味がずれる）。
- 牌譜は run 規約に準じ保全（seed 込み = O2 深掘りの前提資産）。
- **UI の実装規模は 0b で商品範囲が決まるまで最小**（まずは CLI/TUI、Web UI は後段）。

---

## 3. G3 — 経済プリセットの config 化（鳴き祝儀・ウマ表・チップ）

### 3.1 現行の経済表面（F3–F5）

| 軸 | 現行 | 変更容易性 |
|---|---|---|
| ウマ表 | `RewardCalculator(pts=[3,1,-1,-3])` — Python・実質パラメタ化済み | **低コスト**（config 配線のみ。ただし GRP の順位期待値モデルとの整合に注意） |
| チップ単価 | `chip_value=5.0`（千点/枚、[ppo] config） | **既に config** |
| 祝儀発生規則 | Rust `chip_base = 赤+裏+一発+役満×5`（F3）。面前ゲートは裏/一発の立直依存として暗黙（F4） | **要設計**（下記） |

重要な含意（`policy_session_0b_frame.md` §5.2-4 の精密化）: 現行祝儀の3項のうち
裏・一発は実質面前専用、赤のみ鳴き手で有効。つまり現行は「部分的面前祝儀」であり、
鳴き手の祝儀期待値は構造的に細い。**「鳴き祝儀」プリセット（例: 鳴き和了の赤にも
割増、または鳴き和了自体への祝儀）は、測定済み機会費用ギャップ（~5.5–5.9 チップ）を
直接埋めうる最も標的的なレバー** — 議題2 の第一候補軸という §5.2-4 の評価を維持。

### 3.2 実装方式の選択肢

**E1: Rust 側パラメタ化** — `chip_base` を経済 struct で駆動し、p_enrich と同じ
config→Board 配線前例で通す。
- 利点: 発生規則が一箇所。欠点: 経済バリアントごとに Rust を触る（表面積最小原則と
  緊張）。

**E2: 成分の一括露出 + Python 側で祝儀計算** — Metadata に和了成分
（num_aka/num_ura/ippatsu/yakuman/is_menzen/副露有無）を1回だけ露出し、
祝儀バリアントは Python（報酬計算側）で合成。
- 利点: **Rust 変更は成分露出の1回きり**で、以後の全プリセットが Python/config のみ
  = 「1 branch = 1 variable」の変数追加が軽い。チップは対局進行に影響しない
  精算専用値（合法手・状態遷移に非関与）なので、Rust 内で確定させる必然性がない。
- 欠点: chip 解決経路（`online chip resolution` 監視項目）の再配線と検定拡張が必要。

**暫定推奨は E2**（表面積最小原則・プリセット拡張の限界費用最小）。ただし
`chip_deltas` の既存呼び出し面の調査（本家挙動への影響ゼロ確認）が前提。

### 3.3 コストと規律

- **プリセット1つ = 訓練 run 1本**（16k ≈ 2日 GPU + drain ~370GB）。配備税の教訓
  （Stage2）どおり、**配備先経済と同一経済で訓練する**のが原則 — 経済間転移は仮定
  しない。既存 checkpoint からの fine-tune で短縮する案は Stage2b（議題4）と同じ
  問題設定であり、0b で併せて裁定。
- 軸の優先順位（暫定）: **鳴き祝儀 → ウマ表 → チップ単価**。根拠: 鳴き祝儀のみが
  機会費用ギャップに直接作用（他2軸は間接）。かつ「立直特化が最強か」の実証
  （議題1 の芯）に対し、鳴き祝儀プリセットは**経済側から均衡を動かす対照実験**として
  情報量が最大。
- 各 run の判定条件は発進前に design md で事前登録（本プロジェクトの標準規律）。

---

## 4. 依存関係と実施順序（0b への提案）

```
DRCA 残枠（a_init 進行中 → a_s3final → a_s3mid → b_s3final）→ DRCA 判定
  └─ その間に前倒し可能（GPU 不要・要 Gamba 承認）:
       - G2 挿入点調査（H1 イベントタップ可否の read-only 調査）
       - G3 E2 の呼び出し面調査（chip_deltas 利用箇所の read-only 調査)
0b セッション（判定照合 → 議題1–4 裁定）
  └─ 裁定に応じて: G2 MVP(CLI) → G1 O1/O2 → G3 E2 + 鳴き祝儀プリセット run
```

- G1 O2 と G2 は相互依存（O2 は自前アリーナ牌譜が前提、G2 は O1/O2 が解析出力)。
  **着手順は G2 → G1**（牌譜がなければ解析対象がない）。
- G3 は G1/G2 と独立に設計可能だが、run 発進は GPU 1系統規律により DRCA 完走後。
- 本書の各「推奨」は 0b の入力であって決定ではない。裁定は全て Gamba。

---

## 5. read-only 調査結果（2026-07-24 実施、Gamba 承認済み・コード変更なし）

§4 で挙げた前倒し調査2本の結果。**§2.2 / §3.2 の暫定推奨をここで更新する**
（本文は原案として残し、本節が優先）。

### 5.1 調査1: G2/H1 — mjai イベントタップの可否

**結論: イベントは既にエージェント境界まで配られており、Python への橋が
1箇所で捨てているだけ。opt-in の追加的 Rust 小変更1点で実現可能。**

| 事実 | 根拠 |
|---|---|
| Board は対局中 `log: Vec<EventExt>` をインクリメンタルに蓄積（json.gz はゲーム終了時に `take_log` でまとめ書き。対局中のファイル tail は不可能） | `libriichi/src/arena/board.rs:82,223` |
| game loop は**毎 decision で全イベント log** を `BatchAgent::set_scene(index, ctx.log, state, ...)` に渡している | `libriichi/src/arena/game.rs:115-120` |
| Python bridge（`MortalBatchAgent::set_scene`）は log 引数を `_` で**破棄**し、encode 済み obs のみ Python `react_batch` へ | `libriichi/src/agent/mortal.rs:224-228` |
| `EventExt`/`Metadata` は `Serialize` 実装済み（JSON 化は自明） | `libriichi/src/mjai/event.rs:132-141` |
| `enable_quick_eval` は Python engine 属性から getattr で読まれる既存配線。人間席は `False` にすれば打牌1択局面も含め全 decision が Python に届く | `libriichi/src/agent/mortal.rs:68`、`mortal/engine.py:19` |

実現方式: set_scene で log（または前回呼び出し以降の差分イベント）を JSON 化して
opt-in 属性付き engine にのみ渡す（属性不在＝従来挙動ビット不変、p_enrich の
getattr 前例と同型）。**H2（mjai プロトコル層の別経路新設）は不要と判断** —
アリーナ・経済・検定資産をそのまま使える H1 で確定を提案。
UI 側は受信イベント列から盤面描画 + masks で合法手提示、という素直な構成になる。

### 5.2 調査2: G3/E2 — chip_deltas 呼び出し面

**結論: E2 想定（「成分露出の Rust 変更1回が必要」）より良い。
祝儀バリアントは Rust 変更ゼロで実装可能（E2' に更新）。**

| 事実 | 根拠 |
|---|---|
| `AgariDetail` は `#[pyclass]` で**全フィールド `#[pyo3(get)]` 露出済み**（num_aka/num_ura/ippatsu/yakuman/is_tsumo/point/fu/han） | `libriichi/src/state/agari_detail.rs:3-20` |
| `PlayerState.agari_detail(is_ron, ura)` が Python から呼べ、`chip_from_log.py` の**フォールバック経路が現に log 再生 → Python 側チップ計算をしている** | `mortal/chip_from_log.py:39-60` |
| チップ規則の **Python 実装が既に存在**: `preprocess_chips.chip_base` / `hora_chip_deltas`（Rust 側はコメント "Matches preprocess_chips.chip_base" のとおり Python 版の鏡） | `freeparlor/scripts/preprocess_chips.py:24-41` |
| 訓練のチップストリームは **Python 側で log から解決**される（client.py → `load_kyoku_chip_deltas_from_log`）。チップは対局進行・合法手に非関与の精算専用値 | `mortal/client.py:142-173` |
| 現行 `chip_delta_at_hora` は log の `meta.chip_delta`（Rust 産・正典規則）を**優先**し、無いときのみ再計算 | `mortal/chip_from_log.py:41-45` |

E2'（更新後の推奨）: 祝儀バリアント = Python 側の `chip_base`/`hora_chip_deltas` を
経済パラメタ（面前/鳴き条件・係数）で駆動し、**バリアント経済では `meta.chip_delta`
優先を止めて常に再計算**する（Rust 産 meta は正典規則の値のまま log に残るため、
読み手がそれを拾うと経済がずれる）。和了者の面前/副露状態は当該局のイベント列から
Python で導出可能（チー/ポン/大明槓/加槓 = 副露、暗槓は面前維持 — 祝儀ルール上の
扱い自体がプリセットパラメタ）。

**要監査事項（バリアント branch の検定拡張対象)**: `meta.chip_delta` を直接読む
消費者の全数列挙（少なくとも `chip_from_log.py` 優先経路・検定(の chip 配置 e2e)・
eval 集計系。正典経済 branch では現状維持で無害、バリアント branch でのみ
再計算強制へ切り替え）。

### 5.3 §4 実施順への影響

- G2 MVP の Rust 表面積は「set_scene の opt-in イベント転送」1点に確定
  （事前見積り「表面積ゼロの見込み」は 1点に修正)。libriichi 改修を伴うため、
  実装時はビルド手順規律（CLAUDE.md）+ 検定拡張（opt-in 不在時のビット不変性）が必要。
- G3 は Rust 変更ゼロに確定 → **G3 の実装リスクは G2 より低い**。ただし run 発進
  （GPU）が律速である事実は不変。
- いずれも実装承認は 0b 裁定後（本調査は read-only の事前確定のみ）。
