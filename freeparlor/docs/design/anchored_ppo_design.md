# アンカー付き PPO（anchor 系列）設計書 — 基礎劣化対策の単一変数アブレーション

**日付:** 2026-07-25
**ステータス:** **凍結済み（2026-07-25、Gamba 裁定）— 本改訂 commit が事前登録。
以後の変更禁止**（唯一の許容は §6a 再走規定の機械的適用と、その結果数値の §9 への追記）。
実装は Composer/Sonnet へ委任（タスクプロンプト: `../ops/anchor_impl_task_20260725.md`）、
発進は実装検証後・DRCA 第3枠完走後の別タスク
**起草:** 実装エージェント（Claude Code / Fable）
**根拠文書:** `../reports/fundamentals_degradation_diagnosis_20260725.md`（§4 原因仮説・§6 候補列挙）、
`../reports/drca_frame2_a_init_aggregate_20260725.md`（反鳴き均衡は基礎劣化アーティファクトに非ず）

---

## 0. 目的と切り分け対象

診断レポートの確定事実: 現行 PPO（Stage1–3 全系列）は init（`beta1_huber_192x40`、
人間譜由来）が持っていた基礎技能（放銃回避・和了）を訓練で劣化させる
（放銃 12.10%→~15.2%、和了 20.67%→~18.8%、全 16k checkpoint が 1v3 で init に勝てない）。
原因仮説の序列: **主犯 = アンカー不在**（損失に init への引き戻しが無い）、
増幅 = 自己対戦均衡の退化（相手が自分の過去のみ）+ スパース終局報酬。

本系列はこの仮説を**2つの単一変数 arm** で切り分ける:

| arm | 介入 | 検証する仮説 |
|---|---|---|
| **C**（pool 是正） | opponent pool に凍結 init を確率常駐。**損失は不変** | 均衡退化の単独犯性（§4 増幅1 だけで基礎劣化を説明できるか） |
| **K**（KL アンカー） | 損失に凍結 init への KL ペナルティを追加。**pool は不変** | アンカー不在の主犯性（引き戻しだけで基礎が守れるか） |

両 arm が独立に効けば合流形（A = C+K）を第3の run として検討（別途裁定）。

**成功の定義（直観）:** 「init の基礎を維持したまま、フリー雀荘経済（チップ）への
適応だけが乗る」方策。init クローン（アンカー過強）も基礎劣化（アンカー過弱）も失敗。

## 1. 共通条件（両 arm）

- init = `beta1_huber_192x40/mortal.pth`（全 stage と同一）、自然分布（p_enrich=0）、
  call_bonus_b=0、16000 steps、報酬正典3ストリーム不変（β=1、chip_value=5）
- 基底 config = `freeparlor/configs/ppo_stage1 相当`（Stage1 と diff が arm の1変数のみに
  なるよう作成。Stage3 config から call_bonus 3キーを 0 に戻したものと等価）
- run 規約は従来どおり: 日時 suffix run dir・tmux・preflight（残党/rebuild/検定全数 PASS）・
  発進ゲート → 凍結 → 判定窓 **step 8000–16000**
- 実施順（凍結）: **C 先行**（実装が config+pool の小変更で即発進可能）→ C 走行中に
  K を実装・検定 → C 完走後 K 発進（GPU 1系統ルール準拠。C の発進自体は
  DRCA 第3枠完走後）

## 2. Arm C — opponent pool への init 常駐

### 介入定義
`OpponentPool.sample()`（`mortal/opponent_pool.py:36-46`）に anchor 分岐を追加:

```
r ~ U(0,1)
r < anchor_prob             → anchor_checkpoint（凍結 init）を返す
それ以外                     → 従来ロジック（latest_prob → latest / uniform past_k）
```

- 新 config キー（`[opponent_pool]`）: `anchor_prob`（デフォルト **0.0** = 設計された OFF、
  既存 run と bit-compatible）、`anchor_checkpoint`（デフォルト空 = OFF）
- **anchor_prob = 0.25（凍結、§9）**（相手3席の各 draw が独立に 25% で init。
  3席全てが init になる確率 1.6%、少なくとも1席 init 58%）
- 変更ファイル: `opponent_pool.py`（sample 分岐）+ `player.py:245` 付近（config 配線）+
  構成 dump に `anchor_prob`/`anchor_checkpoint` を追加（eval 経路 0/空 assert）

### 期待される読み出し
- 効く場合: 「強い相手の攻めに正しく対応する局面」が訓練分布に復活 → 放銃率の
  劣化が止まる/縮む。損失は無変更なので、効けば「分布の問題」と確定
- 効かない場合: 均衡退化は単独犯ではない → K（損失側）の必要性が強まる

## 3. Arm K — 凍結 init への KL ペナルティ

### 介入定義
`ppo_loss`（`mortal/ppo.py:79-109`）に第4項を追加:

```
KL_ref = E_batch[ KL( π_θ(·|s) ‖ π_ref(·|s) ) ]     # 合法手 mask 上の full KL（46行動、closed form）
total = policy_loss + c_vf·value_loss − c_ent·entropy + kl_beta·KL_ref
```

- **π_ref = PPO step 0 の方策そのもの**（init checkpoint から
  `load_ppo_from_mortal_checkpoint` で構築した Brain+ActorCritic の凍結コピー。
  `requires_grad=False`・eval mode・以後一切更新しない）
- 方向は forward KL（π_θ‖π_ref、mode-seeking = RLHF 標準）。行動空間が 46 と小さいので
  sampled 推定ではなく **masked full KL**（低分散・decision-step ごと閉形式）
- ref forward は rollout バッチ受領時に **epoch ループの外で1回だけ** no_grad 計算し
  キャッシュ（PPO の複数 epoch で不変のため。GPU メモリ +10.8M params ≈ +数百MB、
  現行使用量 ~2.5GB に対し余裕）
- 新 config キー（`[ppo]`）: `kl_beta`（デフォルト **0.0** = 設計された OFF。
  β=0.0 では ref モデルを**ロードすらしない** = 既存経路ビット不変）、
  `kl_ref_checkpoint`（デフォルト空 = init_checkpoint を流用）
- **anneal はしない（一定 β）**: Stage3 の教訓「報酬介入は anneal とセット」は
  一時的足場の話。アンカーは恒久レギュラライザであり、切れば §4 主犯が復活する
- **kl_beta = 0.1（凍結、§9）**。事前スモークによる量級較正は**不採用**:
  π_θ は init から warm-start するため step 0 で KL(π_θ‖π_ref) = 0 であり、
  スモーク規模では KL が立ち上がらず較正情報にならない。スモークは配管検査
  （kl_anchor イベントが出る・有限値・NaN なし）のみに用いる（400 step 規律準拠）。
  β の見込み違いは §6a の再走規定（最大1回・機械的）で扱う
- diag ログ: 毎バッチ `kl_anchor` イベント（kl_beta / kl_ref_mean / kl_term_total）を
  分離ログ（Stage3 `call_bonus` イベントの前例踏襲）

### 失敗モードと診断
- **アンカー過強**（init クローン）: kl_ref_mean ≈ 0 に張り付き + 判定2（チップ獲得）不成立
- **アンカー過弱**（劣化継続）: kl_ref_mean 単調増大 + 判定1（基礎維持）不成立
- 再試行の扱いは §6a に凍結（K は最大1回・機械的な β 変更のみ、C は1発勝負）

## 4. 検定の拡張（`verify_ppo_p1.py`、実装タスクに含む）

- **(19) anchor pool OFF 恒等性**: `anchor_prob=0.0`/キー不在で `sample()` の返り値分布が
  従来実装と一致（シード固定で系列一致）。ON（=1.0 強制）で anchor_checkpoint のみ返る
- **(20) KL 項の正確性と OFF 恒等性**: (a) `kl_beta=0.0`/キー不在で loss dict が
  既存実装とビット一致 + ref 未ロード確認、(b) 合成 logits での masked KL の
  手計算一致、(c) β>0 で ref パラメータに grad が流れない（requires_grad/grad None assert）
- **(17d/18c 同居分)**: eval 構成 dump に `anchor_prob=0`・`kl_beta=0` assert を追加
  （eval 経路は常時両介入 OFF）

## 5. 発進ゲート（v2 教訓準拠: 機械ゲートのみ、学習応答ゲートは置かない）

@step200 の機械ゲート（`ppo_diag.jsonl` から算出、`check_stage3_launch_gate.py` の
機械段の踏襲）:
- Arm C: 全 client の opponent 選択ログで anchor 採択率が anchor_prob ± 0.05
  （pool draw のログ出力を追加実装。n は3席×数百 draw で十分）
- Arm K: 全バッチ kl_beta = 設定値、kl_ref_mean > 0（かつ発散兆候 NaN/inf 無し）

学習応答ゲートは設定しない（Stage3 v1 の較正ミス教訓: 本系列の効果量の
事前較正データが存在しないため、恣意的な閾値は置かず判定窓で正式評価する）。

## 6. 判定条件（**凍結済み — 変更禁止**）

すべて step16000 checkpoint、既存ハーネス（grp_baseline 1v3、n は下記）、argmax/guard ON。

1. **主判定1（基礎維持）**: 放銃率(ckpt) − 放銃率(init) の差が cluster-robust
   （半荘クラスタ）で **z < 2**（有意劣化なし）。測定は
   `analyze_fundamentals_1v3.py`（commit 済み・Stat 値一致で検証済み）による。
   参考副読: avg_rank 差 n.s.
2. **主判定2（経済適応の獲得）**: チップ/半荘(ckpt−init) が **+方向 かつ ≥1SE**。
   検出力対策として 1v3 は両脚とも **n=800 半荘**（seeds [10000,10200)、
   座席ローテは既存ハーネス準拠）に倍増する
   （Stage1 実測 +0.52±0.35@400 が 800 なら ~2.1SE 相当。eval コスト増は数時間で許容）
3. **分岐表**:
   - 判定1◯ + 判定2◯ → arm 成功。合流形 A / 商用候補の議論へ（0b 接続）
   - 判定1◯ + 判定2✗ → 基礎は守れたが適応が乗らない（K ならアンカー過強を疑う）
   - 判定1✗ → arm 失敗（C 単独失敗は K の主犯説を強化、K 失敗は仮説自体の再検討）
4. **副次（記録のみ・判定外)**: 2レンズ規律どおり sampled action_mass も併記。
   メタ対決（ckpt vs init×3 — ミラー較正脚 [バックログ5] の初適用を兼ねる）、
   鳴き率・立直率の自然分布推移、（K のみ）kl_ref_mean 軌跡

### 6a. 再走規定（凍結・機械的適用のみ許容）

- **Arm C は1発勝負**（再走なし。判定1✗なら均衡退化の単独犯説は棄却方向 → K の結果待ち）
- **Arm K は最大1回の再走を許容**。条件と β の変更は機械的に:
  - 判定1◯ かつ 判定2✗（アンカー過強の疑い）→ **kl_beta / 4** で再走
  - 判定1✗（アンカー過弱）→ **kl_beta × 4** で再走
  - 再走 run は新規 run dir・同一判定条件（§6）。再走後の結果が最終（それ以上の再走禁止）。
    適用時は使用した β と結果を §9 に追記する（この追記は凍結違反に当たらない）

## 7. コスト見積り

| 項目 | 見積り |
|---|---|
| Arm C 実装+検定 | 小（pool 分岐 ~10行 + 配線 + 検定19。0.5日） |
| Arm K 実装+検定 | 中（ref モデル管理 + loss 項 + diag + 検定20。1日） |
| 各 run（16k steps） | GPU ~1–1.5日（Stage3 実績準拠）+ drain ~230GB（現在空き 914GB、余裕） |
| 各 eval バッテリー | 800半荘 1v3 + argmax 6ckpt + メタ較正脚。GPU 数時間〜半日 |

GPU 直列運用（凍結時 Gamba 裁定 2026-07-25）: DRCA 第3枠はこのまま完走 →
**次の GPU 枠は Arm C 発進**。DRCA 残枠（a_s3mid / b_s3final）は anchor 系列より
後回し（実施可否は改めて裁定）。

## 8. 本家との関係 / 禁則の遵守

- 訓練 rollout への行動上書きなし（C は相手選択の分布、K は損失項 — いずれも
  π からの純サンプリングは不変)
- 新規 Rust 表面積ゼロ。libriichi 無変更
- サイレントフォールバック禁止準拠: ref ロード失敗・anchor checkpoint 不在は
  発進時に loud FAIL（デフォルト OFF 時はロード自体をしない）

## 9. 凍結記録（2026-07-25、Gamba 裁定。本改訂 commit = 事前登録、以後変更禁止）

- [x] **arm 採否・実施順**: 両 arm 採用、**C → K**。C 発進は DRCA 第3枠完走後の
      GPU 解放時、K は C 走行中に実装・C 完走 eval 後に発進
- [x] **anchor_prob = 0.25**、anchor_checkpoint = `beta1_huber_192x40/mortal.pth`
      （init と同一パス）
- [x] **kl_beta = 0.1**（スモーク帯域較正は不採用 — §3 参照。変更は §6a の
      機械的再走規定のみ）
- [x] **判定条件 = §6 本文**（判定1: 放銃差 z<2 / 判定2: チップ +方向かつ≥1SE /
      1v3 両脚 n=800・seeds [10000,10200)）。再走規定 = §6a（K 最大1回、C 1発勝負）
- [x] **判定窓 = step 8000–16000**（従来踏襲）
- [x] **eval バッテリー** = 標準 argmax 6ckpt + grp_baseline 1v3（n=800）+
      ミラー較正脚（バックログ5 ハーネス初適用）+ メタ対決（ckpt vs init×3）
- [x] **run 命名** = `anchor_c_<日時>` / `anchor_k_<日時>`（日時 suffix・再利用禁止準拠）

（§6a 適用時の追記欄: 未適用）
