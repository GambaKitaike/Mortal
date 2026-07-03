# PPO 移行設計 — 教師データ非依存 本線の実装設計 (2026-07-02)

**位置づけ:** `reward_design_teacherfree.md`（(あ) 報酬設計）を実装に落とす設計書。
分岐（critic スケール処理・希少性探索）は 2026-07-02 のセッションで確定済み。
本書はその確定を前提に、アーキテクチャ・損失・自己対戦構成・段階計画を定式化する。

---

## 0. 前提（確定事実と確定判断）

| 事実/判断 | 根拠 |
|---|---|
| β=1 崩壊の主因はスケール（MSE の外れ値支配）。**192×40 でも確定** | `beta1_huber_192x40_verify.md` Q1（和了14%→40%, avg_rank 1.246→1.006, dqn_loss のみ差替・同 batch/step） |
| Huber(δ=15) はこの報酬分布の重尾に効く（**一次証拠あり**） | 同上 + `beta1_huber_verify.md` |
| 副露局収支は門前とパリティ（+3207 ≈ +3246）。64×10 の「逆転」は確定 claim にしない | `beta1_huber_192x40_verify.md` Q3 + 補足（学習量非対称） |
| 経済（β=1）は「鳴きペナルティを外すレバー」であり「鳴きへ引っ張るレバー」ではない | Q3 パリティの含意 |
| 赤鳴き正例は 20件/28672局 — 希少性は報酬式の射程外 | `aka_call_hora_count.md`, `reward_design_teacherfree.md` §4 |
| **確定判断1: critic = Huber(δ=15) 継続 + advantage per-batch 正規化。リターン正規化は不採用、symlog は保険杭** | 本セッション分岐1 |
| **確定判断2: 希少性は段階制。Stage1 純エントロピー → Stage2 赤濃縮カリキュラム → Stage3 構造化探索（封印気味）** | 本セッション分岐2 |

### コード実地調査で判明した制約（2026-07-02、本設計に直接効く）

| 発見 | 出所 | 含意 |
|---|---|---|
| 牌山は `const UNSHUFFLED: [Tile; 136]`（コンパイル時定数、赤は各色1枚固定） | `libriichi/src/arena/board.rs:822` | 赤枚数の増加はコンパイル時改修が必要 |
| 状態表現は `akas_in_hand: [bool; 3]` — **各色高々1枚を前提** | `libriichi/src/state/` 一帯（`action.rs:221`, `agent_helper.rs` 他） | **「赤5枚/8枚ルール」は観測・行動・mjai 全体に波及する大改修 → 不採用**。Stage2 の実装形を「配牌 rejection sampling」に差し替え（§5.2） |
| DQN は dueling（`v_head` + `a_head`）。`v_chip_head`/`a_chip_head` は Phase5 遺産 | `mortal/model.py:206-` | chip ヘッドは PPO ブランチで撤去。`a_head` は policy head の初期化に転用可能（§2.3） |
| GRP は Brain と独立の LSTM モデル | `mortal/model.py:349-` | PPO 移行で GRP 本体は不変更。訓練データ源の教師依存は宿題のまま（§6） |

---

## 1. スコープ（何を捨て、何を残すか）

| 項目 | 扱い |
|---|---|
| Brain（ResNet エンコーダ 192×40） | **残す**。β=1 Huber 192×40 checkpoint から初期化 |
| DQN ヘッド（v/a） | 学習からは撤去。`a_head` の重みは policy head 初期化に**転用**（§2.3） |
| Q_chip / ChipDQNTarget / β_sel / chip_n_step | **全廃**（`reward_design` 分岐3） |
| CQL | **撤去**（on-policy 化で OOD 過大評価の機構ごと消える） |
| opp 項（λ_opp） | **廃棄**（`reward_design` §2） |
| 教師データ（2009天鳳） | 学習ループから**撤去**。初期化 checkpoint 経由の間接影響のみ残る（明記） |
| GRP | **現行を凍結して流用**（§6）。self-play 再訓練は P4 に繰延 |
| online 3プロセス基盤（server/trainer/client×3） | **流用**。転送ペイロードを PPO transition に差し替え |
| 報酬式 | `reward_design` §2 のまま: α·素点/1000 + γ·GRPΔ + β·chip×5.0、α=β=γ=1 |

**ブランチ運用:** `ppo-migration` ブランチを切り、DQN 経路は main に無傷で残す
（比較 arm・切り戻し先として保全。削除ではなくブランチ分離）。

---

## 2. アーキテクチャ

### 2.1 ヘッド構成

```
Brain(obs) → phi
  ├─ PolicyHead(phi) → logits[ACTION_SPACE]  （mask 適用後 softmax = π）
  └─ ValueHead(phi)  → V(s)  スカラー
```

- PolicyHead / ValueHead は現行 DQN version に合わせた MLP（v2/v3 なら Linear→Mish→Linear）。
- 行動マスクは現行と同じ（illegal action は logits に −inf）。

### 2.2 エピソード境界と割引

- **エピソード = 局（kyoku）**。局末で terminal（V=0 ブートストラップ）。
- 局末 reward = α·(素点Δ/1000) + γ·(GRP 着順期待ウマΔ) + β·(チップΔ×5.0)
  — `reward_design` §2 と同一。GRP 密化により半荘レベルの credit は局単位に落ちている
  ため、局をエピソードとして閉じてよい（半荘跨ぎの割引設計を持ち込まない）。
- GAE(γ_disc, λ): γ_disc=1.0（局内は数十手・undiscounted で問題なし、現行 MC の
  `gamma^steps_to_done` 思想と整合）、λ=0.95 を初期値。

### 2.3 初期化（重要・タダで貰える連続性）

**π₀ = softmax(a_head(phi)/τ) は正確に Boltzmann(Q/τ) と一致する。**
dueling では Q = V + A − mean(A) で、V と mean(A) は行動間で定数 → softmax で相殺。
つまり **β=1 Huber 192×40 の `a_head` 重みを PolicyHead にコピーするだけで、
「検証済み checkpoint の Boltzmann 方策」から PPO を開始できる**。BC も蒸留も不要。

- τ（温度）は既存 boltzmann_temp=0.05 より高めから開始（探索確保、初期値 τ=1.0 で
  logits を 1/τ スケール）。スモークで方策エントロピーを見て調整。
- ValueHead は `v_head` 重みから初期化（V(s) の直系）。
- **教師の間接影響の明記:** 初期化 checkpoint は 2009天鳳で訓練されている。
  「教師撤去」は学習ループからの撤去であり、prior としての影響は残る。
  これは意図的（ゼロからの自己対戦はコスト非現実的）。

## 3. 損失と最適化

```
L = L_clip(θ)                          # PPO clipped surrogate, ε_clip=0.2
  + c_vf · Huber(V(s), R̂_GAE, δ=15)    # 確定判断1。リターン正規化しない
  − c_ent · H(π(·|s))                   # エントロピー正則化（Stage1 の探索本体）
```

- **advantage は per-batch 正規化**（mean/std）。critic ターゲットは生スケール（正規化なし）。
- c_vf=0.5、c_ent は初期 0.01 からスモークで調整。**c_ent の annealing は Stage1 では
  しない**（探索が本体のため一定維持、Stage 移行判定まで固定）。
- clip_grad_norm は現行踏襲。
- **保険杭（symlog）:** Huber は勾配支配を抑えるが critic の予測値の系統的過小評価は
  防がない。PPO では advantage の「大きさ」が policy gradient に直接効くため、
  **チップ和了局面で critic がチップ価値を恒常的に過小評価する症状**（診断: チップ和了
  局の TD 残差が恒常正）が実測されたら、symlog + twohot（DreamerV3 系）へ切替を検討。
  Stage1 の監視項目に入れる（§8）。

---

## 4. 自己対戦構成

- 3プロセス基盤（server×1 / trainer×1 / client×3）を流用。転送内容を
  `(obs, action, logp_old, mask, reward, done)` の trajectory に変更。
- **相手プール（Phase5 振動原因仮説2への対処、旧 Phase6b を吸収）:**
  対戦 4 席のうち trainee 1 席、残り 3 席は
  「最新 checkpoint 50% / 過去 K 個（K=5, save 間隔ごと）から一様 50%」で sample。
  → 方策が動いても学習対象分布の急変を緩和。
- on-policy 性の管理: client の行動方策は学習方策と同一 checkpoint（logp_old を保存）。
  checkpoint 更新ラグによる軽度の staleness は PPO ratio が吸収（ratio 分布を監視、§8）。
- **訓練 rollout への行動上書き（rule-based guard 含む）は禁止。eval は本家準拠で guard ON。**
- インフラ注意（Phase5 の教訓踏襲）: run 起動前に pkill + `ss -tlnp | grep 5000` で
  残党ゼロ確認、buffer/drain クリア。

---

## 5. 希少性の段階設計（確定判断2の実装）

### 5.1 Stage1 — 純エントロピー + 自己対戦

追加機構ゼロ。経済（β=1 Huber で鳴き局収支パリティ）+ エントロピー探索 + 自己対戦で
赤鳴きが内生的に立つかを検証する。**これを飛ばすと「経済+自己対戦だけで学べたか」が
永久に不明になる**（商用化判断の一級知見のため必ず走らせる）。

**移行判定（事前固定・変更禁止）:**
- 監視: 赤保持局面での鳴き**試行**率（打牌ログから、和了まで要らない）、鳴き和了率、
  副露Δ選択性、方策エントロピー。
- 判定窓: step 8,000–16,000（Phase5 で振動収束が見えた帯に合わせる）。
- **Stage2 へ移行する条件: 判定窓の平均で、赤保持局面の鳴き試行率が初期方策比 2 倍未満、
  かつ上昇トレンド無し**（両方成立で「探索不足」と判定）。
- 片方でも不成立（伸びている）なら Stage1 継続。

### 5.2 Stage2 — 赤濃縮カリキュラム（実装形を修正）

**旧案（赤5枚/8枚ルール）は棄却**（§0 コード調査: `akas_in_hand:[bool;3]` 前提が
観測・行動・mjai に波及、改修コスト過大 + 商用ルールからの乖離）。

**新実装形: 配牌 rejection sampling（赤入り配牌への条件付け）**
- 牌山 136 枚・赤3枚は**一切変えない**。シャッフル後、trainee 席の配牌 13 枚に
  赤が 1 枚以上含まれるまで再シャッフル（`board.rs` のシャッフルループに条件、
  enrich 確率 p_enrich で発動）。
- 状態表現・行動空間・mjai・点数計算すべて無傷。分布介入は「初期局面の頻度」のみ。
- 自然発生率: 配牌 13 枚に赤≥1 は約 25%（13×3/136 ≈ 0.287 の近似、正確値は実装時に
  実測ログで確認）→ p_enrich=1.0 で約 4 倍の遭遇率。
- **anneal:** p_enrich 1.0 → 0（自然分布）へ段階降下。降下スケジュールは Stage2 開始時に
  Stage1 の伸び方を見て設計。
- **評価は常に自然分布で行う**（enrich は訓練 client のみ。test_play / 打牌統計は
  p_enrich=0 固定）。value の分布シフト（P(赤|配牌) の過大学習）は既知のトレードオフ
  として明記、anneal で解消を確認する。

### 5.3 Stage3 — 構造化探索（封印気味）

Stage2 でも「遭遇はしているのに学習が立たない」場合のみ。そのとき診断は
「遭遇不足」→「credit assignment 不足」に変わっているはずで、探索ボーナス
（anneal 付き B2）を初めて検討する。**§2 の思想（明示的な鳴き項を置かない）との
衝突を認識した上での最終手段**であることを本書に杭として明記。

---

## 6. GRP の扱い

- **P1–P3: 現行 GRP を凍結流用**。理由: (i) チップ盲は正しい（`reward_audit` 確定）、
  (ii) 着順力学はルール駆動で教師依存が軽い、(iii) 変数を増やさない。
- **既知の宿題（`reward_design` §6 から繰越）:** GRP 訓練データ源の教師依存度は未確認。
  自己対戦方策が人間分布から離れるほど GRP の着順期待が out-of-distribution になる
  リスクがある。**監視項目に GRP 予測 vs 実着順の calibration を追加**（§8）し、
  乖離が出たら P4（GRP の self-play 再訓練）を起動。

---

## 7. 実装フェーズ分割

| Phase | 内容 | GPU | 完了条件 |
|---|---|---|---|
| **P0** | 本設計書 + docs ロック | 不要 | 本書 commit |
| **P1** | 配管: `ppo-migration` ブランチ、PolicyHead/ValueHead、a_head/v_head 初期化ローダ、GAE 計算、PPO 損失、client→trainer の trajectory 転送、chip ヘッド/CQL/opp の撤去 | 不要 | 単体 sanity 全通過（§7.1） |
| **P2** | スモーク: 小予算（数百 step）で NaN 無し・ratio 分布正常・エントロピー挙動・avg_rank 崩壊無しを確認。c_ent / τ / lr の初期調整 | 小 | スモーク md |
| **P3** | Stage1 本走（判定窓 step16k まで）+ 移行判定 | 中 | Stage1 判定 md |
| **P4** | （条件付き）Stage2 rejection sampling / GRP 再訓練 | 中 | — |

### 7.1 P1 の単体 sanity（Composer 必須項目）

- [ ] π₀ 一致検定: PolicyHead 初期化直後、ランダム obs バッチで
      `softmax(policy_logits/τ)` ≡ `softmax(a_head/τ)`（数値一致、atol=1e-5）
- [ ] mask 検定: illegal action の π が厳密 0
- [ ] GAE 検定: 手計算の小系列（3 step）と一致
- [ ] logp_old 整合: client 保存値と trainer 再計算値が一致（同 checkpoint）
- [ ] 報酬合成: 既存 `calc_delta_blend` 系と同値になるユニットケース
      （素点のみ / チップのみ / 混合の 3 ケース）
- [ ] chip ヘッド撤去後も既存 checkpoint の load が壊れない（strict=False の範囲明記）

---

## 8. 監視・評価（P2 以降のダッシュボード）

| 項目 | 閾値/目的 |
|---|---|
| avg_rank（自己対戦） | ≈2.5 から乖離しない（打牌健全性） |
| 放銃率 | 13–18% 帯（Phase5 の健全帯踏襲） |
| 方策エントロピー | 単調崩落しない（探索維持）/ 高止まりしない（学習進行） |
| PPO ratio 分布 | clip 域外比率 <~30%（staleness 監視） |
| critic explained variance | 上昇トレンド |
| **チップ和了局の value 残差** | 恒常正なら critic 過小評価 → symlog 検討（§3 保険杭） |
| **GRP calibration**（予測着順 vs 実着順） | 乖離拡大なら P4 起動（§6） |
| 赤保持局面の鳴き試行率 / 鳴き和了率 / 副露Δ | Stage1 移行判定（§5.1） |
| チップ実現率 / 局収支（素点+chip×5000） | 商用評価軸への布石（`next_steps_2` §5） |

評価は**打牌統計 + 自然分布**で（enrich 中も test は p_enrich=0、自己対戦 avg_rank≈2.5 は
sanity のみ、競技比較は test_play 経路 — 従来通り）。

---

## 9. sanity / 留意

- 本書は紙設計。数値初期値（τ, c_ent, λ, K, p_enrich anneal）は P2 スモークで確定させ、
  本書に追記して版を上げる。
- Stage1 の移行判定条件（§5.1）は**走らせる前に固定**し、結果を見てからの変更を禁止する
  （premature conclusion の逆型 = post-hoc goalpost 移動の防止）。
- 「教師撤去」の正確な意味（学習ループからの撤去、初期化 prior は残る）を §2.3 に明記済み。
- 希少性が Stage1 で解けた場合も Stage2 実装（rejection sampling）は安価なので、
  対照実験として一度は焼く価値がある（商用知見）。ただし優先度は下げる。
