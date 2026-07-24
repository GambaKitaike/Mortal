# run 成果物の保持・清掃運用（drain 恒久運用）

**ステータス: 制定（2026-07-16、バックログ7 の明文化）。**
2026-07-09 ディスク枯渇インシデントの再発防止策のうち「運用の明文化」を担う文書。
launcher 側の防御（DISK_MIN_GB preflight）は §3 参照。

---

## 0. 経緯

- **2026-07-09 ディスク枯渇クラッシュ**: `/`（1TB）が過去 run 群（921GB）+ 進行中
  Stage2 run 自身の `drain/`（6.5h で 167GB）で 100% 枯渇。server `ENOSPC` →
  trainer `UnexpectedEOF` → `/tmp` 同居のため tmux サーバーも道連れ。データ整合性は
  無傷だったが run は resume を要した
- **2026-07-11 一括清掃**: 判定完了済み 4 run（stage1×2・stage2×2）の `drain/`
  計 ~613GB を Gamba 承認の下、許可リスト方式で削除（251GB → 863GB 空きに回復）。
  同日 `run_ppo_stage3.sh` に DISK_MIN_GB preflight を追加
- しかし**運用が明文化されていなかった**ため、Stage3 本走
  （`stage3_20260712_033403`）の `drain/` 368GB が判定完了（2026-07-13）後も残置
  されたままになっていた（2026-07-16 の棚卸しで発見 — §5）。イベント駆動の清掃
  ルールを本書で恒久化する

## 1. 成果物クラスと保持ポリシー

| クラス | 例 | ポリシー |
|---|---|---|
| `checkpoints/` | `step_*.pth` | **恒久保全（削除禁止）** — 判定・eval・再現の根拠 |
| `logs/`（`ppo_diag.jsonl` 等）・`tb/`・`config.toml`・`mortal*.pth` | | **恒久保全（削除禁止）** — 判定集計の一次ソース |
| `test_play/`・`train_play/` | eval 牌譜 json.gz | **恒久保全** — 軽量（~数十MB）かつ定性レビュー資産 |
| `drain/` | 消費済み生ロールアウト（step 別サブディレクトリ） | **当該 run の判定/裁定が md に commit された後、削除可**（§2 の承認手順必須）。判定前は run の一部として温存 |
| `buffer/` | `*.traj`（client→server 受領ステージング領域） | **用途確定（2026-07-24 調査）**: client が送信した rollout を server が受領・蓄積し、trainer の drain 要求で `drain/` へ move する**前段バッファ**（`server.py` buffer_dir、move は同 118–130 行）。run 終了後の残量 = **未 drain・未消費の in-flight データ**（trainer 終了後の残党 client 生産物。stage2_resume で mtime 実証: trainer 最終 checkpoint 11:31 に対し buffer 最古 11:34–最新 13:50 = 全量が終了後生産）。参照コードは `server.py` のみで、**起動時に rmtree→再作成**するため resume にも不要。→ **drain と同ポリシー（判定/裁定 commit 後、§2 手順で削除可）** |
| aborted run（`aborted<N>_` prefix） | | 裁定に使った `logs/`・`config`・`checkpoints/` は恒久保全。**`drain/` は裁定 commit 後に削除可** |
| smoke run | | 対応する検証結果が md に commit された後、run dir ごと削除可 |
| DRCA 採取物 | `bp*.jsonl`・`*.logs/`・script sidecar・`probe_*.jsonl` | **恒久保全** — 測定の再現根拠かつ軽量（run あたり ~10MB） |
| `backups/drca/` の tar | 移設退避用 | 対応する run の判定 commit 後に削除可 |

## 2. 清掃のトリガと手順

**トリガ（イベント駆動が主、定期清掃はしない）:**

1. **判定/裁定の md commit 後**、次の run 発進準備までに、当該 run の `drain/` を
   本手順で清掃する（判定タスクの残務として扱う — 清掃せず放置すると次の launcher
   preflight で FATAL になるのが検出線）
2. launcher の DISK_MIN_GB preflight が FATAL を出した場合、本手順で清掃してから
   再発進する（閾値を下げて通すのは禁止）

**手順（2026-07-11 実績方式の踏襲）:**

1. 削除候補を**許可リスト**としてフルパスで列挙し、`du -sh` で削除前サイズを記録
2. **Gamba の明示承認を得る（毎回・リスト単位。包括承認や「以後自動で」は不可）**
3. 削除は許可リストのパスに限定（`rm -rf` の引数は列挙したパスのみ。glob 展開で
   リスト外に及ぶ書き方をしない）
4. `df -h /` を前後で記録し、当該 run の `checkpoints/`・`logs/`・`config` の
   実在を削除後に確認する
5. 報告（削除前 du・df 前後・実在確認）を CLAUDE.md「現在の状態」または対応する
   report md に残す

## 3. 発進側の防御（launcher preflight、実装済み）

| launcher | DISK_MIN_GB デフォルト | 根拠 |
|---|---|---|
| `run_ppo_stage3.sh`（学習系） | 450GB | 16k step 学習 run の drain 実測: Stage3 = 368GB/16k（~23GB/1000step）。×1.2 マージン |
| `run_drca_main_frame.sh`（DRCA） | 100GB | 採取 json.gz + probe 成果物は run あたり ~10MB オーダーだが、他系統の余裕込み |

新規の学習系 launcher を作る場合は 450GB を下回らないこと。

## 4. 既知の蓄積メカニズム（設計由来、恒久対策は未実施）

- `drain/` は step ごとにサブディレクトリが作られ、**run 中は一切クリーンアップ
  されない**（`server.py` は起動時にのみ rmtree する設計）。16k run 1本で
  ~370GB 級に育つのは仕様であり、run 中の削除は凍結ルール（データ整合性）に
  抵触しうるため**行わない**。清掃は必ず判定後
- `/tmp` が同一 `/` 上にあるため、枯渇時は tmux サーバーごと死ぬ（クラッシュ後の
  現場保全が悪化する）。インシデント時は最終 checkpoint の健全性確認を最優先
- watchdog×Cleanup 競合（バックログ4）は別問題（本書のスコープ外）

## 5. 現況スナップショット（2026-07-16 棚卸し、2026-07-24 更新）と清掃候補

初回棚卸し時 `df`: 495GB 使用 / 462GB 空き（52%）。drain 3件は 2026-07-16 に
§2 手順の初回適用として削除実施済み（Gamba 承認・411GB 回収 → 872GB 空き）。
2026-07-24 に buffer/ の用途確定（上記分類表）を受け候補4件を追加。

| パス | サイズ | 分類 | 処置 |
|---|---|---|---|
| `runs/ppo/stage3_20260712_033403/drain/` | **368GB** | 判定完了済み（2026-07-13、`ppo_p3_stage3_result.md` §9） | **削除実施済み（2026-07-16）** |
| `runs/ppo/aborted1_stage3_20260711_141705/drain/` | 31GB | 裁定完了済み（2026-07-11、ゲート v2 amendment の根拠は `logs/ppo_diag.jsonl` で drain 非依存） | **削除実施済み（2026-07-16）** |
| `runs/ppo/stage1_20260705_014852/drain/` | 13GB | 判定非関与の早期 run（判定根拠 run は `053301`・`020120_resume` のみ） | **削除実施済み（2026-07-16）** |
| `runs/ppo/stage2_20260709_194510_resume/buffer/` | 36GB | `.traj` ×1660 — 全量 trainer 終了後の残党生産・未消費（2026-07-24 用途確定、上記分類表参照） | **削除可 — 承認待ち** |
| `runs/ppo/stage1_20260706_020120_resume/buffer/` | 5.3GB | `.traj` ×240 — 同上（未 drain の in-flight データ。判定根拠 = checkpoints/logs/config は buffer 非依存） | **削除可 — 承認待ち** |
| `runs/ppo/stage3_20260712_033403/buffer/` | 1.4GB | `.traj` ×60 — 同上 | **削除可 — 承認待ち** |
| `runs/ppo/stage2_20260709_092541/buffer/` | 154MB | `.traj` ×71 — 同上（クラッシュ時の in-flight） | **削除可 — 承認待ち** |
| その他（online_* 系 ~12GB、phase4* ~6GB 等） | ~20GB | DQN 経路の遺産（main ブランチ管轄） | 本書のスコープ外 |

現在の承認待ちは buffer/ 4件で **計 ~43GB** 回収可能（承認後、§2 手順で実施）。
実施タイミングは任意（削除は GPU 非依存のため DRCA 測定中でも実施可）。
なお空き 872GB（2026-07-24 時点 871GB）につき緊急性は低い — 次回 preflight
FATAL 時のイベント駆動清掃に回しても可。
