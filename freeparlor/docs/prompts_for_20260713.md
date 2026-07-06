# 2026-07-13 投入用プロンプト集（Composer / Claude Code 共用）

投入順に 3 本。各プロンプトは 1 タスク = 1 push。次のプロンプトは前のレビュー通過後に投入。

---

## プロンプト①: Stage1 残タスク（eval バッテリー + docs 配置）

```
【前提】リポジトリ直下の CLAUDE.md を読み、全規律をそれに従うこと。
ブランチ ppo-migration。GPU は常に1系統（eval は直列実行）。

【タスク】Stage1 の残 eval と docs 配置。学習は行わない。

1. docs 配置:
   - ppo_p3_stage1_result.md と stage2_design.md（別途受領）を freeparlor/docs/ に配置
   - CLAUDE.md の「現在の状態」節を更新（Stage1 判定完了 → Stage2 準備中）
2. 保全確認: run 7a/7b の checkpoints・ppo_diag.jsonl・tb の実在をサイズ付きで確認
3. argmax eval バッテリー（すべて自然分布・同一 seed [10000,10100)・各 100 半荘・直列）:
   (a) 標準: run7 checkpoint 2000/4000/8000/12000/16000 + mortal_init の 6 本
       表: fuuro / riichi / agari / houjuu / ryukyoku
   (b) grp_baseline 対戦 (1v3): step16000 と init の 2 本。avg_rank + 打牌統計
       （検定(9) の DQN エンジン互換経路を使用）
4. 結果を ppo_p3_stage1_result.md の「argmax eval」節として追記。
   サンプルレンズからの予想（riichi ほぼ常時宣言 / fuuro ≈ 0）との照合と、
   放銃率の健全性を明記。予想外の乖離があれば解釈せず数値のみ報告
5. commit & push 後、git ls-remote origin | grep ppo-migration を貼り
   リモート先端一致を確認して報告
```

---

## プロンプト②: Stage2 実装（rejection sampling + 検定）

```
【前提】CLAUDE.md と freeparlor/docs/stage2_design.md を読むこと。
本タスクは実装と検定のみ。発進はしない。

【タスク】配牌 rejection sampling の実装（stage2_design.md §2-3 が仕様）。

1. Rust (board.rs): 局ごとに確率 p_enrich で、trainee 席の配牌 13 枚に
   赤(5mr/5pr/5sr)≥1 が成立するまで再シャッフル。牌山・点数・観測・mjai は不変。
   p_enrich は config から供給（default 0.0 = 従来挙動と厳密一致）
2. 配線: 訓練 client のみ p_enrich を受け取る。eval 経路（test_play / eval_sanity /
   grp_baseline 対戦）は常時 0。全エンジンの構成 dump に p_enrich を含める
3. 検定 (17): self-play 少数ゲームで
   - p_enrich=1.0 → trainee 配牌の赤≥1 率 = 100% を assert
   - p_enrich=0.0 → 自然率（実測値を記録、~25% 想定）かつ既存挙動と統計一致
   - eval 経路の構成 dump で p_enrich=0 を assert
4. 自然分布での配牌赤≥1 の実測値を stage2_design.md §2 に追記
5. 全 17 検定 PASS のログ更新。config: freeparlor/configs/ppo_stage2.toml を
   Stage1 config のコピー + p_enrich=1.0 + run パスのみ差し替えで作成
   （先頭コメントに「diff vs stage1 config: p_enrich only」明記）
6. commit & push 後 ls-remote 貼り
```

---

## プロンプト③: Stage2 発進

```
【前提】CLAUDE.md / stage2_design.md。プロンプト②のレビュー通過後のみ実行。

【タスク】Stage2 本走の発進。

1. preflight: 残党チェック + libriichi rebuild + 全 17 検定 PASS
2. 発進: 新規 run dir（stage2_<日時>）、tmux、init = beta1_huber_192x40、
   config = ppo_stage2.toml（変更禁止）、16,000 steps
3. 開始報告:
   - step 100 到達 + 監視項目（mismatch/fallback/chip=0、loader_delta INFO、
     alive clients 3/3）+ 実効 step/s + Mem/GPU 30min
   - 【操作チェック・発進ゲート】step 500 時点で、鳴き可能局面のうち赤保持の割合
     （action_mass の n_call_possible_aka_held / n_call_possible 累計）≥ 54% を確認。
     未達なら run 停止して報告（stage2_design.md §3）
4. ゲート通過後に凍結宣言 → step 16000 完走まで変更禁止。
   完走時は完走報告のみ（eval・判定集計は別タスク）
5. commit & push 後 ls-remote 貼り
```

---

## 備考

- 判定集計（窓 8000–16000、判定 3 分岐）は Stage1 と同様、設計監督側（Claude chat）が
  ppo_diag.jsonl から直接実施可能。課金状況に応じてどちらでやるか選択
- プロンプト①の (b) で立直マシン（run7 step16000）の対 grp_baseline 成績が出る。
  これは「立直全ツは外部 baseline に対しても強いのか」への最初のデータ点であり、
  Stage2 の解釈（§0 の解釈の限界）に direct に効くので結果に注目
