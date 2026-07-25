#!/usr/bin/env python3
"""フリー雀荘収支 analyzer (challenger視点, OneVsThree 1v3 eval game_logs)。

入力は eval_grp_baseline_1v3.py 等が書き出す game_logs ディレクトリ
(`{seed}_{key}_{split}.json.gz` の集まり)。split a/b/c/d は
libriichi/src/arena/game.rs の `["a","b","c","d"][game_idx % 4]` と
libriichi/src/arena/one_vs_three.rs の座席割当により challenger の
座席 0/1/2/3 に一意に対応する(eval_grp_baseline_1v3.py 内コメント参照)。

各半荘について challenger 視点で以下を算出する:
  - 素点: ログ終端の最終得点から (最終点 - 30000) / 1000
    (25000持ち30000返し。RETURN_SCORE 参照)
  - 順位点: ワンツーウマ+オカ。値は mortal/reward_calculator.py の
    RewardCalculator が使う pts と同一
    (freeparlor/configs/*.toml のほとんどで `pts = [35, 5, -15, -25]` として
    使われている値。同着の着順決定は libriichi/src/rankings.rs の
    `Rankings::new`(降順 stable sort、同点は座席番号が若い方が上位)と
    同一のタイブレークを使う)
  - チップ収支: mortal/chip_from_log.py の
    load_kyoku_chip_deltas_from_log をそのまま呼び出す(ロジック複製禁止)
  - 合算: 素点 + 順位点 + チップ × CHIP_VALUE (チップ1枚 = 5000点相当 = 5.0)

素点の再構成についての注記(重要):
json.gz ログは各半荘の生 mjai イベント列であり、GameResult.scores
(供託精算後の最終スコア)そのものは一切イベント化されない
(libriichi/src/arena/result.rs GameResult::dump_json_log 参照)。
特に、対局終了時に未収供託(見送りリーチ棒)が残っている場合、その精算は
libriichi/src/arena/game.rs Game::commit() でスコア最大者に加算されるが、
この加算は一切イベントとして書き出されない。本スクリプトは:
  1. 各 `start_kyoku.scores` を正とし、`reach_accepted`(-1000、deltas
     フィールド自体を持たない)と `hora`/`ryukyoku` の `deltas` を順に
     加算して素点を再現し、次の `start_kyoku.scores` との突合を全件行う
     (乖離があれば loud に落とす)。
  2. 対局終端で合計が 100000 (25000持ち×4) を下回っていれば、その不足分を
     未収供託とみなし、Game::commit() と同一のタイブレーク
     (`min_by_key(|s| -**s)` = 同点なら先頭 index 優先)でスコア最大者に
     加算する。
不足分が 1000 の倍数でない等、想定と食い違う場合は skip せず例外を投げる。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "mortal"))
from chip_from_log import load_kyoku_chip_deltas_from_log, open_log  # noqa: E402

FILENAME_RE = re.compile(r"^(?P<seed>\d+)_(?P<key>\d+)_(?P<split>[abcd])\.json\.gz$")
SPLIT_TO_SEAT = {"a": 0, "b": 1, "c": 2, "d": 3}

INIT_SCORE = 25000
RETURN_SCORE = 30000  # 配給原点(25000持ち30000返し)
TOTAL_SCORE = INIT_SCORE * 4  # 素点保存則(未収供託検出)に使う

# 順位点(ウマ+オカ)テーブル。25000持ち30000返し、ウマ10-20相当。
# mortal/reward_calculator.py RewardCalculator と freeparlor/configs/*.toml の
# 大半で使われている値 `pts = [35, 5, -15, -25]` と同一。
RANK_PTS = (35.0, 5.0, -15.0, -25.0)

CHIP_VALUE = 5.0  # チップ1枚 = 5000点相当 = 5.0 (千点単位)

# 理論ミラー値: 全4席が同一方策の対称対戦 (X vs 3X) での challenger 視点期待値。
# 座席対称性から順位点・チップは 0、素点は返し点オフセット (INIT-RETURN)/1000。
# ミラー較正脚 (backlog 5): 実測がこの理論値から SE 圏内に収まることを確認し、
# レンズ3 (メタ対決) の「理論ミラー値からの逸脱」解釈のゼロ点をハーネス実測で
# 裏取りする (座席ローテの非対称・有限標本バイアスの検出)。
MIRROR_SOTENSU = (INIT_SCORE - RETURN_SCORE) / 1000.0  # = -5.0
MIRROR_RANK_PTS = 0.0
MIRROR_CHIP = 0.0
MIRROR_COMBINED = MIRROR_SOTENSU + MIRROR_RANK_PTS + MIRROR_CHIP * CHIP_VALUE  # = -5.0


class LogFormatError(RuntimeError):
    """ログ形式が本スクリプトの想定と食い違う場合に loud に投げる。"""


@dataclass
class HanchanResult:
    path: Path
    n_kyoku: int
    sotensu: float
    rank_pts: float
    chip_total: float
    combined: float
    chip_per_kyoku: np.ndarray


def seat_from_filename(path: Path) -> int:
    m = FILENAME_RE.match(path.name)
    if m is None:
        raise LogFormatError(f"unexpected filename format: {path.name}")
    return SPLIT_TO_SEAT[m.group("split")]


def load_events(path: Path) -> list[dict]:
    with open_log(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def reconstruct_final_scores(events: list[dict], path: Path) -> list[int]:
    """全イベントを再生して対局終端の素点(供託精算後)を復元する。"""
    scores: list[int] | None = None

    for ev in events:
        et = ev.get("type")

        if et == "start_kyoku":
            snap = ev.get("scores")
            if snap is None or len(snap) != 4:
                raise LogFormatError(f"start_kyoku missing/malformed scores in {path}")
            if scores is not None and list(scores) != list(snap):
                raise LogFormatError(
                    f"score reconstruction mismatch before a start_kyoku in {path}: "
                    f"replayed={scores} logged={list(snap)}"
                )
            scores = list(snap)

        elif et == "reach_accepted":
            if scores is None:
                raise LogFormatError(f"reach_accepted before any start_kyoku in {path}")
            actor = ev.get("actor")
            if actor is None or not (0 <= actor < 4):
                raise LogFormatError(f"reach_accepted missing/invalid actor in {path}: {ev}")
            scores[actor] -= 1000

        elif et in ("hora", "ryukyoku"):
            if scores is None:
                raise LogFormatError(f"{et} before any start_kyoku in {path}")
            deltas = ev.get("deltas")
            if deltas is None or len(deltas) != 4:
                raise LogFormatError(f"{et} missing/malformed deltas in {path}: {ev}")
            for i in range(4):
                scores[i] += deltas[i]

    if scores is None:
        raise LogFormatError(f"no start_kyoku event found in {path}")

    total = sum(scores)
    deficit = TOTAL_SCORE - total
    if deficit < 0 or deficit % 1000 != 0:
        raise LogFormatError(
            f"score conservation violated in {path}: replayed scores={scores} "
            f"(sum={total}, expected <= {TOTAL_SCORE} with deficit a multiple of 1000)"
        )
    kyotaku_left = deficit // 1000
    if kyotaku_left > 0:
        # Game::commit() の未収供託精算と同一のタイブレーク:
        # 同点なら先頭 index (座席番号が若い方) を優先する。
        winner = max(range(4), key=lambda i: scores[i])
        scores[winner] += kyotaku_left * 1000

    if sum(scores) != TOTAL_SCORE:
        raise LogFormatError(f"post-settlement score sum != {TOTAL_SCORE} in {path}: {scores}")

    return scores


def rank_points(scores: list[int]) -> list[float]:
    """libriichi/src/rankings.rs Rankings::new と同一のタイブレーク
    (降順 stable sort、同点は座席番号が若い方が上位)で順位点を割り当てる。
    """
    order = sorted(range(4), key=lambda i: -scores[i])
    pts = [0.0] * 4
    for rank, pid in enumerate(order):
        pts[pid] = RANK_PTS[rank]
    return pts


def process_hanchan(path: Path) -> HanchanResult:
    events = load_events(path)

    names_ev = next((ev for ev in events if ev.get("type") == "start_game"), None)
    if names_ev is None:
        raise LogFormatError(f"no start_game event found in {path}")
    names = names_ev.get("names") or []
    if len(names) != 4:
        raise LogFormatError(f"start_game.names has unexpected length in {path}: {names}")

    challenger_seat = seat_from_filename(path)
    if names[challenger_seat] != "challenger":
        raise LogFormatError(
            f"filename-derived seat {challenger_seat} is not 'challenger' in {path}: names={names}"
        )

    n_kyoku = sum(1 for ev in events if ev.get("type") == "start_kyoku")
    if n_kyoku == 0:
        raise LogFormatError(f"no kyoku found in {path}")

    final_scores = reconstruct_final_scores(events, path)
    sotensu = (final_scores[challenger_seat] - RETURN_SCORE) / 1000.0
    rpts = rank_points(final_scores)[challenger_seat]

    chip_deltas = load_kyoku_chip_deltas_from_log(path, challenger_seat, n_kyoku)
    chip_total = float(chip_deltas.sum())

    combined = sotensu + rpts + chip_total * CHIP_VALUE

    return HanchanResult(
        path=path,
        n_kyoku=n_kyoku,
        sotensu=sotensu,
        rank_pts=rpts,
        chip_total=chip_total,
        combined=combined,
        chip_per_kyoku=chip_deltas,
    )


def mean_se(values) -> tuple[float, float, int]:
    arr = np.asarray(values, dtype=np.float64)
    n = arr.size
    if n < 2:
        raise LogFormatError(f"need >=2 samples to compute a standard error, got {n}")
    mean = float(arr.mean())
    se = float(arr.std(ddof=1) / np.sqrt(n))
    return mean, se, n


def build_report(results: list[HanchanResult]) -> list[str]:
    n_hanchan = len(results)
    n_kyoku_total = sum(r.n_kyoku for r in results)

    sotensu_mean, sotensu_se, _ = mean_se([r.sotensu for r in results])
    rank_pts_mean, rank_pts_se, _ = mean_se([r.rank_pts for r in results])
    chip_mean, chip_se, _ = mean_se([r.chip_total for r in results])
    combined_mean, combined_se, _ = mean_se([r.combined for r in results])

    chip_per_kyoku = np.concatenate([r.chip_per_kyoku for r in results])
    chip_per_kyoku_mean, chip_per_kyoku_se, _ = mean_se(chip_per_kyoku)

    lines = [
        f"n_hanchan={n_hanchan}",
        f"n_kyoku={n_kyoku_total}",
        f"sotensu_mean={sotensu_mean:.4f}",
        f"sotensu_se={sotensu_se:.4f}",
        f"rank_pts_mean={rank_pts_mean:.4f}",
        f"rank_pts_se={rank_pts_se:.4f}",
        f"chip_mean={chip_mean:.4f}",
        f"chip_se={chip_se:.4f}",
        f"combined_mean={combined_mean:.4f}",
        f"combined_se={combined_se:.4f}",
        f"chip_per_kyoku_mean={chip_per_kyoku_mean:.4f}",
        f"chip_per_kyoku_se={chip_per_kyoku_se:.4f}",
    ]
    return lines


def calibration_lines(results: list[HanchanResult], k_se: float = 2.0) -> tuple[list[str], bool]:
    """ミラー較正: 各ストリーム平均を理論ミラー値と SE 単位で比較 (backlog 5)。
    X vs 3X の対称対戦ログに適用する前提。|z|<=k_se なら PASS。戻り値は
    (出力行, 全ストリーム PASS か)。"""
    checks = (
        ("sotensu", [r.sotensu for r in results], MIRROR_SOTENSU),
        ("rank_pts", [r.rank_pts for r in results], MIRROR_RANK_PTS),
        ("chip", [r.chip_total for r in results], MIRROR_CHIP),
        ("combined", [r.combined for r in results], MIRROR_COMBINED),
    )
    lines = [
        f"# mirror calibration (theoretical: sotensu={MIRROR_SOTENSU:.1f} "
        f"rank_pts={MIRROR_RANK_PTS:.1f} chip={MIRROR_CHIP:.1f} "
        f"combined={MIRROR_COMBINED:.1f}); band=+/-{k_se:g}SE"
    ]
    all_ok = True
    for name, vals, theo in checks:
        mean, se, _ = mean_se(vals)
        dev = mean - theo
        z = dev / se if se > 0 else float("inf")
        ok = abs(z) <= k_se
        all_ok = all_ok and ok
        lines.append(f"mirror_{name}_mean={mean:.4f}")
        lines.append(f"mirror_{name}_theory={theo:.4f}")
        lines.append(f"mirror_{name}_se={se:.4f}")
        lines.append(f"mirror_{name}_dev={dev:.4f}")
        lines.append(f"mirror_{name}_z={z:.4f}")
        lines.append(f"mirror_{name}_within_band={'PASS' if ok else 'FAIL'}")
    lines.append(f"mirror_calibration_overall={'PASS' if all_ok else 'FAIL'}")
    return lines, all_ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "game_logs_dir",
        type=Path,
        help="OneVsThree game_logs ディレクトリ ({seed}_{key}_{split}.json.gz 群)",
    )
    ap.add_argument("-o", "--output", type=Path, default=None, help="key=value 出力の書き出し先")
    ap.add_argument(
        "--mirror-calibration",
        action="store_true",
        help="ミラー較正脚 (backlog 5): 各ストリーム平均を理論ミラー値と SE 単位で比較。"
        " X vs 3X の対称対戦ログに適用する",
    )
    ap.add_argument("--k-se", type=float, default=2.0, help="較正バンド幅 (SE 単位, default 2.0)")
    ap.add_argument(
        "--strict",
        action="store_true",
        help="--mirror-calibration で FAIL 時に exit 1 (default: 報告のみ)",
    )
    args = ap.parse_args()

    files = sorted(args.game_logs_dir.glob("*.json.gz"))
    if not files:
        raise LogFormatError(f"no json.gz logs found in {args.game_logs_dir}")

    results = [process_hanchan(path) for path in files]

    lines = build_report(results)
    calibration_ok = True
    if args.mirror_calibration:
        cal, calibration_ok = calibration_lines(results, k_se=args.k_se)
        lines += [""] + cal
    text = "\n".join(lines) + "\n"
    print(text, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        print(f"Wrote {args.output}", file=sys.stderr)

    if args.mirror_calibration and args.strict and not calibration_ok:
        raise SystemExit("FATAL: mirror calibration FAILED (see mirror_* lines above)")


if __name__ == "__main__":
    main()
