#!/usr/bin/env python3
"""Parse first-P2 mismatch warnings from client logs (old cursor-based client)."""

from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


def main():
    log_dir = Path(sys.argv[1] if len(sys.argv) > 1 else '/home/gamba/mahjong/runs/ppo/smoke_p2_forensic/logs')
    rx = re.compile(
        r'trajectory step count mismatch \((?P<got>\d+)/(?P<exp>\d+)\), '
        r'skipping game client=(?P<client>\S+) file=(?P<file>\S+)'
    )
    rows = []
    for path in sorted(log_dir.glob('client*.log')):
        for i, line in enumerate(path.read_text(errors='replace').splitlines(), 1):
            m = rx.search(line)
            if m:
                rows.append({
                    'log': path.name,
                    'line': i,
                    **m.groupdict(),
                })

    print(f'total={len(rows)}')
    by_client = Counter(r['client'] for r in rows)
    print('by_client:', dict(by_client))
    for r in rows:
        print(
            f"{r['client']}\tstep {r['got']}/{r['exp']}\t"
            f"{Path(r['file']).name}\t{r['log']}:{r['line']}"
        )


if __name__ == '__main__':
    main()
