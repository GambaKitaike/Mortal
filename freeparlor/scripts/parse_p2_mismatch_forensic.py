#!/usr/bin/env python3
"""Parse cursor-era mismatch lines from forensic client logs (by log file + timestamp)."""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path


def main():
    log_dir = Path(sys.argv[1] if len(sys.argv) > 1 else '.')
    rx = re.compile(
        r'^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ .*trajectory step count mismatch'
    )
    rows = []
    for path in sorted(log_dir.glob('client*.log')):
        client = path.stem  # client0
        for line in path.read_text(errors='replace').splitlines():
            m = rx.match(line)
            if m:
                rows.append((client, m.group('ts'), line.strip()))

    print(f'total={len(rows)}')
    print('by_client:', dict(Counter(r[0] for r in rows)))
    for client, ts, line in rows:
        print(f'{client}\t{ts}\t{line}')


if __name__ == '__main__':
    main()
