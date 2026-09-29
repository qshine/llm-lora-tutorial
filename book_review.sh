#!/usr/bin/env bash
# Start the local ebook from any working directory.
set -euo pipefail

BOOK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$BOOK_ROOT"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  printf '用法：%s [端口，默认 8000]\n按 Ctrl+C 停止本地预览。\n' "$0"
  exit 0
fi
BOOK_PORT="${1:-8000}"
if [[ $# -gt 1 || ! "$BOOK_PORT" =~ ^[1-9][0-9]{0,4}$ ]] || ((10#$BOOK_PORT > 65535)); then
  printf '端口必须是 1～65535 的整数，例如：./book_review.sh 8001\n' >&2
  exit 2
fi

BOOK_PYTHON="$BOOK_ROOT/.venv-book/bin/python"
if [[ ! -x "$BOOK_PYTHON" ]]; then
  if ! command -v python3.11 >/dev/null 2>&1; then
    printf '请先安装 Python 3.11，再运行此脚本。\n' >&2
    exit 1
  fi
  printf '首次启动：创建独立的电子书环境……\n'
  python3.11 -m venv "$BOOK_ROOT/.venv-book"
fi

# Reuse the environment offline when its locked dependencies are installed.
if ! "$BOOK_PYTHON" - <<'PY'
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import sys

for line in Path('requirements-book-lock.txt').read_text().splitlines():
    if not line.strip() or line.startswith('#'):
        continue
    package, expected = line.split('==', 1)
    try:
        if version(package) != expected:
            sys.exit(1)
    except PackageNotFoundError:
        sys.exit(1)
PY
then
  printf '安装电子书依赖（首次安装或版本变化时需要联网）……\n'
  "$BOOK_PYTHON" -m ensurepip --upgrade
  "$BOOK_PYTHON" -m pip install -r requirements-book-lock.txt
fi

# A publishing URL must not change the local preview's path.
unset BOOK_SITE_URL
printf '\n本地阅读地址：http://127.0.0.1:%s/\n服务就绪后自动打开默认浏览器。\n按 Ctrl+C 停止。端口被占用时可运行：./book_review.sh 8001\n\n' "$BOOK_PORT"
exec "$BOOK_PYTHON" -m mkdocs serve --dev-addr "127.0.0.1:$BOOK_PORT" --open
