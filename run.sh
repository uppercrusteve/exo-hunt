#!/usr/bin/env bash
# MAST/NASA reachability needs the local SOCKS proxy, spelled socks5://
# (ALL_PROXY=socks:// is rejected by urllib3).
export ALL_PROXY=socks5://127.0.0.1:7890
export SOCKS_PROXY=socks5://127.0.0.1:7890
export HTTPS_PROXY=socks5h://127.0.0.1:7890
export HTTP_PROXY=socks5h://127.0.0.1:7890
exec "$(dirname "$0")/.venv/Scripts/python.exe" "$@"
