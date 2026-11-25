#!/usr/bin/env bash
# MAST/NASA reachability needs a SOCKS proxy on this machine, spelled
# socks5:// (ALL_PROXY=socks:// is rejected by urllib3). Override or disable:
#   EXO_PROXY=socks5://host:port ./run.sh ...   different proxy
#   EXO_NO_PROXY=1 ./run.sh ...                 no proxy at all
if [ -z "${EXO_NO_PROXY:-}" ]; then
  _p="${EXO_PROXY:-socks5://127.0.0.1:7890}"
  export ALL_PROXY="$_p"
  export SOCKS_PROXY="$_p"
  export HTTPS_PROXY="socks5h${_p#socks5}"
  export HTTP_PROXY="socks5h${_p#socks5}"
fi
exec "$(dirname "$0")/.venv/Scripts/python.exe" "$@"
