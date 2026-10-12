#!/bin/bash
# ============================================================================
# Fish-Speech S2 推理提示脚本（实验态）
#
# 当前项目内旧版 Fish-Speech 1.5 runner 已与 vendored S2 仓库脱节：
#   - 旧脚本依赖 tools.llama.generate / tools.vqgan.inference
#   - 当前仓库实际推荐走 API Server / WebUI / 三段式 CLI
#
# 因此这里不再假装“可直接跑通”，而是给出官方推荐入口。
# ============================================================================

set -eo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CODE_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
FISH_CODE="${CODE_ROOT}/utils/audio/fish-speech"

cat <<EOF
⚠️  Fish-Speech 当前未纳入统一 benchmark。

原因：
  1. 当前 vendored 仓库是 S2 版本
  2. 旧版 1.5 Python runner 已与目录结构和接口不兼容
  3. 官方当前更推荐 API Server / WebUI / 三段式 CLI

建议用法：

  # 方式 1：启动官方 API Server
  cd ${FISH_CODE}
  python tools/api_server.py --listen 0.0.0.0:8080

  # 方式 2：调用官方 API Client
  cd ${FISH_CODE}
  python tools/api_client.py -u http://127.0.0.1:8080/v1/tts -t "你好，欢迎使用 Fish Speech。"

  # 方式 3：按官方三段式 CLI 推理
  cd ${FISH_CODE}
  python fish_speech/models/dac/inference.py -i test.wav --checkpoint-path checkpoints/s2-pro/codec.pth
  python fish_speech/models/text2semantic/inference.py --text "你好" --prompt-text "参考文本" --prompt-tokens fake.npy
  python fish_speech/models/dac/inference.py -i codes_0.npy

如果后续要把 Fish-Speech 真正并进 benchmark，建议单独重写一个 S2 runner。
EOF
