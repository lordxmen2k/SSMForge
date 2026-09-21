#!/usr/bin/env bash
# Test runner for SSMForge — handles PYTHONPATH and venv
set -e
cd /workspace/SSMForge
export PYTHONPATH=/workspace/SSMForge/src
exec .venv/bin/pytest "$@"
