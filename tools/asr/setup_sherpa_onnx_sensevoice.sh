#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ASR_DIR="$ROOT_DIR/tools/asr"
BIN_DIR="$ASR_DIR/bin"
MODEL_ROOT="$ROOT_DIR/models/asr"
MODEL_DIR="$MODEL_ROOT/sensevoice-int8"
SHERPA_VERSION="${SHERPA_VERSION:-v1.13.0}"
SHERPA_ARCHIVE="sherpa-onnx-${SHERPA_VERSION}-linux-aarch64-shared-cpu.tar.bz2"
MODEL_ARCHIVE="sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2"
SHERPA_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/${SHERPA_VERSION}/${SHERPA_ARCHIVE}"
MODEL_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/${MODEL_ARCHIVE}"

mkdir -p "$BIN_DIR" "$MODEL_ROOT"

if [ ! -x "$BIN_DIR/sherpa-onnx-offline" ]; then
  tmp_dir="$(mktemp -d)"
  trap 'rm -rf "$tmp_dir"' EXIT
  curl -L --fail --continue-at - "$SHERPA_URL" -o "$tmp_dir/$SHERPA_ARCHIVE"
  tar -xjf "$tmp_dir/$SHERPA_ARCHIVE" -C "$tmp_dir"
  runtime_dir="$(find "$tmp_dir" -maxdepth 1 -type d -name 'sherpa-onnx-*linux-aarch64-shared-cpu' | head -n 1)"
  if [ -z "$runtime_dir" ]; then
    echo "sherpa-onnx runtime archive layout is not recognized" >&2
    exit 1
  fi
  cp "$runtime_dir/bin/sherpa-onnx-offline" "$BIN_DIR/sherpa-onnx-offline"
  find "$runtime_dir/lib" -maxdepth 1 -type f -name '*.so*' -exec cp -P {} "$BIN_DIR/" \;
  chmod +x "$BIN_DIR/sherpa-onnx-offline"
fi

if [ ! -f "$MODEL_DIR/model.int8.onnx" ] || [ ! -f "$MODEL_DIR/tokens.txt" ]; then
  tmp_dir="$(mktemp -d)"
  trap 'rm -rf "$tmp_dir"' EXIT
  curl -L --fail --continue-at - "$MODEL_URL" -o "$tmp_dir/$MODEL_ARCHIVE"
  tar -xjf "$tmp_dir/$MODEL_ARCHIVE" -C "$tmp_dir"
  extracted_dir="$(find "$tmp_dir" -maxdepth 1 -type d -name 'sherpa-onnx-sense-voice-*int8*' | head -n 1)"
  if [ -z "$extracted_dir" ]; then
    echo "SenseVoice model archive layout is not recognized" >&2
    exit 1
  fi
  rm -rf "$MODEL_DIR"
  mkdir -p "$MODEL_DIR"
  cp -R "$extracted_dir"/. "$MODEL_DIR/"
fi

echo "sherpa-onnx-offline: $BIN_DIR/sherpa-onnx-offline"
echo "sensevoice model: $MODEL_DIR/model.int8.onnx"
echo "sensevoice tokens: $MODEL_DIR/tokens.txt"
