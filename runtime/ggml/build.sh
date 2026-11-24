#!/bin/sh
set -eu
src=$1
build=$2
backend=$3
case "$backend" in cpu) vk=OFF;; vulkan) vk=ON;; *) exit 2;; esac
test "$(git -C "$src" rev-parse HEAD)" = 74d4f5b041ad837153b0e90fc864b8290e01d8d5
cmake -S "$src" -B "$build" -DGGML_VULKAN="$vk" -DGGML_METAL=OFF -DGGML_CUDA=OFF -DBUILD_SHARED_LIBS=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build "$build" --target llama -j4
lib=$(cd "$build/bin" && pwd)
c++ -O2 -std=c++17 "$(dirname "$0")/runner.cpp" -I"$src/include" -I"$src/ggml/include" -L"$lib" -Wl,-rpath,"$lib" -lllama -lggml -lggml-base -o "$lib/burda-ggml"
