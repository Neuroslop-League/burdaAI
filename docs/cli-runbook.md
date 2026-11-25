# C07.7. Запуск и диагностика CLI

## Чистая сборка CPU

Требуются Git, стабильный Rust/Cargo и системный компилятор C/C++. Для GGUF дополнительно требуются CMake и закреплённая ревизия llama.cpp. На macOS используется Apple Clang из Command Line Tools. В примере BUNDLE — путь к комплекту reference-f32, GGUF — файл Q4_0. Получение синтетических весов описано в [C07.5](cpu-inference.md) и [C07.6](weak-pc.md). Веса передаются отдельно от исходного кода.

```sh
git clone https://github.com/Neuroslop-League/burdaAI.git burda-clean
cd burda-clean
work=$(mktemp -d)
export CARGO_HOME="$work/cargo"
cargo build --release --locked --manifest-path rust/Cargo.toml --bin burda-generate
cli="$PWD/rust/target/release/burda-generate"
"$cli" --help
"$cli" --bundle "$BUNDLE" --device cpu --placement cpu --text 'Привет, мир!' --max-new-tokens 4

python3.12 -m venv "$work/venv"
. "$work/venv/bin/activate"
python -m pip install cmake==3.31.6 psutil==7.1.0
git clone --depth 1 --branch b5046 https://github.com/ggml-org/llama.cpp.git "$work/llama"
runtime/ggml/build.sh "$work/llama" "$work/cpu" cpu
"$cli" --gguf "$GGUF" --tokenizer artifacts/c02-v1 --runner "$work/cpu/bin/burda-ggml" \
  --device cpu --placement cpu --text 'Привет, мир!' --max-new-tokens 4 2>"$work/cpu.log"
grep -E 'model buffer size|prompt=|compute buffer size' "$work/cpu.log"
```

При ошибке сборки сохранить вывод `rustc -Vv`, `cargo -V`, `c++ --version`, `cmake --version`, `git rev-parse HEAD` и `git -C "$work/llama" rev-parse HEAD`. Последняя команда должна вернуть 74d4f5b041ad837153b0e90fc864b8290e01d8d5. Если GGUF-процесс не стартует, проверить путь к runner и его динамические библиотеки (`ldd "$work/cpu/bin/burda-ggml"` на Linux, `otool -L "$work/cpu/bin/burda-ggml"` на macOS). При несовместимости токенизатора сверить SHA-256 tokenizer.json с манифестом C02. Для reference-f32 ошибка хеша указывает конкретный файл комплекта.

## C07.6. Vulkan на дискретной карте

Выбранный профиль — Linux, одна дискретная AMD amdgpu с 1 ГиБ выделенной VRAM, Vulkan ICD, Vulkan headers, glslc и vulkan-tools. Выполнять из того же клона и окружения. Вывести доступные карты и выбрать соответствующий cardN:

```sh
uname -a
lscpu
free -h
vulkaninfo --summary
glslc --version
for device in /sys/class/drm/card[0-9]*/device; do
  if test -r "$device/mem_info_vram_total"; then
    echo "$device -> $(readlink -f "$device")"
    cat "$device/mem_info_vram_total" "$device/mem_info_vram_used"
  fi
done
```

Ожидаемое значение total — 1073741824 байта. Сборка размещает бинарный файл и библиотеки в отдельном каталоге. Для предварительной проверки выполнить один шаг, затем запустить сценарий с выключенной ранней остановкой:

```sh
runtime/ggml/build.sh "$work/llama" "$work/vk" vulkan
runner="$work/vk/bin/burda-ggml"
"$cli" --gguf "$GGUF" --tokenizer artifacts/c02-v1 --runner "$runner" \
  --device vulkan --placement last:8 --text 'Привет' --max-new-tokens 1 2>"$work/vk-smoke.log"
grep '^completed_gpu_matmul=' "$work/vk-smoke.log"

drm=/sys/class/drm/card0  # заменить на выбранный cardN
python -m scripts.check_c07_1gib "$runner" "$GGUF" --drm "$drm" \
  --log "$work/vk-256x64.log" >"$work/vk-256x64.json"
```

Передать C06 оба файла vk-256x64 и сведения об оборудовании. Для успешной приёмки требуется нулевой код выхода, строка completed_gpu_matmul с именем операции и устройства, 256 входных ID и 64 сгенерированных. JSON содержит SHA-256 модели, исходную и пиковую занятую VRAM всей карты (опрос каждые 20 мс), пиковый RSS процесса по ядру Linux, холодный и тёплый TTFT в секундах, скорость в токенах/с после первого токена. При повторе выбрать новый путь лога. При requested GPU unavailable проверить ICD и что runner взят из каталога vk. При ошибке выделения памяти сохранить исходную занятость карты и лог размещения. Смена числа слоёв означает другой профиль испытания.

## Результат проверки из свежего клона

Проверены исходники main на a21b02e после слияния PR #18: новый клон, пустой CARGO_HOME, новый target, отдельный Python venv, новый checkout llama.cpp и новый каталог сборки GGML. Среда — macOS 26.7.1, Apple M1, системная инструментальная цепочка, CMake 3.31.6. Входные синтетические FP32/Q4_0 веса взяты из C07.5–6. Из запроса «Привет, мир!» обе CPU-команды выдали по четыре токена. Вывод reference-f32 — ` Fort отличfooddate`, Q4_0 — ` осужд Equource творческой`. Запрос Vulkan к CPU-сборке завершился ошибкой requested GPU unavailable. На этом хосте проверка оборудования сообщает об отсутствии счётчиков amdgpu. Сценарий 256+64 на целевой дискретной карте остаётся открытым.

## Приёмка C07.7 на Mac

После описанной выше проверки чистой сборки прогон выполняется через Rust CLI с тем же синтетическим Q4_0, CPU backend, размещением cpu и четырьмя потоками GGML. Текст из 255 повторений строки ` a` токенизируется общим C02 в 255 обычных ID. CLI добавляет BOS. Флаг --benchmark-256x64 проверяет длины, отключает досрочную остановку по EOS и проверяет число возвращённых ID перед декодированием.

```sh
prompt=$(python3 -c 'print(" a" * 255, end="")')
/usr/bin/time -l "$cli" --gguf "$GGUF" --tokenizer artifacts/c02-v1 \
  --runner "$work/cpu/bin/burda-ggml" --device cpu --placement cpu \
  --text "$prompt" --max-new-tokens 64 --benchmark-256x64 \
  >mac-256x64.txt 2>mac-256x64.log
grep -E '^prompt=|maximum resident set size' mac-256x64.log
```

TTFT измеряет backend от своего старта (холодный) и от готовности контекста (тёплый) до первого вычисленного ID. Скорость рассчитана по следующим 63 ID. CLI печатает декодированный текст по завершении генерации.
