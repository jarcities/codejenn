#!/bin/bash
RUNS=30
BIT=32
OPT_FLAGS="-O3 -ffast-math -fopenmp"
ARCH_FLAGS="-march=native"
STATS=()
INFO=()
export OMP_NUM_THREADS=1

source "$(conda info --base)/etc/profile.d/conda.sh"

info() {
    INFO+=("$1")
}

version() {
    python3 -c "import $1; print($1.__version__)" 2>/dev/null
}

bench() {
    local name=$1
    local times=()
    shift
    echo ""
    echo "conducting $name inference"
    printf '*%.0s' {1..50}
    echo ""
    for ((i = 1; i <= RUNS; i++)); do
        t=$("$@" | awk '/seconds!/ {print $1}')
        if [ -z "$t" ]; then
            echo "__Error__ -> $name run $i did not report a time."
            exit 1
        fi
        echo "run $i -> $t seconds"
        times+=("$t")
    done
    STATS+=("$(printf '%s\n' "${times[@]}" | sort -n | awk -v name="$name" '{a[NR] = $1; t += $1} END {
        m = t / NR
        for (i = 1; i <= NR; i++) v += (a[i] - m) ^ 2
        printf "%s -> mean %.6f, std %.6f, median %.6f", name, m, sqrt(v / (NR - 1)), (NR % 2) ? a[(NR + 1) / 2] : (a[NR / 2] + a[NR / 2 + 1]) / 2
    }')")
}

run_env() {
    local env=$1
    local backend=$2
    conda activate "$env" || exit 1
    export KERAS_BACKEND=$backend
    python3 -B train.py --backend="$backend" --bit=$BIT || exit 1
    python3 -B \
        ../../../src/api-core/main.py \
        --input="." \
        --output="." \
        --backend="$backend" \
        --bit=$BIT || exit 1
    rm -rf .vscode/ ../../../src/api-core/__pycache__ ../../../src/dump_model/__pycache__
    python3 -B data.py || exit 1
    bench "${backend} eager" python3 -B test.py --backend="$backend" --bit=$BIT
    bench "${backend} jit" python3 -B test.py --backend="$backend" --bit=$BIT --jit
    g++-15 test.cpp -DBIT=$BIT $OPT_FLAGS $ARCH_FLAGS \
        -I$CONDA_PREFIX/include \
        -L$CONDA_PREFIX/lib \
        -Wl,-rpath,$CONDA_PREFIX/lib \
        -lhdf5_cpp -lhdf5 \
        -o a.out || exit 1
    bench "codejenn (${backend})" ./a.out
    info "python + ${backend/torch/pytorch} version -> $(python3 -c 'import platform; print(platform.python_version())'), $(version $backend)"
}

info "c++ compiler -> $(g++-15 --version | head -1)"
info "opt flags -> $OPT_FLAGS"
info "arch flags -> $ARCH_FLAGS"

run_env keras-tf-3.12 tensorflow
run_env keras-py-3.14 torch

info "keras version -> $(version keras)"
info "keras device -> cpu"
info "precision -> float$BIT"
info "keras jit (tensorflow) -> frozen graph, tf.function(jit_compile=True), concrete function"
info "keras jit (torch) -> torch.jit.trace, torch.jit.freeze, torch.jit.optimize_for_inference, torch.inference_mode"
info "OMP_NUM_THREADS -> $OMP_NUM_THREADS"
info "tensorflow threads -> 1"
info "torch threads -> 1"

echo ""
echo "final stats ($RUNS runs)"
printf '*%.0s' {1..50}
echo ""
printf '%s\n' "${STATS[@]}" | awk -F' -> ' '{k[NR] = $1; v[NR] = substr($0, length($1) + 5); if (length($1) > w) w = length($1)} END {
    for (i = 1; i <= NR; i++) printf "%-" w "s -> %s\n", k[i], v[i]
}'
echo ""
echo "machine info"
printf '*%.0s' {1..50}
echo ""
printf '%s\n' "${INFO[@]}" | awk -F' -> ' '{k[NR] = $1; v[NR] = substr($0, length($1) + 5); if (length($1) > w) w = length($1)} END {
    for (i = 1; i <= NR; i++) printf "%-" w "s -> %s\n", k[i], v[i]
}'
