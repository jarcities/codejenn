#!/bin/bash
RUNS=30
BIT=32
CSV="stats.csv"
OPT_FLAGS="-O3 -ffast-math -fopenmp"
ARCH_FLAGS="-march=native"
TMP=$(mktemp -d)
NAMES=()
export OMP_NUM_THREADS=1

source "$(conda info --base)/etc/profile.d/conda.sh"

info() {
    printf '%s,"%s"\n' "$1" "$2" >> "$TMP/info"
}

version() {
    python3 -c "import $1; print($1.__version__)" 2>/dev/null
}

bench() {
    local name=$1
    shift
    NAMES+=("$name")
    echo ""
    echo "THIS IS $name INFERENCE"
    for ((i = 1; i <= RUNS; i++)); do
        t=$("$@" | awk '/seconds!/ {print $1}')
        if [ -z "$t" ]; then
            echo "__Error__ -> $name run $i did not report a time."
            exit 1
        fi
        echo "run $i -> $t seconds"
        echo "$t" >> "$TMP/$name"
    done
}

stat() {
    sort -n "$TMP/$2" | awk -v s="$1" '{a[NR] = $1; t += $1} END {
        m = t / NR
        for (i = 1; i <= NR; i++) v += (a[i] - m) ^ 2
        if (s == "mean") printf "%.6f", m
        if (s == "std") printf "%.6f", sqrt(v / (NR - 1))
        if (s == "median") printf "%.6f", (NR % 2) ? a[(NR + 1) / 2] : (a[NR / 2] + a[NR / 2 + 1]) / 2
    }'
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
    bench "keras_${backend}_eager" python3 -B test.py --backend="$backend" --bit=$BIT
    bench "keras_${backend}_jit" python3 -B test.py --backend="$backend" --bit=$BIT --jit
    g++-15 test.cpp -DBIT=$BIT $OPT_FLAGS $ARCH_FLAGS \
        -I$CONDA_PREFIX/include \
        -L$CONDA_PREFIX/lib \
        -Wl,-rpath,$CONDA_PREFIX/lib \
        -lhdf5_cpp -lhdf5 \
        -o a.out || exit 1
    bench "codejenn_${backend}" ./a.out
    info "$backend version" "$(version $backend)"
    info "keras version ($backend backend)" "$(version keras)"
}

info "c++ compiler" "$(g++-15 --version | head -1)"
info "optimization flags" "$OPT_FLAGS"
info "architecture flags" "$ARCH_FLAGS (resolves to -mcpu=$(g++-15 $ARCH_FLAGS -Q --help=target | awk '/-mcpu=/ {print $2}'))"

run_env keras-tf-3.12 tensorflow
run_env keras-py-3.14 torch

info "keras backend" "set per run with KERAS_BACKEND (tensorflow / torch)"
info "keras device" "cpu"
info "keras precision" "float$BIT"
info "keras eager" "model(x, training=False)"
info "keras jit (tensorflow)" "frozen graph, tf.function(jit_compile=True), concrete function"
info "keras jit (torch)" "torch.jit.trace, torch.jit.freeze, torch.jit.optimize_for_inference, torch.inference_mode"
info "OMP_NUM_THREADS" "$OMP_NUM_THREADS"
info "tensorflow threads (intra / inter)" "1 / 1"
info "torch threads" "1"
info "cpu frequency scaling" "not controlled"
info "other system processes" "not controlled"

FILES=()
HEADER="run"
for name in "${NAMES[@]}"; do
    FILES+=("$TMP/$name")
    HEADER="$HEADER,$name"
done
echo "$HEADER" > "$CSV"
paste -d, "${FILES[@]}" | awk '{print NR "," $0}' >> "$CSV"
for s in mean std median; do
    row=$s
    for name in "${NAMES[@]}"; do
        row="$row,$(stat "$s" "$name")"
    done
    echo "$row" >> "$CSV"
done
echo "" >> "$CSV"
echo "info,value" >> "$CSV"
cat "$TMP/info" >> "$CSV"
rm -rf "$TMP"

echo ""
sed -n -e 1p -e "$((RUNS + 2)),$((RUNS + 4))p" "$CSV" | column -s, -t
echo ""
echo "Saved \"$CSV\""
