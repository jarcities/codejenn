#!/bin/bash
RUNS=30
CSV="stats.csv"
FLAGS="-O3 -march=native -ffast-math -fopenmp"
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
    python3 -B train.py --backend="$backend" || exit 1
    python3 -B \
        ../../../src/api-core/main.py \
        --input="." \
        --output="." \
        --backend="$backend" \
        --bit=64 || exit 1
    rm -rf .vscode/ ../../../src/api-core/__pycache__ ../../../src/dump_model/__pycache__
    python3 -B data.py || exit 1
    bench "keras_${backend}_eager" python3 -B test.py --backend="$backend"
    bench "keras_${backend}_jit" python3 -B test.py --backend="$backend" --jit
    g++-15 test.cpp $FLAGS \
        -I$CONDA_PREFIX/include \
        -L$CONDA_PREFIX/lib \
        -Wl,-rpath,$CONDA_PREFIX/lib \
        -lhdf5_cpp -lhdf5 \
        -o a.out || exit 1
    bench "codejenn_${backend}" ./a.out
    info "$env python" "$(python3 -c 'import platform; print(platform.python_version())')"
    info "$env keras" "$(version keras)"
    info "$env $backend" "$(version $backend)"
    info "$env numpy" "$(version numpy)"
    info "$env h5py" "$(version h5py)"
    info "$env hdf5" "$(conda list | awk '$1 == "hdf5" {print $2}')"
    info "$env llvm-openmp" "$(conda list | awk '$1 == "llvm-openmp" {print $2}')"
}

info "date at start" "$(date)"
info "machine" "$(sysctl -n hw.model)"
info "chip" "$(sysctl -n machdep.cpu.brand_string)"
info "cores (total / performance / efficiency)" "$(sysctl -n hw.ncpu) / $(sysctl -n hw.perflevel0.logicalcpu) / $(sysctl -n hw.perflevel1.logicalcpu)"
info "memory (GB)" "$(($(sysctl -n hw.memsize) / 1073741824))"
info "os" "$(sw_vers -productName) $(sw_vers -productVersion) ($(sw_vers -buildVersion))"
info "compiler" "$(g++-15 --version | head -1)"
info "compiler flags" "$FLAGS"
info "c++ standard (__cplusplus)" "$(g++-15 -dM -E -x c++ /dev/null | awk '/__cplusplus/ {print $3}')"
info "-march=native resolves to" "-mcpu=$(g++-15 $FLAGS -Q --help=target | awk '/-mcpu=/ {print $2}')"
info "OMP_NUM_THREADS" "$OMP_NUM_THREADS"
info "tensorflow threads (intra / inter)" "1 / 1"
info "torch threads" "1"
info "keras device" "cpu"
info "keras precision" "float64"
info "keras jit (tensorflow)" "tf.function(jit_compile=True)"
info "keras jit (torch)" "torch.jit.trace"
info "codejenn precision" "double (--bit=64)"
info "runs per case" "$RUNS"
info "inferences per run" "10000"
info "power source" "$(pmset -g batt | head -1)"
info "power mode" "$(pmset -g | awk '/powermode/ {print $2}')"
info "thermal state at start" "$(pmset -g therm | paste -sd' ' -)"
info "load average at start" "$(sysctl -n vm.loadavg)"

run_env keras-tf-3.12 tensorflow
run_env keras-py-3.14 torch

info "thermal state at end" "$(pmset -g therm | paste -sd' ' -)"
info "load average at end" "$(sysctl -n vm.loadavg)"
info "date at end" "$(date)"

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
