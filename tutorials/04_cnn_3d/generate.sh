python3 \
    ../../src/api-core/main.py \
    --input="." \
    --output="." \
    --backend="tensorflow" \
    --bit=64 \
    --debug \
    # --model_image

rm -rf .vscode/ api-core/__pycache__ dump_model/__pycache__