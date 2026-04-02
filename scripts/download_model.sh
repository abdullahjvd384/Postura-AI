#!/bin/bash
# Downloads EfficientDet-Lite0 TFLite model with COCO labels
# Places it at assets/ml/efficientdet_lite0.tflite

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$PROJECT_ROOT/assets/ml"
OUTPUT_FILE="$OUTPUT_DIR/efficientdet_lite0.tflite"
TEMP_DIR=$(mktemp -d)

# EfficientDet-Lite0 with built-in NMS post-processing (~4.5MB).
# This version outputs 4 tensors (boxes, classes, scores, count) ready for use.
# The MediaPipe version (without NMS) outputs raw anchors and requires manual
# post-processing — do NOT use that one.
MODEL_URL="https://github.com/schu-lab/Tensorflow-Object-Detection/raw/main/efficientdet_lite0.tflite"

echo "Downloading EfficientDet-Lite0..."
mkdir -p "$OUTPUT_DIR"

if command -v curl &> /dev/null; then
    curl -L -o "$OUTPUT_FILE" "$MODEL_URL"
elif command -v wget &> /dev/null; then
    wget -O "$OUTPUT_FILE" "$MODEL_URL"
else
    echo "Error: neither curl nor wget found. Install one and retry."
    exit 1
fi

FILE_SIZE=$(wc -c < "$OUTPUT_FILE" | tr -d ' ')
echo "Downloaded: $OUTPUT_FILE ($FILE_SIZE bytes)"

if [ "$FILE_SIZE" -lt 1000000 ]; then
    echo "Warning: File seems too small. Expected ~4.4MB for EfficientDet-Lite0."
    exit 1
fi

echo "Done. Model ready at assets/ml/efficientdet_lite0.tflite"

rm -rf "$TEMP_DIR"
