# Postura Step 1 — ML Backend Implementation Plan

## Context

Postura is a posture monitoring app. Step 1 is the entry gate: detect both a **person** and a **monitor** in the camera frame before pose analysis begins. The project is currently empty — no Flutter project, no code, no model files. This plan covers the ML backend layer only (lib/ml/, tests, model download script, integration doc). The Flutter project scaffold is owned by the frontend developer.

---

## Architecture

```
Frontend (camera stream @ 2fps)
    │
    │ CameraImage
    ▼
DetectionService                        ← public API
    ├── FramePreprocessor               ← CameraImage → 320x320 RGB bytes
    │     (YUV420 on Android, BGRA8888 on iOS)
    ├── ModelHandler                    ← TFLite interpreter lifecycle + inference
    │     (EfficientDet-Lite0, confidence ≥ 0.5)
    └── RollingWindow                  ← 5/6 frames → confirmed
    │
    ▼
onDetectionResult(DetectionState) → callback to frontend
```

## File Structure

```
lib/
  ml/
    detection_state.dart            — DetectionState enum
    label_constants.dart            — COCO class IDs for person + tv
    rolling_window.dart             — 6-frame rolling confirmation
    frame_preprocessor.dart         — CameraImage → Uint8List RGB (platform-aware)
    model_handler.dart              — TFLite load, infer, dispose
    detection_service.dart          — Orchestrator: public interface contract

assets/
  ml/
    efficientdet_lite0.tflite       — Bundled model (downloaded via script)

scripts/
  download_model.sh                 — Downloads EfficientDet-Lite0 from TF Hub

test/
  ml/
    rolling_window_test.dart
    frame_preprocessor_test.dart
    detection_service_test.dart

INTEGRATION.md                      — Frontend developer wiring instructions
```

## Packages

| Package | Version | Purpose |
|---------|---------|---------|
| `camera` | ^0.11.0 | Camera access + CameraImage stream |
| `tflite_flutter` | ^0.11.0 | TFLite interpreter (direct, not ML Kit) |

**Not using:** `google_mlkit_object_detection` (less control), `tflite_flutter_helper` (deprecated), old `tflite` package (abandoned).

## Model

- **EfficientDet-Lite0** from TF Hub / Kaggle (`tensorflow/efficientdet/tfLite/lite0-detection-metadata`)
- Input: `[1, 320, 320, 3]` RGB uint8
- Outputs: boxes `[1, 25, 4]`, classes `[1, 25]`, scores `[1, 25]`, count `[1]`
- COCO labels — `person` (index 0), `tv` (index ~71, must verify from model labelmap)

---

## Implementation Phases

> **Note:** The Flutter project scaffold (flutter create, pubspec.yaml, platform configs, permissions) is owned by the **frontend developer**. This plan only covers ML-layer files under `lib/ml/`, tests under `test/ml/`, the model download script, and integration instructions for the frontend dev.

### Phase 0: Model Setup

1. **`scripts/download_model.sh`** — Shell script that downloads EfficientDet-Lite0 `.tflite` from TF Hub and places it at `assets/ml/efficientdet_lite0.tflite`
2. **`assets/ml/`** — Create directory, add `.gitkeep` so it's tracked (model file itself is gitignored)
3. **`INTEGRATION.md`** — Instructions for the frontend developer:
   - Required pubspec.yaml dependencies: `camera: ^0.11.0`, `tflite_flutter: ^0.11.0`
   - Asset declaration: `assets/ml/efficientdet_lite0.tflite`
   - Android: `minSdkVersion 21`, `aaptOptions { noCompress 'tflite' }`
   - iOS: platform `ios, '15.5'`
   - How to wire `DetectionService` into the camera stream

### Phase 1: Pure Logic Layer (no TFLite, no camera — unit-testable)

4. **`detection_state.dart`** — Enum: `searching`, `partial`, `confirmed`
5. **`label_constants.dart`** — COCO class IDs for `person` and `tv` (verify after downloading model)
6. **`rolling_window.dart`** — Core confirmation logic:
    - Maintains FIFO queue of max 6 booleans
    - `addResult(bool)` — appends, trims to 6
    - `currentState` getter → `confirmed` if ≥5 true, `partial` if ≥1 true, `searching` if 0 true
    - Reset window on 4+ consecutive false frames (prevents stale positives)
    - Lock after `confirmed` (stop accepting results)
7. **`rolling_window_test.dart`** — Tests: all-true, 5/6, 4/6, all-false, reset on sustained false, lock after confirmed

### Phase 2: Frame Preprocessing

8. **`frame_preprocessor.dart`** — `Uint8List preprocessCameraImage(CameraImage image)`:
    - **Android YUV420**: Read Y from `planes[0]` using `bytesPerRow` stride (NOT `image.width`). U/V from `planes[1]`/`planes[2]` at half resolution using their respective `bytesPerRow`. Convert YUV→RGB with standard formula.
    - **iOS BGRA8888**: Single plane, 4 bytes/pixel (B,G,R,A), extract RGB.
    - Resize to 320×320 via nearest-neighbor interpolation.
    - Output: `Uint8List` of length 307,200 (320×320×3).
9. **`frame_preprocessor_test.dart`** — Synthetic CameraImage with deliberate row padding, verify correct RGB output and 320×320 size.

### Phase 3: TFLite Model Handler

10. **`model_handler.dart`** — `ModelHandler` class:
    - `loadModel()`: `Interpreter.fromAsset('ml/efficientdet_lite0.tflite')`, allocate tensors. Print input/output shapes on first load for verification.
    - `runInference(Uint8List rgbBuffer)`: Reshape to `[1,320,320,3]`, prepare output maps, run, parse into `List<Detection>` (classId, confidence, boundingBox), filter confidence ≥ 0.5.
    - `dispose()`: Close interpreter, null out, set disposed flag.
    - `Detection` data class: `int classId`, `double confidence`, `Rect boundingBox`
11. **Verify model I/O shapes on device** — Print tensor shapes, confirm label map class IDs match constants.

### Phase 4: Detection Service (Orchestrator)

12. **`detection_service.dart`** — `DetectionService` class:
    - `initialize()` → calls `modelHandler.loadModel()`
    - `processFrame(CameraImage)` → preprocess → infer → check both person+monitor present → update rolling window → fire `onDetectionResult` callback
    - `shutdown()` → set shutdown flag, dispose model, reject further frames
    - `onDetectionResult` callback: `typedef DetectionCallback = void Function(DetectionState)`
13. **`detection_service_test.dart`** — Mock ModelHandler, test full flow: person+monitor→confirmed, person-only→partial, empty→searching, shutdown rejects frames.

### Phase 5: Integration Instructions

14. **`INTEGRATION.md`** — Document for frontend developer covering:
    - How to import and initialize `DetectionService`
    - How to wire `processFrame` into the camera stream at 2fps
    - How to listen to `onDetectionResult` callback
    - How to call `shutdown()` on confirmed
    - Required pubspec dependencies and platform config

### Phase 6: Hardening

15. **Error handling**: try-catch around `runInference` (corrupted frame → skip, don't crash)
16. **Memory**: Ensure buffers aren't retained across frames. Profile with DevTools over 3 min.
17. **Isolate** (only if needed): If UI jank is observed, move preprocess+inference to background isolate. Note: `Interpreter` can't cross isolates — must load model inside isolate.

---

## Verification

| Test | Method | Expected |
|------|--------|----------|
| Rolling window logic | Unit tests (`rolling_window_test.dart`) | All state transitions correct |
| Frame preprocessing | Unit tests with synthetic data | Correct RGB, stride handling, 320×320 output |
| Detection service flow | Unit tests with mocked model | Correct state callbacks |
| Model loads on device | Run on Android/iOS device | No crash, shapes printed correctly |
| Person + monitor visible | On-device, point at person + screen | `confirmed` within ~3 seconds |
| Person only | On-device, person only | Stays `searching`/`partial` |
| Empty room | On-device, empty frame | Stays `searching` |
| Shutdown | Trigger confirmed, verify | Model disposed, no further processing |
| Memory stability | Run 3 minutes, check DevTools | No leak, no crash |
| Android pixel correctness | On Android device | No skewed/corrupted inference results |

---

## Key Risks

| Risk | Mitigation |
|------|-----------|
| Wrong COCO class ID for monitor | Print all detections with labels during initial testing; verify labelmap first |
| Output tensor shape differs | Print shapes on load (step 15), adjust buffers |
| Android YUV stride bug | Unit test with padded synthetic data; visual inspection of preprocessed output |
| `tflite_flutter` version issues | Pin version; fallback to platform channels if blocked |
