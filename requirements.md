# Postura Step 1 — ML Backend Requirements

## Role

ML Backend Developer — owns all detection logic, model integration, inference, confidence filtering, rolling confirmation, and the interface contract with the Flutter frontend. No UI work.

---

## 1. Model

- **EfficientDet-Lite0** via TFLite Task Vision `ObjectDetector`
- Bundled in Flutter assets — no runtime download
- Load model on initialization before first frame arrives
- MobileNet is **not** suitable (classifier only, can't detect two objects simultaneously)

## 2. Detection Classes

Detect exactly two COCO classes:

| Class | Label to verify |
|-------|----------------|
| Person | `person` |
| Monitor | `tv` or `tvmonitor` — verify against the model's labelmap file |

> Check the exact label string in the EfficientDet-Lite labelmap before integrating. It varies by model variant.

## 3. Frame Input

- Frontend passes a `CameraImage` at **2 fps**
- ML layer receives the raw `CameraImage` and handles all byte conversion internally

### 3.1 Image Format Handling (Critical)

| Platform | Format | Row Reading Rule |
|----------|--------|-----------------|
| Android | YUV420 | Use `bytesPerRow` (stride), **NOT** `image.width` — buffer contains row padding |
| iOS | BGRA8888 | Standard pixel access — no stride issue |

> **Android bug warning:** Using `image.width` instead of `bytesPerRow` corrupts pixel data from row 2 onward. Always use `bytesPerRow`.

## 4. Inference & Confidence Filtering

- Run inference on every frame received from frontend
- Extract all detected objects with class label + confidence score
- **Confidence threshold: 0.5 (50%)** — ignore anything below
- Per-frame boolean result:
  - `true` — both `person` AND monitor detected ≥ 0.5 confidence
  - `false` — one or both missing or below threshold

## 5. Rolling Confirmation Window

| Parameter | Value |
|-----------|-------|
| Window size | 6 frames |
| Required positives | 5 out of 6 |
| Trigger | Fire `CONFIRMED` to frontend |

- A single `true` frame is not enough — prevents flicker
- Reset the window when frames consistently return `false` — do not carry stale positives across a detection dropout

## 6. Interface Contract

Expose these to the Flutter frontend:

| Method / Callback | Direction | Description |
|-------------------|-----------|-------------|
| `processFrame(CameraImage image)` | Frontend → ML | Called at 2fps with each camera frame |
| `onDetectionResult(DetectionState state)` | ML → Frontend | Callback with current state after each frame |
| `shutdown()` | Frontend → ML | Fully dispose model and release memory |

### DetectionState Enum

```
searching  — no confirmation yet
partial    — one object detected, waiting for second
confirmed  — both objects confirmed across rolling window
```

## 7. Model Loading & Disposal

- Load on init, before first frame
- On `shutdown()`: fully dispose interpreter, release all memory
- After `shutdown()`: reject any further frame processing
- Model loading takes 1–2s on first launch — coordinate with frontend for loading state

## 8. Acceptance Criteria

| Test | Expected Result |
|------|----------------|
| Person + monitor in frame | `confirmed` after 5/6 positive frames |
| Person only | Stays `searching` or `partial` — never `confirmed` |
| Empty room | Stays `searching` — never `confirmed` |
| Confidence below 0.5 | Detection ignored, doesn't count toward window |
| `shutdown()` called | Model fully disposed, no further processing |
| Android `bytesPerRow` | Pixel data reads correctly, no skewed output |
| 2–3 minutes sustained | No memory leak, no crash |

## 9. Platform Targets

| | iOS | Android |
|--|-----|---------|
| Min version | iOS 15.5 | API Level 21 |
| Orientation | Locked portrait | Locked portrait |
