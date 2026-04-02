# ML Layer Integration Guide — For Frontend Developer

This document explains how to wire the ML detection layer into the Flutter app.

---

## 1. Dependencies

Add to `pubspec.yaml`:

```yaml
dependencies:
  camera: ^0.11.0
  tflite_flutter: ^0.11.0

flutter:
  assets:
    - assets/ml/efficientdet_lite0.tflite
```

## 2. Model File

Run the download script to fetch the EfficientDet-Lite0 model (with built-in NMS):

```bash
bash scripts/download_model.sh
```

This places the `.tflite` file at `assets/ml/efficientdet_lite0.tflite` (~4.5MB).

**Important:** The model must include NMS post-processing (4 output tensors: boxes, classes, scores, count). The MediaPipe version without NMS will NOT work — it outputs raw anchors that the code cannot parse. The download script fetches the correct version.

## 3. Platform Configuration

### Android (`android/app/build.gradle`)

```groovy
android {
    defaultConfig {
        minSdkVersion 21
    }

    aaptOptions {
        noCompress 'tflite'
    }
}
```

### iOS (`ios/Podfile`)

```ruby
platform :ios, '15.5'
```

### Permissions

**Android** (`android/app/src/main/AndroidManifest.xml`):
```xml
<uses-permission android:name="android.permission.CAMERA" />
```

**iOS** (`ios/Runner/Info.plist`):
```xml
<key>NSCameraUsageDescription</key>
<string>Postura needs camera access to detect your posture setup.</string>
```

### Orientation Lock

Lock both platforms to portrait mode.

## 4. Wiring DetectionService

```dart
import 'package:postura/ml/detection_service.dart';
import 'package:postura/ml/detection_state.dart';

// 1. Create and initialize
final detectionService = DetectionService();
detectionService.onDetectionResult = _onDetectionResult;
await detectionService.initialize();  // loads model (~1-2s)

// 2. Start camera stream, throttle to 2fps
DateTime _lastFrame = DateTime.now();

void _onCameraFrame(CameraImage image) {
  final now = DateTime.now();
  if (now.difference(_lastFrame).inMilliseconds < 500) return; // 2fps
  _lastFrame = now;

  detectionService.processFrame(image);
}

// 3. Handle state changes
void _onDetectionResult(DetectionState state) {
  // Update UI indicator based on state:
  //   searching  → neutral indicator
  //   partial    → amber/grey indicator
  //   confirmed  → GREEN indicator

  if (state == DetectionState.confirmed) {
    // Stop camera stream
    _cameraController.stopImageStream();

    // Shut down ML layer completely
    detectionService.shutdown();

    // Optional: brief pause so GREEN is visible
    // await Future.delayed(Duration(milliseconds: 300));

    // Launch Step 2 (MediaPipe pose extraction)
  }
}
```

## 5. Interface Contract

| Method / Callback | Direction | Description |
|-------------------|-----------|-------------|
| `initialize()` | Frontend → ML | Load model. Call once before starting frames. |
| `processFrame(CameraImage)` | Frontend → ML | Call at 2fps with each camera frame. |
| `onDetectionResult` callback | ML → Frontend | Fires after each frame with `DetectionState`. |
| `shutdown()` | Frontend → ML | Call after `confirmed`. Disposes model, rejects further frames. |

## 6. DetectionState Enum

```dart
enum DetectionState {
  searching,   // No confirmation yet
  partial,     // One object detected, waiting for second
  confirmed,   // Both person + monitor confirmed (5/6 frames)
}
```

## 7. Important Notes

- **Step 1 and Step 2 must never run simultaneously.** Call `shutdown()` and dispose fully before initializing Step 2.
- **Model loading takes 1-2 seconds.** Show a loading indicator during `initialize()`.
- The ML layer handles all image format conversion internally — just pass the raw `CameraImage`.
- Set `initialize(logShapes: true)` on first run to verify model tensor shapes in the console. You should see:
  ```
  Input tensor: [1, 320, 320, 3]
  Output tensor 0: [1, 25, 4]    ← boxes
  Output tensor 1: [1, 25]       ← class IDs
  Output tensor 2: [1, 25]       ← scores
  Output tensor 3: [1]           ← detection count
  ```
  If you see `[1, 19206, 90]` instead, the wrong model is bundled (no NMS). Re-run `scripts/download_model.sh`.

## 8. What the ML Layer Detects

The model detects COCO objects. For Step 1, we check for:

| Object | Model class IDs | Notes |
|--------|----------------|-------|
| Person | 0 | COCO "person" |
| Monitor/screen | 71 (tv) or 72 (laptop) | COCO "tv" and "laptop" — the model classifies screens as either depending on appearance |

Both must be detected with confidence >= 0.5 in at least 5 out of 6 consecutive frames to reach `confirmed`.

## 9. Validation Results

Tested on 58 real images (2026-04-02):

| Image set | both_present | partial | searching |
|-----------|-------------|---------|-----------|
| 50 posture images (person + monitor) | 47 | 3 | 0 |
| 8 monitor-only (no person) | 0 | 7 | 1 |

The 3 partial results in posture images are extreme poses (-15 degree angles, person reaching far down) where one object falls below the 0.5 confidence threshold. The rolling window handles these gracefully — a brief dropout won't block confirmation.
