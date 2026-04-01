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

Run the download script to fetch the EfficientDet-Lite0 model:

```bash
bash scripts/download_model.sh
```

This places the `.tflite` file at `assets/ml/efficientdet_lite0.tflite`.

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
- Set `initialize(logShapes: true)` on first run to verify model tensor shapes in the console.
