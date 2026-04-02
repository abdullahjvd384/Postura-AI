# Postura — ML Backend (Step 1)

Object detection layer for Postura's entry gate: confirms both a **person** and a **monitor/screen** are visible in the camera frame before pose analysis (Step 2) begins.

## How It Works

```
Camera (2fps) → preprocessCameraImage → EfficientDet-Lite0 → Rolling Window → DetectionState
                (YUV420/BGRA → RGB)     (on-device TFLite)    (5/6 frames)    (callback)
```

The model detects COCO objects on-device. If both a person (class 0) and a screen (class 71 or 72) are found with confidence >= 0.5 in at least 5 out of 6 consecutive frames, the state transitions to `confirmed`.

## Quick Start

### 1. Prerequisites

- Flutter SDK >= 3.16.0
- Dart SDK >= 3.2.0

### 2. Clone and install dependencies

```bash
git clone <repo-url>
cd Postura
flutter pub get
```

### 3. Download the model

```bash
bash scripts/download_model.sh
```

This places `efficientdet_lite0.tflite` (~4.5MB, with built-in NMS) at `assets/ml/`.

### 4. Run tests

```bash
flutter test
```

## Project Structure

```
lib/
└── ml/
    ├── detection_service.dart    # Public API — processFrame(), onDetectionResult, shutdown()
    ├── detection_state.dart      # Enum: searching, partial, confirmed
    ├── model_handler.dart        # TFLite interpreter lifecycle and inference
    ├── frame_preprocessor.dart   # CameraImage → 320x320 RGB conversion
    ├── rolling_window.dart       # 5/6 frame confirmation logic
    └── label_constants.dart      # COCO class IDs and confidence threshold

test/ml/
    ├── detection_service_test.dart
    ├── frame_preprocessor_test.dart
    └── rolling_window_test.dart

assets/ml/
    └── efficientdet_lite0.tflite # EfficientDet-Lite0 with NMS (download via script)

scripts/
    └── download_model.sh         # Fetches the correct model file
```

## Frontend Integration

See [INTEGRATION.md](INTEGRATION.md) for the full wiring guide. The short version:

```dart
import 'package:postura/ml/detection_service.dart';
import 'package:postura/ml/detection_state.dart';

final service = DetectionService();
service.onDetectionResult = (state) {
  // searching → neutral, partial → amber, confirmed → green
  if (state == DetectionState.confirmed) {
    service.shutdown();
    // → launch Step 2
  }
};
await service.initialize();

// In camera callback (throttled to 2fps):
service.processFrame(cameraImage);
```

## Platform Setup

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

Add camera permission in `AndroidManifest.xml`:
```xml
<uses-permission android:name="android.permission.CAMERA" />
```

### iOS (`ios/Podfile`)

```ruby
platform :ios, '15.5'
```

Add to `Info.plist`:
```xml
<key>NSCameraUsageDescription</key>
<string>Postura needs camera access to detect your posture setup.</string>
```

## Model Details

| Property | Value |
|----------|-------|
| Model | EfficientDet-Lite0 (int8 quantized, with NMS) |
| Input | `[1, 320, 320, 3]` uint8 RGB |
| Outputs | 4 tensors: boxes `[1,25,4]`, classes `[1,25]`, scores `[1,25]`, count `[1]` |
| Size | ~4.5 MB |
| Max detections | 25 per frame |
| Target classes | person (0), tv (71), laptop (72) |
| Confidence threshold | 0.5 |

## Verification

On first run, call `initialize(logShapes: true)` and check the console output:

```
Input tensor: [1, 320, 320, 3]
Output tensor 0: [1, 25, 4]
Output tensor 1: [1, 25]
Output tensor 2: [1, 25]
Output tensor 3: [1]
```

If you see `[1, 19206, 90]` instead, the wrong model is bundled (missing NMS). Re-run `scripts/download_model.sh`.

## Key Design Decisions

- **On-device inference** — no server, no internet required, camera frames never leave the phone
- **2fps throttle** — frontend controls frame rate, ML layer processes whatever it receives
- **Rolling window (5/6)** — prevents false positives from single-frame noise
- **Accepts both tv + laptop classes** — COCO model classifies screens as either depending on appearance
- **Android YUV420 stride handling** — uses `bytesPerRow` (not `image.width`) to avoid row-padding corruption
