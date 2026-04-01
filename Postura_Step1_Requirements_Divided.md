**POSTURA --- Step 1 Technical Requirements**

*Frame Validation: Person + Monitor Detection*

Metaviz AI \| Version 1.0 \| April 2026

**Overview --- What Step 1 Does**

Step 1 is the entry gate for the entire Postura pipeline. Before any pose analysis happens, the app must confirm that both a person and a monitor are visible in the camera frame at the same time. Only then does the GREEN indicator fire and Step 2 (MediaPipe pose extraction) begins.

Step 1 runs at 2 frames per second to stay lightweight. Once GREEN is confirmed, Step 1 shuts down completely before Step 2 starts. Both never run simultaneously.

| **Who builds what**        | **Developer Role**                  | **Core Responsibility**                                                                         |
|----------------------------|-------------------------------------|-------------------------------------------------------------------------------------------------|
| Flutter Frontend Developer | UI + Camera Feed + State            | Camera preview, GREEN indicator UI, state management between detection states                   |
| ML Backend Developer       | Model Integration + Detection Logic | EfficientDet-Lite integration, object detection, confidence filtering, frame confirmation logic |

> *⚠ Both developers must coordinate on the interface contract --- specifically how the ML layer communicates detection results back to the Flutter UI layer.*

**PART A --- Flutter Frontend Developer**

**A1. Responsibility Summary**

The Flutter frontend developer owns everything the user sees and interacts with during Step 1. This includes the live camera preview, the detection state indicator, and triggering the handoff to Step 2 when GREEN is confirmed. The frontend developer does not write any ML or model loading code.

**A2. Camera Feed Setup**

**A2.1 Package**

Use the official Flutter camera package. Initialize the camera on app launch and display a live preview filling the screen in portrait orientation.

**A2.2 Frame Extraction for ML**

The camera stream must feed frames to the ML layer at 2 frames per second. Do not pass every frame. Use a timer or frame throttle to sample at 2fps and pass the CameraImage object to the ML detection function.

**A2.3 Image Format by Platform**

| **Platform** | **Camera Image Format** | **Action Required**                                                    |
|--------------|-------------------------|------------------------------------------------------------------------|
| Android      | YUV420                  | Pass raw CameraImage to ML layer --- ML developer handles byte reading |
| iOS          | BGRA8888                | Pass raw CameraImage to ML layer --- ML developer handles byte reading |

> *⚠ The frontend developer does not process image bytes directly. Just pass the CameraImage object to the ML layer at 2fps.*

**A3. Detection States and UI**

The frontend must handle three visual states. These states are driven by results returned from the ML layer.

| **State Name**     | **Trigger Condition**                                         | **UI Behaviour**                                                            |
|--------------------|---------------------------------------------------------------|-----------------------------------------------------------------------------|
| SEARCHING          | App is running, no confirmation yet                           | Neutral indicator --- no green, show searching state                        |
| PARTIAL (optional) | One object detected, waiting for second                       | Amber or grey indicator --- useful UX feedback but not mandatory for Step 1 |
| CONFIRMED GREEN    | ML layer reports both objects confirmed across rolling window | GREEN indicator fires --- visible, clear, full width bar or badge           |

> *⚠ Do not trigger GREEN from the frontend directly. Only switch to GREEN when the ML layer explicitly returns a confirmed result.*

**A4. Handoff to Step 2**

When the ML layer returns a CONFIRMED GREEN state, the frontend must do the following in order:

- Stop the 2fps frame sampling loop

- Call the ML layer shutdown method to fully dispose EfficientDet-Lite

- Pause briefly (optional 300ms) so the GREEN state is visible to the user

- Initialize and launch Step 2 (MediaPipe pose extraction)

> *⚠ Step 1 and Step 2 must never run at the same time. Dispose Step 1 completely before initializing Step 2.*

**A5. Platform Requirements**

| **Requirement**            | **iOS**                             | **Android**                       |
|----------------------------|-------------------------------------|-----------------------------------|
| Minimum version            | iOS 15.5                            | API Level 21                      |
| Camera permission handling | Info.plist NSCameraUsageDescription | AndroidManifest CAMERA permission |
| Portrait orientation       | Locked portrait                     | Locked portrait                   |
| Delivery format            | Simulator + real device             | APK on real device                |

**A6. Frontend Acceptance Criteria**

| **Test**                                   | **Expected Result**                            |
|--------------------------------------------|------------------------------------------------|
| Camera opens on launch                     | Live preview visible immediately               |
| GREEN fires when ML confirms both objects  | Clear green indicator appears within 3 seconds |
| No GREEN when ML returns partial detection | Indicator stays neutral or amber               |
| GREEN does not flicker                     | Stable --- no rapid on/off switching           |
| App stable after 2-3 minutes               | No crash, no memory leak                       |
| Step 2 launches after GREEN                | Smooth transition, no overlap                  |

**PART B --- ML Backend Developer**

**B1. Responsibility Summary**

The ML backend developer owns everything related to the detection model --- loading EfficientDet-Lite, processing camera frames, running inference, applying confidence filtering, managing the rolling confirmation window, and returning a clean result to the Flutter frontend. This developer does not build any UI.

**B2. Model Selection --- EfficientDet-Lite**

Use EfficientDet-Lite0 via the TFLite Task Vision ObjectDetector. This is the correct model for this step because it detects and localizes multiple objects in the same frame simultaneously, which is required to confirm both person and monitor are present at the same time.

| **Model**          | **Why NOT used**                                                                                |          |
|--------------------|-------------------------------------------------------------------------------------------------|----------|
| MobileNet          | Image classifier --- identifies dominant object only, cannot confirm two objects simultaneously |          |
| EfficientDet-Lite0 | Object detector --- confirms person AND monitor in same frame. Correct choice.                  | USE THIS |

**B2.1 Detection Classes**

Configure the model to detect exactly two classes:

- person

- tv_monitor (check exact label in the model\'s labelmap file)

> *⚠ Verify the label string for monitor in the EfficientDet-Lite labelmap before integrating. It may be tv_monitor, monitor, or screen depending on the model variant used.*

**B3. Frame Processing**

**B3.1 Input from Frontend**

The frontend passes a CameraImage object at 2fps. The ML layer receives this and is responsible for converting it to the format required by EfficientDet-Lite.

**B3.2 Image Format Handling --- Critical**

This is the most common source of bugs in Flutter camera ML pipelines. Handle both platforms correctly:

| **Platform** | **Format** | **Row Reading Rule**                         | **Why**                                                                                                                                                                      |
|--------------|------------|----------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Android      | YUV420     | Use bytesPerRow (stride) --- NOT image.width | Buffer contains row padding. image.width is the pixel count but bytesPerRow is the actual memory stride. Using image.width causes misaligned reads and corrupted pixel data. |
| iOS          | BGRA8888   | Standard pixel access                        | No stride padding issue on iOS                                                                                                                                               |

> *⚠ If you use image.width on Android instead of bytesPerRow, your brightness values and pixel reads will be wrong from row 2 onward. This error accumulates and corrupts all downstream calculations. Always use bytesPerRow.*

**B4. Inference and Confidence Filtering**

**B4.1 Running Inference**

Run EfficientDet-Lite inference on each frame received from the frontend. Extract all detected objects with their class label and confidence score.

**B4.2 Confidence Threshold**

Ignore any detection with a confidence score below 0.5 (50%). Only detections at or above this threshold count toward confirmation. This filters out weak or partial detections.

**B4.3 Per-Frame Result**

After filtering, determine for each frame whether both classes are present above threshold. Produce a simple per-frame boolean result:

- true --- both person AND monitor detected above 0.5 confidence

- false --- one or both missing or below threshold

**B5. Rolling Confirmation Window**

A single true frame is not enough to trigger GREEN. Use a rolling window of the last 6 frames. GREEN fires only when 5 out of 6 consecutive frames return true. This prevents flicker from a single good frame among noise.

| **Window size** | **Required positives** | **Trigger condition**            |
|-----------------|------------------------|----------------------------------|
| 6 frames        | 5 out of 6             | Fire CONFIRMED event to frontend |

> *⚠ Reset the window if frames consistently return false --- do not carry stale positives across a gap where detection dropped out entirely.*

**B6. Interface Contract with Frontend**

The ML layer must expose a clean interface to the frontend. The frontend calls these methods and the ML layer handles everything internally:

| **Method / Callback**                   | **Direction** | **Description**                                            |
|-----------------------------------------|---------------|------------------------------------------------------------|
| processFrame(CameraImage image)         | Frontend → ML | Frontend calls this at 2fps with each camera frame         |
| onDetectionResult(DetectionState state) | ML → Frontend | ML calls this callback with current state after each frame |
| shutdown()                              | Frontend → ML | Frontend calls this after GREEN to fully dispose the model |

DetectionState should be a simple enum with three values: searching, partial, confirmed.

**B7. Model Loading and Disposal**

- Load EfficientDet-Lite model on initialization, before the first frame arrives

- Model file must be bundled in Flutter assets --- do not download at runtime

- On shutdown() call, fully dispose the interpreter and release all memory

- After shutdown(), the ML layer must not process any further frames

> *⚠ Model loading can take 1-2 seconds on first launch. Consider showing a loading state in the UI during this window. Coordinate with frontend developer.*

**B8. ML Acceptance Criteria**

| **Test**                  | **Expected Result**                                     |
|---------------------------|---------------------------------------------------------|
| Person + monitor in frame | CONFIRMED returned after 5/6 positive frames            |
| Person only               | State stays at searching or partial --- never confirmed |
| Empty room                | State stays at searching --- never confirmed            |
| Confidence below 0.5      | Detection ignored --- does not count toward window      |
| shutdown() called         | Model fully disposed --- no further processing          |
| Android bytesPerRow       | Pixel data reads correctly --- no skewed output         |
| 2-3 minutes sustained     | No memory leak, no crash                                |

*Postura Step 1 Requirements \| Metaviz AI \| Confidential*
