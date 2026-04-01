# Postura — ML Backend Developer Guide

## Project

Postura is a posture monitoring app. Step 1 is the entry gate: confirm both a **person** and a **monitor** are visible in the camera frame before any pose analysis (Step 2) begins.

## Your Scope

You are the **ML Backend Developer**. You own:
- EfficientDet-Lite0 model integration (TFLite Task Vision ObjectDetector)
- Camera frame processing and byte-level image conversion
- Inference execution and confidence filtering (threshold: 0.5)
- Rolling confirmation window (5/6 frames)
- Clean interface contract with Flutter frontend (`processFrame`, `onDetectionResult`, `shutdown`)

You do **not** build any UI, camera setup, or state management.

## Key Technical Decisions

- **Model:** EfficientDet-Lite0 — object detector that can find multiple objects per frame. Not MobileNet (classifier only).
- **Frame rate:** 2 fps — frontend throttles, you receive pre-throttled frames.
- **Confirmation logic:** Rolling window of 6 frames, GREEN fires at 5+ true frames. Reset window on sustained detection dropout.
- **Step 1 and Step 2 never run simultaneously.** After `shutdown()`, reject all further frames.

## Critical Bug to Avoid

On **Android (YUV420)**, always use `bytesPerRow` (stride) when reading pixel rows — **never** `image.width`. The buffer has row padding; using width corrupts pixels from row 2 onward.

## Coordination with Frontend Developer

- Frontend calls `processFrame(CameraImage)` at 2fps
- You call back `onDetectionResult(DetectionState)` after each frame
- Frontend calls `shutdown()` after confirmed — you dispose the model completely
- Model loading takes 1–2s; frontend may show a loading state during this window
- Agree on the `DetectionState` enum: `searching`, `partial`, `confirmed`

## Requirements Doc

See [requirements.md](requirements.md) for full acceptance criteria, interface contract, and platform details.
