/// COCO class IDs used by EfficientDet-Lite0.
///
/// These must match the label map embedded in the .tflite model metadata.
/// Run [ModelHandler.loadModel] and inspect printed output tensor shapes
/// and detection labels to verify these values on first integration.

/// COCO class index for "person" (0-indexed, background class excluded).
const int cocoPersonClassId = 0;

/// COCO class index for "tv" (monitors, screens).
/// Model uses COCO category ID - 1:  tv = 72 - 1 = 71.
const int cocoTvClassId = 71;

/// COCO class index for "laptop".
/// Model uses COCO category ID - 1:  laptop = 73 - 1 = 72.
/// The model often classifies desktop monitors as "laptop", so we accept both.
const int cocoLaptopClassId = 72;

/// All class IDs that count as a "screen/monitor" detection.
const Set<int> monitorClassIds = {cocoTvClassId, cocoLaptopClassId};

/// Confidence threshold — detections below this are ignored.
const double confidenceThreshold = 0.5;
