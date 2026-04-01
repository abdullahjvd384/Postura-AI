/// COCO class IDs used by EfficientDet-Lite0.
///
/// These must match the label map embedded in the .tflite model metadata.
/// Run [ModelHandler.loadModel] and inspect printed output tensor shapes
/// and detection labels to verify these values on first integration.

/// COCO class index for "person" (0-indexed, background class excluded).
const int cocoPersonClassId = 0;

/// COCO class index for "tv" (monitors, screens).
/// In the standard COCO 90-class label map this is index 62 (0-indexed).
/// Verify against your specific model variant's label map.
const int cocoTvClassId = 62;

/// Confidence threshold — detections below this are ignored.
const double confidenceThreshold = 0.5;

/// Set of all acceptable label strings for the monitor class.
/// Used only if parsing string labels instead of class IDs.
const Set<String> monitorLabels = {'tv', 'tvmonitor', 'monitor', 'screen'};
