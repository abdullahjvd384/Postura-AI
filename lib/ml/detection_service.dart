import 'package:camera/camera.dart';

import 'detection_state.dart';
import 'frame_preprocessor.dart';
import 'label_constants.dart';
import 'model_handler.dart';
import 'rolling_window.dart';

/// Callback signature for detection state updates.
typedef DetectionCallback = void Function(DetectionState state);

/// Step 1 detection service — the public interface for the Flutter frontend.
///
/// Orchestrates frame preprocessing, EfficientDet-Lite0 inference, and the
/// rolling confirmation window. The frontend calls [processFrame] at 2fps and
/// receives state updates via [onDetectionResult].
///
/// ## Usage
/// ```dart
/// final service = DetectionService();
/// service.onDetectionResult = (state) {
///   if (state == DetectionState.confirmed) {
///     service.shutdown();
///     // → launch Step 2
///   }
/// };
/// await service.initialize();
/// // Then call service.processFrame(cameraImage) at 2fps.
/// ```
class DetectionService {
  final ModelHandler _modelHandler;
  final RollingWindow _rollingWindow;
  bool _isShutdown = false;

  /// Callback invoked after each frame with the current [DetectionState].
  DetectionCallback? onDetectionResult;

  /// Creates a new detection service.
  ///
  /// Accepts optional [modelHandler] and [rollingWindow] for testing.
  DetectionService({
    ModelHandler? modelHandler,
    RollingWindow? rollingWindow,
  })  : _modelHandler = modelHandler ?? ModelHandler(),
        _rollingWindow = rollingWindow ?? RollingWindow();

  /// Whether the service has been shut down.
  bool get isShutdown => _isShutdown;

  /// Load the EfficientDet-Lite0 model. Must be called before [processFrame].
  ///
  /// Set [logShapes] to `true` on first integration to verify tensor shapes.
  Future<void> initialize({bool logShapes = false}) async {
    await _modelHandler.loadModel(logShapes: logShapes);
  }

  /// Process a single camera frame.
  ///
  /// Called by the frontend at 2fps. Preprocesses the image, runs inference,
  /// updates the rolling window, and fires [onDetectionResult].
  ///
  /// No-op if the service has been shut down.
  void processFrame(CameraImage image) {
    if (_isShutdown) return;

    // 1. Convert CameraImage → 320x320 RGB buffer.
    final rgbBuffer = preprocessCameraImage(image);

    // 2. Run EfficientDet-Lite0 inference.
    final detections = _modelHandler.runInference(rgbBuffer);

    // 3. Check if both person AND monitor are present above threshold.
    final bothPresent = _checkBothPresent(detections);

    // 4. Update rolling confirmation window.
    _rollingWindow.addResult(bothPresent);

    // 5. Report current state.
    final state = _rollingWindow.currentState;
    onDetectionResult?.call(state);
  }

  /// Check whether both a person and a monitor/tv were detected.
  bool _checkBothPresent(List<Detection> detections) {
    bool hasPerson = false;
    bool hasMonitor = false;

    for (final d in detections) {
      if (d.classId == cocoPersonClassId) hasPerson = true;
      if (d.classId == cocoTvClassId) hasMonitor = true;
    }

    return hasPerson && hasMonitor;
  }

  /// Shut down the detection service.
  ///
  /// Disposes the TFLite model and rejects all further [processFrame] calls.
  /// Call this after receiving [DetectionState.confirmed].
  void shutdown() {
    _isShutdown = true;
    _modelHandler.dispose();
  }
}
