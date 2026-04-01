import 'dart:typed_data';
import 'dart:ui';

import 'package:tflite_flutter/tflite_flutter.dart';

import 'label_constants.dart';

/// A single object detection result from EfficientDet-Lite0.
class Detection {
  /// COCO class index (0-indexed, background excluded).
  final int classId;

  /// Detection confidence score (0.0 – 1.0).
  final double confidence;

  /// Bounding box in normalized coordinates (0.0 – 1.0).
  /// Format: top, left, bottom, right as returned by the model.
  final Rect boundingBox;

  const Detection({
    required this.classId,
    required this.confidence,
    required this.boundingBox,
  });

  @override
  String toString() =>
      'Detection(classId: $classId, confidence: ${confidence.toStringAsFixed(2)}, '
      'box: $boundingBox)';
}

/// Manages the TFLite EfficientDet-Lite0 interpreter lifecycle.
///
/// Handles model loading, inference execution, output parsing, and disposal.
/// After [dispose] is called, all further [runInference] calls return empty.
class ModelHandler {
  static const String _modelAssetPath = 'ml/efficientdet_lite0.tflite';

  /// Maximum number of detections the model returns per frame.
  static const int _maxDetections = 25;

  /// Model input dimensions.
  static const int _inputSize = 320;

  Interpreter? _interpreter;
  bool _isDisposed = false;

  /// Whether the model has been loaded and is ready for inference.
  bool get isReady => _interpreter != null && !_isDisposed;

  /// Load the EfficientDet-Lite0 model from Flutter assets.
  ///
  /// Must be called before [runInference]. Allocates tensors and optionally
  /// logs input/output shapes for verification.
  Future<void> loadModel({bool logShapes = false}) async {
    if (_isDisposed) {
      throw StateError('Cannot load model after disposal.');
    }

    final options = InterpreterOptions()..threads = 2;

    _interpreter = await Interpreter.fromAsset(
      _modelAssetPath,
      options: options,
    );

    _interpreter!.allocateTensors();

    if (logShapes) {
      _logTensorShapes();
    }
  }

  /// Run inference on a preprocessed RGB buffer.
  ///
  /// [rgbBuffer] must be a flat `Uint8List` of length 320 * 320 * 3.
  /// Returns detections filtered by [confidenceThreshold].
  ///
  /// Returns empty list if model is disposed or not loaded.
  List<Detection> runInference(Uint8List rgbBuffer) {
    if (_isDisposed || _interpreter == null) return [];

    assert(
      rgbBuffer.length == _inputSize * _inputSize * 3,
      'Expected ${_inputSize * _inputSize * 3} bytes, got ${rgbBuffer.length}',
    );

    // Reshape flat buffer to [1, 320, 320, 3] input tensor.
    final input = rgbBuffer.reshape([1, _inputSize, _inputSize, 3]);

    // Prepare output buffers matching EfficientDet-Lite0 output tensors:
    //   0: bounding boxes [1, 25, 4] (float32) — top, left, bottom, right
    //   1: class indices  [1, 25]    (float32)
    //   2: scores         [1, 25]    (float32)
    //   3: detection count [1]       (float32)
    final outputBoxes = List.generate(
      1,
      (_) => List.generate(_maxDetections, (_) => List.filled(4, 0.0)),
    );
    final outputClasses = List.generate(
      1,
      (_) => List.filled(_maxDetections, 0.0),
    );
    final outputScores = List.generate(
      1,
      (_) => List.filled(_maxDetections, 0.0),
    );
    final outputCount = List.filled(1, 0.0);

    final outputs = <int, Object>{
      0: outputBoxes,
      1: outputClasses,
      2: outputScores,
      3: outputCount,
    };

    try {
      _interpreter!.runForMultipleInputs([input], outputs);
    } catch (e) {
      // Corrupted frame or tensor mismatch — skip, don't crash.
      return [];
    }

    return _parseDetections(
      outputBoxes[0],
      outputClasses[0],
      outputScores[0],
      outputCount[0].toInt(),
    );
  }

  /// Parse raw model output into filtered [Detection] objects.
  List<Detection> _parseDetections(
    List<List<double>> boxes,
    List<double> classes,
    List<double> scores,
    int count,
  ) {
    final detections = <Detection>[];
    final numDetections = count.clamp(0, _maxDetections);

    for (int i = 0; i < numDetections; i++) {
      final confidence = scores[i];
      if (confidence < confidenceThreshold) continue;

      detections.add(Detection(
        classId: classes[i].toInt(),
        confidence: confidence,
        boundingBox: Rect.fromLTRB(
          boxes[i][1], // left
          boxes[i][0], // top
          boxes[i][3], // right
          boxes[i][2], // bottom
        ),
      ));
    }

    return detections;
  }

  /// Fully dispose the interpreter and release all memory.
  ///
  /// After calling this, [runInference] returns empty and [loadModel] throws.
  void dispose() {
    _interpreter?.close();
    _interpreter = null;
    _isDisposed = true;
  }

  /// Print input/output tensor shapes for debugging during initial integration.
  void _logTensorShapes() {
    final interp = _interpreter!;
    print('[ModelHandler] Input tensor: ${interp.getInputTensor(0).shape}');
    for (int i = 0; i < interp.getOutputTensors().length; i++) {
      print('[ModelHandler] Output tensor $i: ${interp.getOutputTensor(i).shape}');
    }
  }
}
