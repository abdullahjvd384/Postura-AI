import 'dart:typed_data';
import 'dart:ui';

import 'package:flutter_test/flutter_test.dart';
import 'package:postura/ml/detection_service.dart';
import 'package:postura/ml/detection_state.dart';
import 'package:postura/ml/label_constants.dart';
import 'package:postura/ml/model_handler.dart';
import 'package:postura/ml/rolling_window.dart';

// ---------------------------------------------------------------------------
// Mock ModelHandler that returns controlled detection lists.
// ---------------------------------------------------------------------------

class MockModelHandler extends ModelHandler {
  List<Detection> nextDetections = [];

  @override
  Future<void> loadModel({bool logShapes = false}) async {
    // No-op for testing.
  }

  @override
  List<Detection> runInference(Uint8List rgbBuffer) {
    return nextDetections;
  }

  bool disposed = false;

  @override
  void dispose() {
    disposed = true;
    super.dispose();
  }
}

// ---------------------------------------------------------------------------
// Helper detections.
// ---------------------------------------------------------------------------

Detection _person({double confidence = 0.9}) => Detection(
      classId: cocoPersonClassId,
      confidence: confidence,
      boundingBox: const Rect.fromLTRB(0.1, 0.1, 0.5, 0.8),
    );

Detection _tv({double confidence = 0.8}) => Detection(
      classId: cocoTvClassId,
      confidence: confidence,
      boundingBox: const Rect.fromLTRB(0.5, 0.1, 0.9, 0.6),
    );

// ---------------------------------------------------------------------------
// Fake CameraImage (minimal, enough for DetectionService to call
// preprocessCameraImage which will be tested separately).
// We bypass the actual preprocessor by having MockModelHandler ignore input.
// ---------------------------------------------------------------------------

// Since DetectionService calls preprocessCameraImage internally and we can't
// easily mock it, we provide a minimal YUV420 CameraImage that won't crash
// the preprocessor.
import 'package:camera/camera.dart';

CameraImage _fakeCameraImage() {
  const width = 320;
  const height = 320;

  final yBytes = Uint8List(width * height);
  final uvSize = ((width + 1) ~/ 2) * ((height + 1) ~/ 2);
  final uBytes = Uint8List(uvSize);
  final vBytes = Uint8List(uvSize);

  return _FakeCameraImage(
    width: width,
    height: height,
    format: _FakeImageFormat(ImageFormatGroup.yuv420),
    planes: [
      _FakePlane(bytes: yBytes, bytesPerRow: width, bytesPerPixel: 1),
      _FakePlane(bytes: uBytes, bytesPerRow: (width + 1) ~/ 2, bytesPerPixel: 1),
      _FakePlane(bytes: vBytes, bytesPerRow: (width + 1) ~/ 2, bytesPerPixel: 1),
    ],
  );
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

void main() {
  late MockModelHandler mockModel;
  late DetectionService service;
  late List<DetectionState> stateLog;

  setUp(() {
    mockModel = MockModelHandler();
    service = DetectionService(modelHandler: mockModel);
    stateLog = [];
    service.onDetectionResult = (state) => stateLog.add(state);
  });

  group('DetectionService', () {
    test('person + monitor → confirmed after 5/6 frames', () async {
      await service.initialize();
      mockModel.nextDetections = [_person(), _tv()];

      final image = _fakeCameraImage();
      for (int i = 0; i < 6; i++) {
        service.processFrame(image);
      }

      // Should reach confirmed by frame 5 or 6.
      expect(stateLog.last, DetectionState.confirmed);
    });

    test('person only → stays partial, never confirmed', () async {
      await service.initialize();
      mockModel.nextDetections = [_person()];

      final image = _fakeCameraImage();
      for (int i = 0; i < 10; i++) {
        service.processFrame(image);
      }

      expect(stateLog, isNot(contains(DetectionState.confirmed)));
      expect(stateLog.last, DetectionState.partial);
    });

    test('empty detections → stays searching', () async {
      await service.initialize();
      mockModel.nextDetections = [];

      final image = _fakeCameraImage();
      for (int i = 0; i < 6; i++) {
        service.processFrame(image);
      }

      expect(stateLog.last, DetectionState.searching);
    });

    test('shutdown rejects further frames', () async {
      await service.initialize();
      mockModel.nextDetections = [_person(), _tv()];

      final image = _fakeCameraImage();
      // Process a few frames.
      service.processFrame(image);
      service.processFrame(image);

      service.shutdown();
      expect(service.isShutdown, true);

      // Clear log and try more frames.
      stateLog.clear();
      service.processFrame(image);
      service.processFrame(image);

      // No new states should have been emitted.
      expect(stateLog, isEmpty);
    });

    test('shutdown disposes model handler', () async {
      await service.initialize();
      service.shutdown();
      expect(mockModel.disposed, true);
    });

    test('callback fires on every frame', () async {
      await service.initialize();
      mockModel.nextDetections = [_person()];

      final image = _fakeCameraImage();
      service.processFrame(image);
      service.processFrame(image);
      service.processFrame(image);

      expect(stateLog.length, 3);
    });

    test('mixed detections produce correct state sequence', () async {
      await service.initialize();
      final image = _fakeCameraImage();

      // Frame 1-3: both present
      mockModel.nextDetections = [_person(), _tv()];
      service.processFrame(image);
      service.processFrame(image);
      service.processFrame(image);

      // Frame 4: only person
      mockModel.nextDetections = [_person()];
      service.processFrame(image);

      // Frame 5-6: both present again
      mockModel.nextDetections = [_person(), _tv()];
      service.processFrame(image);
      service.processFrame(image);

      // 5 out of 6 true → confirmed
      expect(stateLog.last, DetectionState.confirmed);
    });
  });
}

// ---------------------------------------------------------------------------
// Minimal fakes for CameraImage (same pattern as frame_preprocessor_test).
// ---------------------------------------------------------------------------

class _FakeCameraImage implements CameraImage {
  @override
  final int width;
  @override
  final int height;
  @override
  final ImageFormat format;
  @override
  final List<Plane> planes;

  _FakeCameraImage({
    required this.width,
    required this.height,
    required this.format,
    required this.planes,
  });

  @override
  dynamic noSuchMethod(Invocation invocation) => super.noSuchMethod(invocation);
}

class _FakeImageFormat implements ImageFormat {
  @override
  final ImageFormatGroup group;
  _FakeImageFormat(this.group);
  @override
  dynamic noSuchMethod(Invocation invocation) => super.noSuchMethod(invocation);
}

class _FakePlane implements Plane {
  @override
  final Uint8List bytes;
  @override
  final int bytesPerRow;
  @override
  final int? bytesPerPixel;

  _FakePlane({
    required this.bytes,
    required this.bytesPerRow,
    this.bytesPerPixel,
  });

  @override
  dynamic noSuchMethod(Invocation invocation) => super.noSuchMethod(invocation);
}
