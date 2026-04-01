import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:postura/ml/frame_preprocessor.dart';

// ---------------------------------------------------------------------------
// Helpers to build synthetic CameraImage objects for testing.
// ---------------------------------------------------------------------------

/// Creates a fake YUV420 CameraImage with known pixel values and optional
/// row padding (bytesPerRow > width) to simulate the Android stride issue.
CameraImage createFakeYuv420({
  required int width,
  required int height,
  int paddingPerRow = 0,
  required int yValue,
  required int uValue,
  required int vValue,
}) {
  final int yRowStride = width + paddingPerRow;
  final int uvWidth = (width + 1) ~/ 2;
  final int uvHeight = (height + 1) ~/ 2;
  final int uvRowStride = uvWidth + paddingPerRow;

  // Build Y plane with padding.
  final yBytes = Uint8List(yRowStride * height);
  for (int row = 0; row < height; row++) {
    for (int col = 0; col < width; col++) {
      yBytes[row * yRowStride + col] = yValue;
    }
    // Padding bytes are left as 0 (garbage).
  }

  // Build U and V planes (half resolution) with padding.
  final uBytes = Uint8List(uvRowStride * uvHeight);
  final vBytes = Uint8List(uvRowStride * uvHeight);
  for (int row = 0; row < uvHeight; row++) {
    for (int col = 0; col < uvWidth; col++) {
      uBytes[row * uvRowStride + col] = uValue;
      vBytes[row * uvRowStride + col] = vValue;
    }
  }

  return _FakeCameraImage(
    width: width,
    height: height,
    format: _FakeImageFormat(ImageFormatGroup.yuv420),
    planes: [
      _FakePlane(bytes: yBytes, bytesPerRow: yRowStride, bytesPerPixel: 1),
      _FakePlane(bytes: uBytes, bytesPerRow: uvRowStride, bytesPerPixel: 1),
      _FakePlane(bytes: vBytes, bytesPerRow: uvRowStride, bytesPerPixel: 1),
    ],
  );
}

/// Creates a fake BGRA8888 CameraImage with known pixel values.
CameraImage createFakeBgra8888({
  required int width,
  required int height,
  required int rValue,
  required int gValue,
  required int bValue,
}) {
  final int rowStride = width * 4;
  final bytes = Uint8List(rowStride * height);
  for (int row = 0; row < height; row++) {
    for (int col = 0; col < width; col++) {
      final offset = row * rowStride + col * 4;
      bytes[offset] = bValue; // B
      bytes[offset + 1] = gValue; // G
      bytes[offset + 2] = rValue; // R
      bytes[offset + 3] = 255; // A
    }
  }

  return _FakeCameraImage(
    width: width,
    height: height,
    format: _FakeImageFormat(ImageFormatGroup.bgra8888),
    planes: [
      _FakePlane(bytes: bytes, bytesPerRow: rowStride, bytesPerPixel: 4),
    ],
  );
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

void main() {
  group('FramePreprocessor', () {
    test('output buffer is exactly 320*320*3 bytes (YUV420)', () {
      final image = createFakeYuv420(
        width: 640,
        height: 480,
        yValue: 128,
        uValue: 128,
        vValue: 128,
      );
      final result = preprocessCameraImage(image);
      expect(result.length, 320 * 320 * 3);
    });

    test('output buffer is exactly 320*320*3 bytes (BGRA8888)', () {
      final image = createFakeBgra8888(
        width: 640,
        height: 480,
        rValue: 128,
        gValue: 128,
        bValue: 128,
      );
      final result = preprocessCameraImage(image);
      expect(result.length, 320 * 320 * 3);
    });

    test('pure white YUV converts to near-white RGB', () {
      // Y=255, U=128, V=128 → R≈255, G≈255, B≈255
      final image = createFakeYuv420(
        width: 320,
        height: 320,
        yValue: 255,
        uValue: 128,
        vValue: 128,
      );
      final result = preprocessCameraImage(image);

      // Check first pixel.
      expect(result[0], closeTo(255, 2)); // R
      expect(result[1], closeTo(255, 2)); // G
      expect(result[2], closeTo(255, 2)); // B
    });

    test('pure black YUV converts to near-black RGB', () {
      // Y=0, U=128, V=128 → R≈0, G≈0, B≈0
      final image = createFakeYuv420(
        width: 320,
        height: 320,
        yValue: 0,
        uValue: 128,
        vValue: 128,
      );
      final result = preprocessCameraImage(image);

      expect(result[0], closeTo(0, 2)); // R
      expect(result[1], closeTo(0, 2)); // G
      expect(result[2], closeTo(0, 2)); // B
    });

    test('YUV with row padding produces correct output', () {
      // This is THE critical Android test.
      // With 16 bytes of padding per row, using image.width instead of
      // bytesPerRow would read garbage from row 2 onward.
      final withPadding = createFakeYuv420(
        width: 640,
        height: 480,
        paddingPerRow: 16,
        yValue: 200,
        uValue: 128,
        vValue: 128,
      );
      final withoutPadding = createFakeYuv420(
        width: 640,
        height: 480,
        paddingPerRow: 0,
        yValue: 200,
        uValue: 128,
        vValue: 128,
      );

      final resultPadded = preprocessCameraImage(withPadding);
      final resultNoPadding = preprocessCameraImage(withoutPadding);

      // Both should produce identical output since the actual pixel data
      // is the same — only the stride differs.
      expect(resultPadded, equals(resultNoPadding));
    });

    test('BGRA8888 extracts RGB correctly', () {
      final image = createFakeBgra8888(
        width: 320,
        height: 320,
        rValue: 100,
        gValue: 150,
        bValue: 200,
      );
      final result = preprocessCameraImage(image);

      // Check first pixel: should be R=100, G=150, B=200.
      expect(result[0], 100); // R
      expect(result[1], 150); // G
      expect(result[2], 200); // B
    });

    test('throws on unsupported format', () {
      final image = _FakeCameraImage(
        width: 320,
        height: 320,
        format: _FakeImageFormat(ImageFormatGroup.jpeg),
        planes: [],
      );
      expect(
        () => preprocessCameraImage(image),
        throwsA(isA<UnsupportedError>()),
      );
    });
  });
}

// ---------------------------------------------------------------------------
// Fake CameraImage implementation for unit testing.
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
