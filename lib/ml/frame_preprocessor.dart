import 'dart:typed_data';
import 'package:camera/camera.dart';

/// Target dimensions for EfficientDet-Lite0 input tensor.
const int _modelInputSize = 320;

/// Total bytes in the output RGB buffer: 320 * 320 * 3.
const int _outputBufferSize = _modelInputSize * _modelInputSize * 3;

/// Converts a [CameraImage] to a flat RGB [Uint8List] of size 320x320x3,
/// suitable for feeding into the EfficientDet-Lite0 input tensor.
///
/// Handles platform-specific formats:
/// - **Android (YUV420):** Uses `bytesPerRow` (stride) — NOT `image.width` —
///   to avoid row-padding corruption.
/// - **iOS (BGRA8888):** Standard 4-byte pixel access.
///
/// Resizes via nearest-neighbor interpolation (sufficient for object detection
/// at 2fps).
Uint8List preprocessCameraImage(CameraImage image) {
  switch (image.format.group) {
    case ImageFormatGroup.yuv420:
      return _convertYuv420(image);
    case ImageFormatGroup.bgra8888:
      return _convertBgra8888(image);
    default:
      throw UnsupportedError(
        'Unsupported camera image format: ${image.format.group}',
      );
  }
}

/// Converts YUV420 (Android) to 320x320 RGB.
///
/// CRITICAL: Uses `planes[n].bytesPerRow` for row stride, NOT `image.width`.
/// The buffer may contain padding bytes at the end of each row. Using
/// `image.width` causes misaligned reads from row 2 onward.
Uint8List _convertYuv420(CameraImage image) {
  final int srcWidth = image.width;
  final int srcHeight = image.height;

  final yPlane = image.planes[0];
  final uPlane = image.planes[1];
  final vPlane = image.planes[2];

  final int yRowStride = yPlane.bytesPerRow;
  final int uvRowStride = uPlane.bytesPerRow;
  final int uvPixelStride = uPlane.bytesPerPixel ?? 1;

  final yBytes = yPlane.bytes;
  final uBytes = uPlane.bytes;
  final vBytes = vPlane.bytes;

  final output = Uint8List(_outputBufferSize);

  for (int outY = 0; outY < _modelInputSize; outY++) {
    // Map output row to source row (nearest-neighbor).
    final int srcRow = (outY * srcHeight) ~/ _modelInputSize;

    for (int outX = 0; outX < _modelInputSize; outX++) {
      // Map output col to source col (nearest-neighbor).
      final int srcCol = (outX * srcWidth) ~/ _modelInputSize;

      // Y value — use stride, NOT width.
      final int yIndex = srcRow * yRowStride + srcCol;
      final int y = yBytes[yIndex];

      // U and V are subsampled at half resolution.
      final int uvRow = srcRow ~/ 2;
      final int uvCol = srcCol ~/ 2;
      final int uvIndex = uvRow * uvRowStride + uvCol * uvPixelStride;

      final int u = uBytes[uvIndex];
      final int v = vBytes[uvIndex];

      // YUV → RGB conversion.
      final int r = (y + 1.370705 * (v - 128)).round().clamp(0, 255);
      final int g =
          (y - 0.337633 * (u - 128) - 0.698001 * (v - 128))
              .round()
              .clamp(0, 255);
      final int b = (y + 1.732446 * (u - 128)).round().clamp(0, 255);

      final int outIndex = (outY * _modelInputSize + outX) * 3;
      output[outIndex] = r;
      output[outIndex + 1] = g;
      output[outIndex + 2] = b;
    }
  }

  return output;
}

/// Converts BGRA8888 (iOS) to 320x320 RGB.
Uint8List _convertBgra8888(CameraImage image) {
  final int srcWidth = image.width;
  final int srcHeight = image.height;

  final plane = image.planes[0];
  final int rowStride = plane.bytesPerRow;
  final bytes = plane.bytes;

  final output = Uint8List(_outputBufferSize);

  for (int outY = 0; outY < _modelInputSize; outY++) {
    final int srcRow = (outY * srcHeight) ~/ _modelInputSize;

    for (int outX = 0; outX < _modelInputSize; outX++) {
      final int srcCol = (outX * srcWidth) ~/ _modelInputSize;

      // BGRA byte order: B=0, G=1, R=2, A=3.
      final int pixelOffset = srcRow * rowStride + srcCol * 4;

      final int outIndex = (outY * _modelInputSize + outX) * 3;
      output[outIndex] = bytes[pixelOffset + 2]; // R
      output[outIndex + 1] = bytes[pixelOffset + 1]; // G
      output[outIndex + 2] = bytes[pixelOffset]; // B
    }
  }

  return output;
}
