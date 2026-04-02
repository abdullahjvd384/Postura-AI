@TestOn('vm')
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:postura/ml/detection_state.dart';
import 'package:postura/ml/label_constants.dart';
import 'package:postura/ml/rolling_window.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

/// Sustained stability test — B8 requirement.
///
/// Runs inference continuously for ~3 minutes at simulated 2fps (360 frames)
/// against the real EfficientDet-Lite0 model. Verifies:
///   - No crash over the full run
///   - Memory does not grow unbounded (RSS delta stays within tolerance)
///   - Model produces consistent detections throughout
///   - RollingWindow state machine stays correct after hundreds of frames
void main() {
  const testDurationFrames = 360; // 3 minutes × 2fps = 360 frames
  const inputSize = 320;

  test(
    'sustained 3-minute inference — no crash, no memory leak',
    timeout: const Timeout(Duration(minutes: 10)),
    () {
      // ── Load model ──
      final modelPath =
          '${Directory.current.path}${Platform.pathSeparator}assets${Platform.pathSeparator}ml${Platform.pathSeparator}efficientdet_lite0.tflite';
      final modelFile = File(modelPath);
      expect(modelFile.existsSync(), isTrue,
          reason: 'Model file must exist at $modelPath');

      final interpreter = Interpreter.fromFile(modelFile,
          options: InterpreterOptions()..threads = 2)
        ..allocateTensors();

      // ── Prepare test images ──
      final rgbBuffers = <Uint8List>[];

      final imagesDir = Directory(
          '${Directory.current.path}${Platform.pathSeparator}Postura-images');
      if (imagesDir.existsSync()) {
        final imageFiles = imagesDir
            .listSync()
            .whereType<File>()
            .where((f) =>
                f.path.toLowerCase().endsWith('.jpg') ||
                f.path.toLowerCase().endsWith('.png'))
            .take(5)
            .toList();

        for (final file in imageFiles) {
          final decoded = img.decodeImage(file.readAsBytesSync());
          if (decoded != null) {
            final resized = img.copyResize(decoded,
                width: inputSize,
                height: inputSize,
                interpolation: img.Interpolation.linear);
            final buf = Uint8List(inputSize * inputSize * 3);
            var o = 0;
            for (int y = 0; y < inputSize; y++) {
              for (int x = 0; x < inputSize; x++) {
                final p = resized.getPixel(x, y);
                buf[o++] = p.r.toInt();
                buf[o++] = p.g.toInt();
                buf[o++] = p.b.toInt();
              }
            }
            rgbBuffers.add(buf);
          }
        }
      }

      // Fallback: synthetic grey image.
      if (rgbBuffers.isEmpty) {
        rgbBuffers.add(Uint8List(inputSize * inputSize * 3)
          ..fillRange(0, inputSize * inputSize * 3, 128));
      }

      print('Loaded ${rgbBuffers.length} test image(s)');
      print('Running $testDurationFrames frames (~3 min at 2fps)...\n');

      // ── Baseline memory ──
      final rssBefore = _getCurrentRssBytes();

      // ── Run sustained inference loop ──
      final rollingWindow = RollingWindow();
      var confirmedCount = 0;
      var partialCount = 0;
      var searchingCount = 0;
      var inferenceErrors = 0;
      var totalDetectionsKept = 0;

      final stopwatch = Stopwatch()..start();

      for (int frame = 0; frame < testDurationFrames; frame++) {
        final rgbBuffer = rgbBuffers[frame % rgbBuffers.length];
        final input = rgbBuffer.reshape([1, inputSize, inputSize, 3]);

        // Allocate fresh output buffers each frame (matches real code path).
        final outBoxes = List.generate(
            1, (_) => List.generate(25, (_) => List.filled(4, 0.0)));
        final outClasses = List.generate(1, (_) => List.filled(25, 0.0));
        final outScores = List.generate(1, (_) => List.filled(25, 0.0));
        final outCount = List.filled(1, 0.0);

        try {
          interpreter.runForMultipleInputs(
              [input], {0: outBoxes, 1: outClasses, 2: outScores, 3: outCount});
        } catch (e) {
          inferenceErrors++;
          continue;
        }

        // Parse detections (mirrors ModelHandler._parseDetections).
        final count = outCount[0].toInt().clamp(0, 25);
        var hasPerson = false;
        var hasMonitor = false;

        for (int i = 0; i < count; i++) {
          if (outScores[0][i] < confidenceThreshold) continue;
          totalDetectionsKept++;
          final classId = outClasses[0][i].toInt();
          if (classId == cocoPersonClassId) hasPerson = true;
          if (monitorClassIds.contains(classId)) hasMonitor = true;
        }

        final bothPresent = hasPerson && hasMonitor;

        // Reset rolling window after confirmed to simulate multiple cycles.
        if (rollingWindow.isLocked) {
          rollingWindow.reset();
        }
        rollingWindow.addResult(bothPresent);

        final state = rollingWindow.currentState;
        switch (state) {
          case DetectionState.confirmed:
            confirmedCount++;
          case DetectionState.partial:
            partialCount++;
          case DetectionState.searching:
            searchingCount++;
        }

        // Progress log every 60 frames (~30s of simulated time).
        if ((frame + 1) % 60 == 0) {
          final elapsed = stopwatch.elapsed;
          print(
              '  Frame ${frame + 1}/$testDurationFrames  '
              'elapsed=${elapsed.inSeconds}s  '
              'confirmed=$confirmedCount  partial=$partialCount  '
              'searching=$searchingCount  errors=$inferenceErrors');
        }
      }

      stopwatch.stop();

      // ── Memory after ──
      final rssAfter = _getCurrentRssBytes();

      // ── Report ──
      print('\n=== STABILITY TEST RESULTS ===');
      print('Total frames: $testDurationFrames');
      print('Wall time: ${stopwatch.elapsed.inSeconds}s');
      print('Inference errors: $inferenceErrors');
      print('Total detections kept: $totalDetectionsKept');
      print('State counts -> confirmed: $confirmedCount, '
          'partial: $partialCount, searching: $searchingCount');

      if (rssBefore > 0 && rssAfter > 0) {
        final beforeMB = rssBefore / (1024 * 1024);
        final afterMB = rssAfter / (1024 * 1024);
        final deltaMB = afterMB - beforeMB;
        print('RSS before: ${beforeMB.toStringAsFixed(1)} MB');
        print('RSS after:  ${afterMB.toStringAsFixed(1)} MB');
        print('RSS delta:  ${deltaMB.toStringAsFixed(1)} MB');

        // Allow up to 100 MB growth — anything beyond suggests a leak.
        expect(deltaMB, lessThan(100),
            reason: 'Memory grew by ${deltaMB.toStringAsFixed(1)} MB over '
                '$testDurationFrames frames — possible memory leak');
      } else {
        print('Memory tracking: not available on this platform');
      }

      // ── Assertions ──
      expect(inferenceErrors, 0,
          reason: 'Inference should never error on valid RGB buffers');

      expect(totalDetectionsKept, greaterThan(0),
          reason: 'Model should produce detections across $testDurationFrames frames');

      // Verify the rolling window cycled correctly — should have confirmed
      // multiple times if test images contain person + monitor.
      if (rgbBuffers.length > 1) {
        expect(confirmedCount, greaterThan(0),
            reason: 'Should reach confirmed state at least once with real images');
      }

      print('\nSTABILITY TEST PASSED');

      interpreter.close();
    },
  );
}

/// Get current process RSS in bytes. Returns 0 if unavailable.
int _getCurrentRssBytes() {
  try {
    if (Platform.isWindows) {
      final result = Process.runSync('tasklist', [
        '/FI', 'PID eq ${pid}',
        '/FO', 'CSV',
        '/NH',
      ]);
      if (result.exitCode == 0) {
        final line = (result.stdout as String).trim();
        final parts = line.split('","');
        if (parts.length >= 5) {
          // Last field is memory like: 123,456 K" or 123.456 K"
          final memStr = parts[4]
              .replaceAll('"', '')
              .replaceAll(' K', '')
              .replaceAll(',', '')
              .replaceAll('.', '')
              .trim();
          final kb = int.tryParse(memStr);
          if (kb != null) return kb * 1024;
        }
      }
    } else {
      final result = Process.runSync('ps', ['-o', 'rss=', '-p', '$pid']);
      if (result.exitCode == 0) {
        final kb = int.tryParse((result.stdout as String).trim());
        if (kb != null) return kb * 1024;
      }
    }
  } catch (_) {}
  return 0;
}
