import 'package:flutter_test/flutter_test.dart';
import 'package:postura/ml/detection_state.dart';
import 'package:postura/ml/rolling_window.dart';

void main() {
  late RollingWindow window;

  setUp(() {
    window = RollingWindow();
  });

  group('RollingWindow', () {
    test('starts in searching state', () {
      expect(window.currentState, DetectionState.searching);
      expect(window.currentSize, 0);
      expect(window.positiveCount, 0);
    });

    test('6 true frames yields confirmed', () {
      for (int i = 0; i < 6; i++) {
        window.addResult(true);
      }
      expect(window.currentState, DetectionState.confirmed);
      expect(window.isLocked, true);
    });

    test('5 true + 1 false yields confirmed', () {
      // Add 5 trues then 1 false
      for (int i = 0; i < 5; i++) {
        window.addResult(true);
      }
      window.addResult(false);
      expect(window.currentState, DetectionState.confirmed);
    });

    test('4 true + 2 false yields partial (not confirmed)', () {
      for (int i = 0; i < 4; i++) {
        window.addResult(true);
      }
      window.addResult(false);
      window.addResult(false);
      expect(window.currentState, DetectionState.partial);
    });

    test('all false yields searching', () {
      for (int i = 0; i < 6; i++) {
        window.addResult(false);
      }
      // After 4+ consecutive false, window resets — state is searching
      expect(window.currentState, DetectionState.searching);
    });

    test('1 true among 5 false yields partial', () {
      window.addResult(true);
      window.addResult(false);
      window.addResult(false);
      window.addResult(false);
      // 3 consecutive false — not enough to reset (need 4)
      expect(window.currentState, DetectionState.partial);
    });

    test('sustained false (4+ consecutive) resets window', () {
      // Build up 3 trues
      window.addResult(true);
      window.addResult(true);
      window.addResult(true);
      expect(window.positiveCount, 3);

      // Now 4 consecutive false — triggers reset
      window.addResult(false);
      window.addResult(false);
      window.addResult(false);
      window.addResult(false);
      expect(window.currentSize, 0);
      expect(window.currentState, DetectionState.searching);

      // After reset, 2 trues should give partial, not confirmed
      window.addResult(true);
      window.addResult(true);
      expect(window.currentState, DetectionState.partial);
      expect(window.positiveCount, 2);
    });

    test('locks after confirmed — further results ignored', () {
      for (int i = 0; i < 6; i++) {
        window.addResult(true);
      }
      expect(window.isLocked, true);
      expect(window.currentState, DetectionState.confirmed);

      // Add false — should be ignored
      window.addResult(false);
      expect(window.currentState, DetectionState.confirmed);
      expect(window.positiveCount, 6);
    });

    test('window trims to max 6 entries', () {
      for (int i = 0; i < 10; i++) {
        window.addResult(true);
      }
      // Should lock at 5 or 6, but currentSize should never exceed 6
      expect(window.currentSize, lessThanOrEqualTo(RollingWindow.windowSize));
    });

    test('interleaved true/false pattern', () {
      // true, false, true, false, true, false = 3 true, 3 false → partial
      for (int i = 0; i < 6; i++) {
        window.addResult(i.isEven);
      }
      expect(window.currentState, DetectionState.partial);
      expect(window.positiveCount, 3);
    });

    test('reset clears all state', () {
      for (int i = 0; i < 5; i++) {
        window.addResult(true);
      }
      window.reset();
      expect(window.currentState, DetectionState.searching);
      expect(window.currentSize, 0);
      expect(window.isLocked, false);
    });

    test('confirmation at exactly 5 of 6 boundary', () {
      // false, true, true, true, true, true = 5 true → confirmed
      window.addResult(false);
      for (int i = 0; i < 5; i++) {
        window.addResult(true);
      }
      expect(window.currentState, DetectionState.confirmed);
    });
  });
}
