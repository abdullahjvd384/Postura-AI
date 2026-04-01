import 'dart:collection';
import 'detection_state.dart';

/// Rolling confirmation window for Step 1 frame validation.
///
/// Tracks the last [windowSize] per-frame detection results and determines
/// the overall [DetectionState]. GREEN fires only when [requiredPositives]
/// out of [windowSize] frames are positive — this prevents flicker from
/// a single good frame among noise.
class RollingWindow {
  /// Number of frames in the rolling window.
  static const int windowSize = 6;

  /// Minimum positive frames needed to trigger [DetectionState.confirmed].
  static const int requiredPositives = 5;

  /// Number of consecutive false frames that triggers a full window reset.
  /// Prevents stale positives from carrying across a sustained detection dropout.
  static const int resetAfterConsecutiveFalse = 4;

  final Queue<bool> _window = Queue<bool>();
  int _consecutiveFalse = 0;
  bool _locked = false;

  /// Add a per-frame detection result.
  ///
  /// [detected] is `true` when both person AND monitor were found above
  /// the confidence threshold in this frame.
  ///
  /// After [DetectionState.confirmed] is returned, the window locks and
  /// ignores further results — the frontend should call shutdown.
  void addResult(bool detected) {
    if (_locked) return;

    // Track consecutive false frames for reset logic.
    if (detected) {
      _consecutiveFalse = 0;
    } else {
      _consecutiveFalse++;
      if (_consecutiveFalse >= resetAfterConsecutiveFalse) {
        _window.clear();
        _consecutiveFalse = 0;
        return;
      }
    }

    _window.addLast(detected);
    while (_window.length > windowSize) {
      _window.removeFirst();
    }

    // Lock on confirmed so no further changes can occur.
    if (currentState == DetectionState.confirmed) {
      _locked = true;
    }
  }

  /// Current detection state based on the rolling window contents.
  DetectionState get currentState {
    if (_locked) return DetectionState.confirmed;

    final positives = _window.where((r) => r).length;

    if (positives >= requiredPositives) {
      return DetectionState.confirmed;
    } else if (positives > 0) {
      return DetectionState.partial;
    } else {
      return DetectionState.searching;
    }
  }

  /// Whether the window has locked after confirmation.
  bool get isLocked => _locked;

  /// Number of positive results currently in the window.
  int get positiveCount => _window.where((r) => r).length;

  /// Current window size (may be less than [windowSize] at startup).
  int get currentSize => _window.length;

  /// Reset to initial state. Used for testing or re-initialization.
  void reset() {
    _window.clear();
    _consecutiveFalse = 0;
    _locked = false;
  }
}
