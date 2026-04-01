/// Detection states for Step 1 frame validation.
///
/// Communicated from ML layer → Frontend via [DetectionService.onDetectionResult].
enum DetectionState {
  /// No confirmation yet — neither object reliably detected.
  searching,

  /// One object detected (person or monitor), waiting for the other.
  partial,

  /// Both person AND monitor confirmed across the rolling window.
  /// Frontend should stop the frame loop and call [DetectionService.shutdown].
  confirmed,
}
