# Input Layer Boundary Notes

This document records the current ownership decision for tablet input handling in
`canva-test`, `oani-input`, and `canvas_core`.

## Current Finding

The latest tap and short-stroke defects were reproduced while the input audit
reported:

- `core upload/dispatch: 0/0`
- touch move batches arriving before any committed core brush dispatch
- begin pressure sometimes coming from fallback state while the first touch move
  already carried device pressure

That means these failures happened before `canvas_core` received real brush work.
The stroke-session rules now live in `oani-input` as
`canva_input::StrokeInputSession`, not in `canvas_core`.

## Current `canva-test` Responsibilities

`canva-test` still owns UI-framework and app-adapter work:

- reading `egui::Event::Touch` and `PointerMoved` as fallback input
- installing the Windows Pointer capture guard and draining normalized native
  events from `oani-input`
- converting viewport positions into canvas coordinates
- adapting egui samples into `canva_input::StrokeInputPoint`
- translating normalized input actions into `CanvasCommand`
- drawing the raw lead preview overlay from the latest normalized input sample
- collecting audit data, internal latency stages, and drawing the raw-input
  overlay

`canva-test` should not own tap jitter thresholds, pointer-to-touch reanchor
policy, pressure provenance policy, or tap fallback semantics.

## `oani-input` Session Boundary

`oani-input` owns raw input normalization and stroke-session state:

```text
egui fallback events or Windows Pointer native events
  -> oani-input platform/session normalizers
  -> normalized stroke input events
  -> canva-test CanvasCommand adapter
  -> canvas_core
```

The normalized session API describes intent, not UI-framework details:

- `StrokeInputSession::begin(...)`
- `StrokeInputSession::append_batch(...)`
- `StrokeInputSession::mark_core_dispatched()`
- `StrokeInputSession::end()`
- `StrokeInputSession::reset()`

The session owns:

- pointer fallback reanchor to the first valid touch sample
- touch begin reanchor when current-event device pressure first appears
- tap candidate lifetime
- tap jitter filtering
- fallback tap commit when an active stroke never dispatches
- pressure provenance rules through `resolve_stroke_pressure`

## `canvas_core` Boundary

`canvas_core` should remain independent from platform input details. It should
continue to own:

- stroke command execution
- stabilizer behavior
- GPU brush dispatch
- active-stroke preview buffers
- history and undo grouping
- render/composite/present behavior

Core changes are appropriate only when the defect exists after normalized stroke
commands reach the engine, or when shader/stabilizer/history behavior is wrong.

## Migration Checklist

1. Done: add an `oani-input` session normalizer for adapted points with source
   and pressure provenance metadata.
2. Done: move tap candidate, tap jitter, reanchor, and pressure provenance rules
   from `canva-test` into that normalizer.
3. Done: port the current regression cases into `oani-input` tests:
   - missing touch start can begin from first move
   - touch start is not appended as a move sample
   - pointer fallback can reanchor to first touch sample
   - first device-pressure move can reanchor a fallback-pressure begin
   - short tap jitter does not become a one-sided smear
4. Done: replace the local `canva-test` state machine with a simple adapter from
   normalized input events to `CanvasCommand`.
5. Started: keep the input audit overlay in `canva-test`, but feed it from
   diagnostics emitted by `oani-input`.
6. Started: draw a raw lead preview from the latest normalized input point so
   visual feedback can bypass stabilizer/core dispatch delay.
7. Done: connect Windows Pointer native capture for the test app. The native
   path subclasses the eframe `HWND` through `oani-input`, captures
   `WM_POINTER*`, reads `GetPointerPenInfoHistory`, maps native QPC timestamps
   into the app frame-time domain, and falls back to egui input when no native
   samples are available.
8. Started: measure internal latency stages with a monotonic clock:
   source timestamp age, receive-to-session, receive-to-core-done, core append,
   and receive-to-paint. The Windows Pointer path now feeds QPC/history
   timestamps into the source timestamp stage. The UI also reports newest/oldest
   source timestamp age for each input batch, so coalesced history samples can
   expose how old the oldest point was when the frame processed it.
9. Next: add a process-level GUI automation test once the app has a stable test
   harness for synthetic egui input.
