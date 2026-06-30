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
The current fixes therefore live in `canva-test/src/main.rs` as a temporary input
bridge patch, not in `canvas_core`.

## Temporary `canva-test` Responsibilities

`canva-test` currently still owns several input-session decisions:

- reading `egui::Event::Touch` and `PointerMoved`
- choosing a begin sample from touch start or touch move
- ignoring touch start samples during append
- reanchoring a pointer fallback begin to the first valid touch sample
- reanchoring a touch begin when device pressure first appears
- tracking whether pressure came from the current raw event or inherited state
- keeping a tap candidate until a real core dispatch happens
- filtering small touch movement as tap jitter
- falling back to a single dab when a tap never dispatches an active stroke
- collecting audit data and drawing the raw-input overlay

These rules are needed for the test app today, but they are not the right long
term home for platform input semantics.

## Intended Long-Term Boundary

`oani-input` should own raw input normalization and stroke-session state.
`canva-test` should become a thin adapter:

```text
egui raw events
  -> oani-input session normalizer
  -> normalized stroke input events
  -> canva-test CanvasCommand adapter
  -> canvas_core
```

The normalized events should describe intent, not UI-framework details. A useful
target shape is:

- `Begin { point, source, pressure_source }`
- `Append { points, source, pressure_source }`
- `CommitTap { point }`
- `End`
- `Cancel`

After that migration, `canva-test` should no longer contain tap jitter thresholds,
pointer-to-touch reanchor policy, pressure provenance policy, or touch phase
filtering beyond forwarding egui events into `oani-input`.

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

1. Add an `oani-input` session normalizer that accepts raw samples with phase,
   position, force, timestamp, and source metadata.
2. Move tap candidate, tap jitter, reanchor, and pressure provenance rules from
   `canva-test` into that normalizer.
3. Port the current `canva-test` regression cases into `oani-input` tests:
   - missing touch start can begin from first move
   - touch start is not appended as a move sample
   - pointer fallback can reanchor to first touch sample
   - first device-pressure move can reanchor a fallback-pressure begin
   - short tap jitter does not become a one-sided smear
4. Replace the local `canva-test` state machine with a simple adapter from
   normalized input events to `CanvasCommand`.
5. Keep the input audit overlay in `canva-test`, but feed it from normalized
   diagnostics emitted by `oani-input`.
