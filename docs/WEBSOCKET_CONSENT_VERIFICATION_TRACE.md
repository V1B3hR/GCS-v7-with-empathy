# WebSocket, Consent, and Verification: Implementation Trace

This document describes the current code paths, not intended or deployment-ready
behavior. The linked source is authoritative if this trace becomes outdated.

## 1. Real-time affective-state WebSocket

### Server path

[`backend/gcs/serving/server.py`](../backend/gcs/serving/server.py) defines
`AffectiveServer` and the FastAPI `websocket_endpoint` at `/ws`.

1. On application startup, `startup_event` constructs `AffectiveServer` from
   `affective_config.yaml`, builds and compiles the affective model, and starts
   configured EEG, physiology, and voice adapters with
   `start_streaming_adapters()`. The application shutdown handler calls
   `stop_streaming_adapters()`.
2. A client connects to `/ws`; the handler accepts it, adds it to
   `active_connections`, and sends `{"status":"connected"}`.
3. The handler loops at `1 / push_hz` seconds (default 3 Hz). Each iteration,
   `get_streaming_data()` collects any available adapter windows/features and
   adds a batch dimension. If it has no modality data, it returns `None` and
   that iteration sends no prediction.
4. With inputs, `predict()` calls the model's
   `predict_with_uncertainty()`, derives valence, arousal, categorical emotion
   and confidence, and checks the configured crisis thresholds. The
   `format_for_frontend()` result is sent with `send_json()`.
5. `WebSocketDisconnect` and other exceptions leave the loop; the `finally`
   block removes the socket from `active_connections` and updates the optional
   connection metric.

The prediction message has this shape:

```json
{
  "affective": {
    "label": "emotion",
    "icon": "😐",
    "strength": 50,
    "valence": 0.0,
    "arousal": 0.0,
    "confidence": 0.9
  },
  "empathic_response": {
    "content": "I sense you're feeling emotion.",
    "intensity": "moderate",
    "type": "validation",
    "confidence": 0.9
  },
  "privacy_protected": true,
  "crisis_detected": false,
  "cultural_adaptation": "neutral"
}
```

The values above illustrate the keys, not a guaranteed prediction. In
particular, `predict()` also calculates uncertainty and inference time, but
`format_for_frontend()` does not include those fields in the WebSocket payload.
The WebSocket path calls `predict(inputs)` without a `user_id`.

### Frontend path and lifecycle

[`frontend/src/hooks/useBCIData.js`](../frontend/src/hooks/useBCIData.js)
opens a hard-coded `ws://localhost:8000/ws` socket in the hook's connection
effect.

- `onopen` sets `connectionStatus` to `CONNECTED`.
- `onmessage` parses JSON and acts only when `data.affective` exists. It maps
  the affective fields plus `crisis_detected` and `empathic_response` into
  `affectiveState`, then appends that state to `affectiveHistory` (bounded to
  30 entries in this path).
- The initial server `{"status":"connected"}` message has no `affective` key,
  so it does not update affective state or history. Connection status comes
  from the browser's `onopen`, not that message.
- `onclose` sets `SIMULATED` and schedules a reconnect after five seconds;
  `onerror` closes the socket. The hook cleanup marks the effect unsubscribed,
  closes the current socket, and clears a scheduled reconnect.
- A separate interval always synthesizes vitals, EEG-band values, region
  activations, and somatic hotspots from the selected preset. While status is
  anything other than `CONNECTED`, that interval also synthesizes
  `affectiveState` from the preset scenarios and periodically logs it.

### Boundaries and limitations

- The server does **not** send synthetic affective messages when adapters have
  no data; it simply skips that loop iteration. The frontend's simulation is
  separate local scenario logic, not a backend fallback or replay of sensor
  data.
- If the socket opens but the server sends no affective messages, the frontend
  remains `CONNECTED` and its local affective simulation is paused. The last
  affective state therefore remains displayed until a prediction arrives.
- The frontend consumes the backend's `affective`, `empathic_response`, and
  `crisis_detected` keys. It ignores `privacy_protected` and
  `cultural_adaptation`; it also does not receive the backend's uncertainty or
  inference-time values. There is no browser-to-server message flow in this
  hook.
- The frontend's `EMOTIONS` taxonomy and preset labels are not an enforced
  contract with the backend's `EmotionalState`. Unknown labels can still be
  displayed; the hook supplies fallback values for missing affective fields.
- The local URL is fixed to `ws://localhost:8000/ws`, not derived from runtime
  configuration. Deployment must account for that limitation, including
  transport security and the actual backend host.

## 2. Consent replay and staleness protections

[`backend/gcs/consent_gate.py`](../backend/gcs/consent_gate.py) implements
`InterventionConsentGate` as a single-current-request state machine. It is a
guard for a proposal, not an identity provider or a hardware safety system.

### Request identity and binding

- `propose(modality, params, target)` trims and validates nonempty modality and
  target strings. It serializes parameters as deterministic JSON
  (`canonicalize_parameters`: sorted keys, compact separators, no non-finite
  numbers), stores that normalized parameter mapping, and stores its SHA-256
  fingerprint.
- Every accepted proposal receives a fresh UUID `request_id`. While a request
  is pending or approved, another proposal is suppressed. A terminal request
  can be replaced only by a later proposal; its approval is not carried over.
- `record_decision(request_id, decision, actor_id)` requires the supplied ID
  to equal the current request ID. `authorize_execution(...)` independently
  checks that ID and requires exact normalized modality and target equality
  and a matching fingerprint for the parameters. Reordering parameter keys
  does not change the canonical fingerprint; changing a value or key does.

### Expiration, decisions, cooldown, and consumption

- The default request TTL is 60 seconds. `propose()` sets `expires_at` from its
  creation time; `_expire_if_needed()` expires pending **and approved**
  requests when `now >= expires_at`. Approval does not restart the TTL.
  Expiration is applied when gate operations call `_expire_if_needed()`.
- Allowed decisions are `accept`, `decline`, `defer`, and `revoke`.
  Acceptance, decline, and defer require a pending request; revoke accepts a
  pending or approved request. Only an approved, unexpired request can execute.
  Declined, deferred, revoked, expired, and consumed requests fail execution.
- Decline, defer, and revoke set their respective cooldowns (each defaults to
  300 seconds and can be configured). `propose()` suppresses creation while a
  cooldown is active. Acceptance does not set a cooldown.
- After all identity, status, expiry, modality, target, and parameter checks
  pass, `authorize_execution()` changes the request to `consumed` before it
  returns the execution payload. A second call therefore fails. In
  `ClosedLoopAgent.execute_approved_request()` this consumption happens before
  low-level controller delivery; delivery failure is recorded and is not
  automatically retried.

### Actor requirement and caller responsibilities

`record_decision()` checks that `actor_id` is a nonempty string and stores it
with the decision. **It does not authenticate that actor or check their
authorization.** The gate and
[`ClosedLoopAgent.record_consent_decision()`](../backend/gcs/closed_loop_agent.py)
explicitly require callers to authenticate and authorize the human decision
actor before invoking the decision method. The class itself cannot establish
that a caller has done so. Model output, inferred affect, EEG, voice, and
physiological signals are not valid consent sources.

The gate also does not perform the hardware action itself. Its one-slot
in-memory state is not a distributed lock, durable audit store, or substitute
for hardware interlocks. Callers must preserve the authenticate/authorize
boundary and pass the actual intended execution parameters to
`authorize_execution()`.

## 3. Verification framework maturity

[`backend/gcs/verification_framework.py`](../backend/gcs/verification_framework.py)
contains useful runtime-monitoring and reporting scaffolding. Its class and
method names describe aspirations as well as implemented behavior; the
placeholder paths below are not formal assurance results.

### Implemented behavior

- `FormalProperty`, `VerificationEvidence`, and `AssuranceCase` are data
  records. `VerificationFramework.__init__()` creates the configured data
  directory, initializes in-memory registries, and registers five core
  properties in `_initialize_core_properties()`.
- `register_property()` stores a property and registers it with
  `RuntimeMonitor` when `monitor_enabled` is true.
- `RuntimeMonitor.check_property(property_id, state)` calls
  `_evaluate_specification()`, updates the property status, and, on a false
  result, records a violation, increments `violation_count`, and logs an alert
  when configured. `get_violations()` returns recorded entries, optionally
  filtered by timestamp.
- `generate_verification_report()` summarizes the current property statuses,
  violation counts, assurance-level coverage, assurance-case completeness,
  and recent runtime violations. `check_phase21_exit_criteria()` compares
  those computed values against configured numeric targets.
- `add_evidence()` appends evidence to an in-memory per-property list.
  `create_assurance_case()` stores a case and calculates a simple completeness
  score: evidence count divided by strategy count, capped at 1.0 (or zero if
  there are no strategies).

### Simplifications, stubs, and caveats

- `_evaluate_specification()` is not a formal-specification evaluator. It
  dispatches on a few words (`latency`, `accuracy`, `fairness`, `privacy`,
  `ethical`), checks a specific state key, and extracts the **first number**
  from the specification as a threshold. If it does not recognize a property
  or the expected key is missing, it returns `True` by default. That is
  “assumed to hold,” not verified.
- The built-in properties illustrate concrete mismatches with that evaluator:
  the liveness specification (`EVENTUALLY_WITHIN_5MIN...`) has no matching
  evaluator branch; the privacy specification names
  `unauthorized_access_count` but the privacy branch reads `privacy_violations`;
  and the performance specification begins with `P95`, so the simple
  threshold extractor sees `95` before the stated `150`. These specs do not
  establish the advertised liveness, privacy, or percentile guarantees.
- `verify_property()` returns the current status for `runtime_monitoring`;
  it does not itself sample system state or perform a check. `_model_check()`
  is explicitly a placeholder that unconditionally sets status to
  `VERIFIED`; `_test_property()` likewise sets `VERIFIED` without running
  property-based tests. There is no TLA+, SPIN, NuSMV, Z3, Hypothesis, or
  QuickCheck integration in these methods.
- `RuntimeMonitor` performs only the simplified one-state check above. This
  is not temporal model checking, robust formal parsing, or proof. Statuses,
  evidence, assurance cases, and violations are held in memory; the methods
  shown do not persist them to `data_dir`.
- Consequently, report coverage and Phase 21 exit-criteria results are
  bookkeeping derived from these statuses and counts. In particular, a
  `VERIFIED` status from a placeholder is not evidence of a proof, and a
  criterion passing is not an independent assurance result.

## Maintainer inspection checks

From the repository root, inspect the linked method bodies directly:

```sh
rg -n 'websocket_endpoint|start_streaming_adapters|get_streaming_data|predict|format_for_frontend' backend/gcs/serving/server.py
rg -n 'WebSocket Connection Handler|onmessage|onclose|connectionStatus !== .CONNECTED.' frontend/src/hooks/useBCIData.js
rg -n 'def propose|def record_decision|def authorize_execution|def _expire_if_needed' backend/gcs/consent_gate.py
rg -n 'def _evaluate_specification|def verify_property|def _model_check|def _test_property|def generate_verification_report' backend/gcs/verification_framework.py
```

The consent behavior also has focused tests for exact binding, expiry,
terminal decisions, cooldowns, and one-time execution:

```sh
PYTHONPATH=backend python -m unittest discover -s backend/gcs/tests -p 'test_consent_gate.py'
```

For a live WebSocket check, start the backend with its required model and
streaming dependencies, then connect a WebSocket client to
`ws://localhost:8000/ws`. Expect the initial status message first; prediction
messages arrive only when at least one configured adapter supplies data.
