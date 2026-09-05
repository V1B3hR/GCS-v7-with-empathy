import copy
import hashlib
import json
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Mapping, Optional


CONSENT_PENDING = "pending_consent"
CONSENT_APPROVED = "approved"
CONSENT_DECLINED = "declined"
CONSENT_DEFERRED = "deferred"
CONSENT_REVOKED = "revoked"
CONSENT_EXPIRED = "expired"
CONSENT_CONSUMED = "consumed"

ACTIVE_STATUSES = {CONSENT_PENDING, CONSENT_APPROVED}
TERMINAL_STATUSES = {
    CONSENT_DECLINED,
    CONSENT_DEFERRED,
    CONSENT_REVOKED,
    CONSENT_EXPIRED,
    CONSENT_CONSUMED,
}
DECISION_TO_STATUS = {
    "accept": CONSENT_APPROVED,
    "decline": CONSENT_DECLINED,
    "defer": CONSENT_DEFERRED,
    "revoke": CONSENT_REVOKED,
}


class ConsentGateError(RuntimeError):
    """Base error for consent gate failures."""


class ConsentDecisionError(ConsentGateError):
    """Raised when a consent decision is invalid."""


class ConsentExecutionError(ConsentGateError):
    """Raised when execution is attempted without valid consent."""


def _isoformat(timestamp: Optional[float]) -> Optional[str]:
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def canonicalize_parameters(params: Mapping[str, Any]) -> str:
    """Return deterministic JSON for exact parameter binding and comparison."""
    if not isinstance(params, Mapping):
        raise TypeError(f"params must be a mapping, got {type(params)}")
    try:
        return json.dumps(params, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"params must be JSON-serializable with finite values: {exc}") from exc


def parameter_fingerprint(params: Mapping[str, Any]) -> str:
    canonical = canonicalize_parameters(params)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass
class ConsentRequest:
    request_id: str
    modality: str
    params: Dict[str, Any]
    target: str
    created_at: float
    expires_at: float
    parameter_hash: str
    status: str = CONSENT_PENDING
    decision: Optional[str] = None
    decision_actor_id: Optional[str] = None
    decision_at: Optional[float] = None
    consumed_at: Optional[float] = None
    expired_at: Optional[float] = None
    expired_by: Optional[str] = None
    cooldown_until: Optional[float] = None
    cooldown_reason: Optional[str] = None

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "modality": self.modality,
            "params": copy.deepcopy(self.params),
            "target": self.target,
            "created_at": _isoformat(self.created_at),
            "expires_at": _isoformat(self.expires_at),
            "status": self.status,
            "decision": self.decision,
            "decision_actor_id": self.decision_actor_id,
            "decision_at": _isoformat(self.decision_at),
            "consumed_at": _isoformat(self.consumed_at),
            "expired_at": _isoformat(self.expired_at),
            "expired_by": self.expired_by,
            "cooldown_until": _isoformat(self.cooldown_until),
            "cooldown_reason": self.cooldown_reason,
        }


class InterventionConsentGate:
    """
    Manages a single explicit neuromodulation consent request.

    Callers must authenticate and authorize human decision actors before calling
    `record_decision()`. Model output, inferred affect, EEG, voice, physiological
    signals, or other automated signals are never valid consent sources.
    Requests remain bound to their TTL even after approval, so stale approvals
    expire instead of authorizing delayed hardware delivery. Approved requests
    may be revoked while they remain unexpired; after TTL they fail closed as expired.
    """

    def __init__(
        self,
        *,
        request_ttl_seconds: float = 60.0,
        decline_cooldown_seconds: float = 300.0,
        defer_cooldown_seconds: float = 300.0,
        revoke_cooldown_seconds: float = 300.0,
        time_fn: Callable[[], float] = time.time,
    ) -> None:
        self.request_ttl_seconds = self._validate_duration(request_ttl_seconds, "request_ttl_seconds", allow_zero=False)
        self.decline_cooldown_seconds = self._validate_duration(
            decline_cooldown_seconds, "decline_cooldown_seconds", allow_zero=True
        )
        self.defer_cooldown_seconds = self._validate_duration(
            defer_cooldown_seconds, "defer_cooldown_seconds", allow_zero=True
        )
        self.revoke_cooldown_seconds = self._validate_duration(
            revoke_cooldown_seconds, "revoke_cooldown_seconds", allow_zero=True
        )
        self._time_fn = time_fn
        self._request: Optional[ConsentRequest] = None
        self._cooldown_until: Optional[float] = None
        self._cooldown_reason: Optional[str] = None

    @staticmethod
    def _validate_duration(value: Any, name: str, *, allow_zero: bool) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be numeric, got {value!r}") from exc
        minimum = 0.0 if allow_zero else 0.0
        if numeric < minimum or (not allow_zero and numeric <= 0.0):
            comparator = ">= 0" if allow_zero else "> 0"
            raise ValueError(f"{name} must be {comparator}, got {numeric}")
        return numeric

    def _now(self) -> float:
        return float(self._time_fn())

    def _expire_if_needed(self, now: Optional[float] = None) -> None:
        now = self._now() if now is None else float(now)
        if self._request and self._request.status in ACTIVE_STATUSES and now >= self._request.expires_at:
            self._request.status = CONSENT_EXPIRED
            self._request.expired_at = now
            self._request.expired_by = "system"

    def _clear_cooldown_if_elapsed(self, now: Optional[float] = None) -> None:
        now = self._now() if now is None else float(now)
        if self._cooldown_until is not None and now >= self._cooldown_until:
            self._cooldown_until = None
            self._cooldown_reason = None

    def _ensure_actor_id(self, actor_id: str) -> str:
        if not isinstance(actor_id, str) or not actor_id.strip():
            raise ConsentDecisionError("actor_id must be a non-empty authenticated actor identifier")
        return actor_id.strip()

    def _normalize_string(self, value: str, field_name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must be a non-empty string")
        return value.strip()

    def _set_cooldown(self, status: str, now: float) -> None:
        seconds = {
            CONSENT_DECLINED: self.decline_cooldown_seconds,
            CONSENT_DEFERRED: self.defer_cooldown_seconds,
            CONSENT_REVOKED: self.revoke_cooldown_seconds,
        }.get(status)
        if seconds is None:
            return
        self._cooldown_until = now + seconds
        self._cooldown_reason = status
        if self._request:
            self._request.cooldown_until = self._cooldown_until
            self._request.cooldown_reason = status

    def get_current_request(self) -> Optional[Dict[str, Any]]:
        now = self._now()
        self._expire_if_needed(now)
        self._clear_cooldown_if_elapsed(now)
        if self._request is None:
            return None
        self._request.cooldown_until = self._cooldown_until
        self._request.cooldown_reason = self._cooldown_reason
        return self._request.to_public_dict()

    def propose(self, modality: str, params: Mapping[str, Any], target: str) -> Dict[str, Any]:
        now = self._now()
        self._expire_if_needed(now)
        self._clear_cooldown_if_elapsed(now)

        modality = self._normalize_string(modality, "modality")
        target = self._normalize_string(target, "target")
        canonical = canonicalize_parameters(params)
        params_dict = json.loads(canonical)

        if self._request and self._request.status in ACTIVE_STATUSES:
            return {"created": False, "suppressed": True, "reason": "active_request", "request": self._request.to_public_dict()}

        if self._cooldown_until is not None and now < self._cooldown_until:
            return {
                "created": False,
                "suppressed": True,
                "reason": "cooldown",
                "request": self._request.to_public_dict() if self._request else None,
                "cooldown_until": _isoformat(self._cooldown_until),
                "cooldown_reason": self._cooldown_reason,
            }

        self._request = ConsentRequest(
            request_id=str(uuid.uuid4()),
            modality=modality,
            params=params_dict,
            target=target,
            created_at=now,
            expires_at=now + self.request_ttl_seconds,
            parameter_hash=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        )
        return {"created": True, "suppressed": False, "reason": "created", "request": self._request.to_public_dict()}

    def record_decision(self, request_id: str, decision: str, actor_id: str) -> Dict[str, Any]:
        now = self._now()
        self._expire_if_needed(now)
        self._clear_cooldown_if_elapsed(now)
        actor_id = self._ensure_actor_id(actor_id)

        if not isinstance(decision, str):
            raise ConsentDecisionError("decision must be a string")
        decision = decision.strip().lower()
        if decision not in DECISION_TO_STATUS:
            raise ConsentDecisionError(f"decision must be one of {sorted(DECISION_TO_STATUS)}")
        if self._request is None:
            raise ConsentDecisionError("no active consent request exists")
        if request_id != self._request.request_id:
            raise ConsentDecisionError("decision request_id does not match the active request")
        if self._request.status not in ACTIVE_STATUSES:
            raise ConsentDecisionError(f"cannot record {decision!r} for request in status {self._request.status!r}")
        if decision == "accept" and self._request.status != CONSENT_PENDING:
            raise ConsentDecisionError("only a pending request can be accepted")
        if decision in {"decline", "defer"} and self._request.status != CONSENT_PENDING:
            raise ConsentDecisionError(f"only a pending request can be {decision}d")
        if decision == "revoke" and self._request.status not in {CONSENT_PENDING, CONSENT_APPROVED}:
            raise ConsentDecisionError("only a pending or approved request can be revoked")

        new_status = DECISION_TO_STATUS[decision]
        self._request.status = new_status
        self._request.decision = decision
        self._request.decision_actor_id = actor_id
        self._request.decision_at = now
        self._set_cooldown(new_status, now)
        return self._request.to_public_dict()

    def authorize_execution(
        self,
        *,
        request_id: str,
        modality: str,
        params: Mapping[str, Any],
        target: str,
    ) -> Dict[str, Any]:
        now = self._now()
        self._expire_if_needed(now)

        if self._request is None:
            raise ConsentExecutionError("no consent request is available for execution")
        if request_id != self._request.request_id:
            raise ConsentExecutionError("execution request_id does not match the approved request")
        if self._request.status != CONSENT_APPROVED:
            raise ConsentExecutionError(f"request is not executable in status {self._request.status!r}")

        modality = self._normalize_string(modality, "modality")
        target = self._normalize_string(target, "target")
        supplied_hash = parameter_fingerprint(params)
        if modality != self._request.modality:
            raise ConsentExecutionError("execution modality does not match the approved request")
        if target != self._request.target:
            raise ConsentExecutionError("execution target does not match the approved request")
        if supplied_hash != self._request.parameter_hash:
            raise ConsentExecutionError("execution parameters do not match the approved request")

        self._request.status = CONSENT_CONSUMED
        self._request.consumed_at = now
        return {
            "request_id": self._request.request_id,
            "modality": self._request.modality,
            "params": copy.deepcopy(self._request.params),
            "target": self._request.target,
            "approved_by": self._request.decision_actor_id,
            "approved_at": _isoformat(self._request.decision_at),
            "consumed_at": _isoformat(self._request.consumed_at),
        }
