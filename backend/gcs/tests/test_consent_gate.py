import importlib
import sys
import types
import unittest
from collections import deque

from gcs.consent_gate import ConsentDecisionError, ConsentExecutionError, InterventionConsentGate


class FakeClock:
    def __init__(self, start=1_700_000_000.0):
        self.current = float(start)

    def time(self):
        return self.current

    def advance(self, seconds):
        self.current += float(seconds)


class FakeController:
    def __init__(self, should_fail=False):
        self.should_fail = should_fail
        self.calls = []
        self.targets = []

    def set_target(self, target):
        self.targets.append(target)

    def configure_and_trigger(self, modality, params):
        self.calls.append((modality, params))
        if self.should_fail:
            raise RuntimeError("hardware offline")
        return {"status": "completed", "modality": modality, "params": dict(params)}


class FakeNDArray:
    def __init__(self, size=1):
        self.size = size


def load_closed_loop_agent_class():
    module_name = "gcs.closed_loop_agent"
    for name in [
        module_name,
        "numpy",
        "tensorflow",
        "gcs.inference",
        "gcs.online_learning_module",
        "gcs.feedback_detector",
    ]:
        sys.modules.pop(name, None)

    fake_numpy = types.ModuleType("numpy")
    fake_numpy.ndarray = FakeNDArray

    fake_tf = types.ModuleType("tensorflow")
    fake_tf.keras = types.SimpleNamespace(
        config=types.SimpleNamespace(enable_unsafe_deserialization=lambda: None),
        models=types.SimpleNamespace(load_model=lambda *args, **kwargs: object()),
    )

    fake_inference = types.ModuleType("gcs.inference")
    fake_inference.GCSInference = type("GCSInference", (), {})

    fake_olm = types.ModuleType("gcs.online_learning_module")
    fake_olm.OnlineLearningModule = type("OnlineLearningModule", (), {})

    fake_feedback = types.ModuleType("gcs.feedback_detector")
    fake_feedback.AdaptiveFeedbackDetector = type("AdaptiveFeedbackDetector", (), {})

    sys.modules["numpy"] = fake_numpy
    sys.modules["tensorflow"] = fake_tf
    sys.modules["gcs.inference"] = fake_inference
    sys.modules["gcs.online_learning_module"] = fake_olm
    sys.modules["gcs.feedback_detector"] = fake_feedback

    module = importlib.import_module(module_name)
    return module.ClosedLoopAgent


def build_agent(clock, *, should_fail=False):
    closed_loop_cls = load_closed_loop_agent_class()
    agent = closed_loop_cls.__new__(closed_loop_cls)
    agent.config = {
        "neuromodulation": {
            "enabled": True,
            "default_target_nerve": "vagus",
            "available_modalities": ["ultrasound"],
            "ultrasound_params": {"duration_s": 1.0, "intensity": 0.5},
        }
    }
    agent.is_running = False
    agent.mod_controller = FakeController(should_fail=should_fail)
    agent.consent_gate = InterventionConsentGate(
        request_ttl_seconds=60,
        decline_cooldown_seconds=300,
        defer_cooldown_seconds=300,
        revoke_cooldown_seconds=300,
        time_fn=clock.time,
    )
    agent.session_history = deque(maxlen=25)
    agent.inference_engine = types.SimpleNamespace(
        predict=lambda _source: [{"label": "PAIN_SIGNATURE", "confidence": 0.95}]
    )
    agent._run_affective_inference = lambda _live_data: (2.0, 8.5)
    return agent


class TestInterventionConsentGate(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.gate = InterventionConsentGate(
            request_ttl_seconds=10,
            decline_cooldown_seconds=30,
            defer_cooldown_seconds=40,
            revoke_cooldown_seconds=50,
            time_fn=self.clock.time,
        )

    def _create_request(self):
        proposal = self.gate.propose("ultrasound", {"duration_s": 1.0, "intensity": 0.5}, "vagus")
        self.assertTrue(proposal["created"])
        return proposal["request"]["request_id"]

    def test_valid_explicit_approval_executes_exactly_once(self):
        request_id = self._create_request()
        decision = self.gate.record_decision(request_id, "accept", "user-123")
        self.assertEqual(decision["status"], "approved")

        authorized = self.gate.authorize_execution(
            request_id=request_id,
            modality="ultrasound",
            params={"intensity": 0.5, "duration_s": 1.0},
            target="vagus",
        )
        self.assertEqual(authorized["request_id"], request_id)

        with self.assertRaises(ConsentExecutionError):
            self.gate.authorize_execution(
                request_id=request_id,
                modality="ultrasound",
                params={"duration_s": 1.0, "intensity": 0.5},
                target="vagus",
            )

    def test_request_id_mismatch_fails(self):
        request_id = self._create_request()
        with self.assertRaises(ConsentDecisionError):
            self.gate.record_decision("wrong-request", "accept", "user-123")

        self.gate.record_decision(request_id, "accept", "user-123")
        with self.assertRaises(ConsentExecutionError):
            self.gate.authorize_execution(
                request_id="wrong-request",
                modality="ultrasound",
                params={"duration_s": 1.0, "intensity": 0.5},
                target="vagus",
            )

    def test_parameter_target_and_modality_mismatch_fail(self):
        request_id = self._create_request()
        self.gate.record_decision(request_id, "accept", "user-123")

        with self.assertRaises(ConsentExecutionError):
            self.gate.authorize_execution(
                request_id=request_id,
                modality="electrical",
                params={"duration_s": 1.0, "intensity": 0.5},
                target="vagus",
            )

        with self.assertRaises(ConsentExecutionError):
            self.gate.authorize_execution(
                request_id=request_id,
                modality="ultrasound",
                params={"duration_s": 2.0, "intensity": 0.5},
                target="vagus",
            )

        with self.assertRaises(ConsentExecutionError):
            self.gate.authorize_execution(
                request_id=request_id,
                modality="ultrasound",
                params={"duration_s": 1.0, "intensity": 0.5},
                target="trigeminal",
            )

    def test_expired_declined_deferred_and_revoked_requests_fail_execution(self):
        expired_id = self._create_request()
        self.gate.record_decision(expired_id, "accept", "user-123")
        self.clock.advance(11)
        with self.assertRaises(ConsentExecutionError):
            self.gate.authorize_execution(
                request_id=expired_id,
                modality="ultrasound",
                params={"duration_s": 1.0, "intensity": 0.5},
                target="vagus",
            )

        self.clock.advance(100)
        declined_id = self._create_request()
        self.gate.record_decision(declined_id, "decline", "user-123")
        with self.assertRaises(ConsentExecutionError):
            self.gate.authorize_execution(
                request_id=declined_id,
                modality="ultrasound",
                params={"duration_s": 1.0, "intensity": 0.5},
                target="vagus",
            )

        self.clock.advance(100)
        deferred_id = self._create_request()
        self.gate.record_decision(deferred_id, "defer", "user-123")
        with self.assertRaises(ConsentExecutionError):
            self.gate.authorize_execution(
                request_id=deferred_id,
                modality="ultrasound",
                params={"duration_s": 1.0, "intensity": 0.5},
                target="vagus",
            )

        self.clock.advance(100)
        revoked_id = self._create_request()
        self.gate.record_decision(revoked_id, "accept", "user-123")
        self.gate.record_decision(revoked_id, "revoke", "user-123")
        with self.assertRaises(ConsentExecutionError):
            self.gate.authorize_execution(
                request_id=revoked_id,
                modality="ultrasound",
                params={"duration_s": 1.0, "intensity": 0.5},
                target="vagus",
            )

    def test_cooldown_suppresses_repeated_proposal_creation(self):
        request_id = self._create_request()
        self.gate.record_decision(request_id, "decline", "user-123")

        suppressed = self.gate.propose("ultrasound", {"duration_s": 1.0, "intensity": 0.5}, "vagus")
        self.assertFalse(suppressed["created"])
        self.assertEqual(suppressed["reason"], "cooldown")

        self.clock.advance(31)
        created = self.gate.propose("ultrasound", {"duration_s": 1.0, "intensity": 0.5}, "vagus")
        self.assertTrue(created["created"])


class TestClosedLoopConsentIntegration(unittest.TestCase):
    def test_policy_recommendation_does_not_invoke_hardware(self):
        clock = FakeClock()
        agent = build_agent(clock)

        agent.run_cycle({"source_eeg": FakeNDArray(), "adj_matrix": FakeNDArray()})

        self.assertEqual(agent.mod_controller.calls, [])
        request = agent.get_pending_consent_request()
        self.assertIsNotNone(request)
        self.assertEqual(request["status"], "pending_consent")

    def test_decline_cooldown_suppresses_reprompt_in_run_cycle(self):
        clock = FakeClock()
        agent = build_agent(clock)
        live_data = {"source_eeg": FakeNDArray(), "adj_matrix": FakeNDArray()}

        agent.run_cycle(live_data)
        request = agent.get_pending_consent_request()
        agent.record_consent_decision(request["request_id"], "decline", "user-123")
        agent.run_cycle(live_data)

        proposed_events = [entry for entry in agent.session_history if entry.get("event") == "consent_proposed"]
        self.assertEqual(len(proposed_events), 1)
        self.assertEqual(agent.consent_gate.get_current_request()["status"], "declined")

    def test_post_intervention_feedback_does_not_authorize_new_action(self):
        clock = FakeClock()
        agent = build_agent(clock)
        agent.run_cycle({"source_eeg": FakeNDArray(), "adj_matrix": FakeNDArray()})
        request = agent.get_pending_consent_request()

        agent.record_consent_decision(request["request_id"], "accept", "user-123")
        agent.execute_approved_request(request["request_id"])
        feedback = agent.record_intervention_feedback(request["request_id"], "helpful ", "user-123")

        self.assertEqual(feedback["request_id"], request["request_id"])
        self.assertEqual(feedback["feedback"], "helpful")
        self.assertEqual(len(agent.mod_controller.calls), 1)
        self.assertIsNone(agent.get_pending_consent_request())
        with self.assertRaises(ConsentExecutionError):
            agent.execute_approved_request(request["request_id"])

    def test_hardware_failure_is_surfaced_and_not_automatically_retried(self):
        clock = FakeClock()
        agent = build_agent(clock, should_fail=True)
        agent.run_cycle({"source_eeg": FakeNDArray(), "adj_matrix": FakeNDArray()})
        request = agent.get_pending_consent_request()
        agent.record_consent_decision(request["request_id"], "accept", "user-123")

        with self.assertRaises(RuntimeError):
            agent.execute_approved_request(request["request_id"])

        self.assertEqual(len(agent.mod_controller.calls), 1)
        with self.assertRaises(ConsentExecutionError):
            agent.execute_approved_request(request["request_id"])
        self.assertEqual(len(agent.mod_controller.calls), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
