import json, os, time
from vllm_coldstart import phase_timer


class FakeWorker:
    def load_model(self, *, load_dummy_weights=False):
        time.sleep(0.01)
        return "ok"


def test_wrap_records_phase(tmp_path):
    out = str(tmp_path / "p.jsonl")
    phase_timer._wrap(FakeWorker, "load_model", out)
    assert FakeWorker().load_model() == "ok"
    rec = json.loads(open(out).read().strip())
    assert rec["phase"] == "load_model" and rec["secs"] >= 0.01


def test_wrap_is_idempotent(tmp_path):
    out = str(tmp_path / "p.jsonl")
    phase_timer._wrap(FakeWorker, "load_model", out)
    f1 = FakeWorker.load_model
    phase_timer._wrap(FakeWorker, "load_model", out)
    assert FakeWorker.load_model is f1
