"""Protocol v1.1 fixes found by the held-out verification on 7 October 2026.

Every model input must carry what its scoring requires, and descriptive text
fields are scored on their content, not their wording.
"""
from study1.calibrate import clean_contexts
from study1.harness import ENUM_VALUES, task_models
from study1.instances import generate
from study1.oracle import CONTROL_CALLSIGN, gold, score
from dcarr_runtime.handoff import validate


def heldout_with_made_call():
    for instance in generate()["instances"]:
        if instance["split"] == "heldout" and gold(instance).get("CALL", {}).get("call_status") == "made":
            return instance
    raise AssertionError("no held-out instance with a made call")


def test_call_input_carries_the_control_callsign_and_the_allowed_statuses():
    instance = heldout_with_made_call()
    context = clean_contexts(instance)["CALL"]
    assert context["orientation"]["comms_plan"]["control_callsign"] == CONTROL_CALLSIGN
    assert context["hand_off_allowed_values"]["call_status"] == ["made", "deferred"]
    assert CONTROL_CALLSIGN not in context["decision_rule"]  # the name comes from data, not the instruction


def test_a_correct_made_call_can_now_be_scored_correct_from_its_inputs():
    instance = heldout_with_made_call()
    expected = gold(instance)
    context = clean_contexts(instance)["CALL"]
    distance = expected["ACK"]["report_at_nm"]
    frequency = expected["CALL"]["frequency"]
    payload = {"call_status": "made", "frequency": frequency,
               "message": f"{context['orientation']['comms_plan']['control_callsign']}, aircraft 2, {distance:g} NM from the tanker."}
    assert score("CALL", payload, expected["CALL"], instance)


def test_off_list_enum_values_fail_validation():
    models = task_models()
    for task, field, allowed in ENUM_VALUES:
        models.get(task).hand_off.enum_values[field] = list(allowed)
    spec = models.get("CALL").hand_off
    good = {"call_status": "deferred", "message": "Hold.", "frequency": "Channel 5"}
    assert not validate(good, spec)
    assert any("not one of" in p for p in validate({**good, "call_status": "pending"}, spec))


def test_loc_and_cfg_are_scored_on_content_not_wording():
    instance = heldout_with_made_call()
    expected = gold(instance)
    loc = expected["LOC"]
    radius, speed = loc["orbit"].split(" NM")[0], loc["orbit"].split(", ")[1].split(" kts")[0]
    paraphrase = {"tanker": loc["tanker"].lower(), "tanker_found": True, "orbit": f"circular, r={radius} NM at {speed} kts"}
    assert score("LOC", paraphrase, loc, instance)
    wrong = {**paraphrase, "orbit": f"circular, r={float(radius) + 1:.2f} NM at {speed} kts"}
    assert not score("LOC", wrong, loc, instance)
    tanker = loc["tanker"]
    assert score("CFG", {"tool_ready": True, "line": f"Aircraft two to {tanker}"}, expected["CFG"], instance)
    assert not score("CFG", {"tool_ready": True, "line": "aircraft 1 to the tanker"}, expected["CFG"], instance)
    assert not score("CFG", {"tool_ready": False, "line": f"aircraft 2 to {tanker}"}, expected["CFG"], instance)


def test_a_wrongly_acknowledged_misaddressed_call_reaches_dev_without_a_key_error():
    from study1.harness import StudyPerson
    instance = next(i for i in generate()["instances"] if i["address_kind"] == "misaddressed")
    person = StudyPerson(instance, policies_enabled=False)
    answer = person.decide({"task_id": "DEV", "situation": {}})
    assert answer.payload["choice"] == instance["white_force_plan"]["deviation"]
    import pytest
    with pytest.raises(ValueError, match="no reference recovery for LOC"):
        person.decide({"task_id": "LOC", "situation": {}})


def test_parser_reads_the_first_complete_object_and_ignores_trailing_text():
    from dcarr_runtime.executors import _parse_json_object
    held_out_reply = '{"decided":true,"payload":{"goal_met":false},"confidence":0.95,"framings":[]}}'
    assert _parse_json_object(held_out_reply)["payload"] == {"goal_met": False}
    assert _parse_json_object('Here it is: {"decided": false} Thanks.')["decided"] is False
    assert _parse_json_object('{"decided": true, "payload": ')["reason"] == "reply was not valid JSON"
    assert _parse_json_object('{"confidence": NaN}')["reason"] == "reply was not valid JSON"
    assert _parse_json_object("no object")["reason"] == "reply was not a JSON object"


def test_exported_schema_matches_the_metamodel():
    import json
    from pathlib import Path
    from dcarr_runtime.metamodel import json_schema
    exported = json.loads((Path(__file__).parents[1] / "schema" / "task_model_set.schema.json").read_text())
    assert exported == json_schema()


def test_parser_skips_prose_with_a_stray_brace_and_prefers_an_answer_object():
    from dcarr_runtime.executors import _parse_json_object
    reply = 'explanation {not JSON} then {"decided":true,"payload":{},"confidence":1}'
    assert _parse_json_object(reply) == {"decided": True, "payload": {}, "confidence": 1}
    # A malformed outer answer does not turn its nested payload into an answer.
    nested = '{"decided": true, "payload": {"acknowledged": true}, oops}'
    assert "decided" not in _parse_json_object(nested)
    # The single-model answer, keyed by task, has no "decided" field and is still read.
    assert _parse_json_object('{"ACK": {"acknowledged": false}, "FLY": null}')["ACK"] == {"acknowledged": False}


def test_loc_rejects_swapped_quantities_and_cfg_rejects_a_longer_tanker_name():
    instance = {i["id"]: i for i in generate()["instances"]}["heldout-004"]
    expected = gold(instance)
    swapped = {"tanker": expected["LOC"]["tanker"], "tanker_found": True, "orbit": "Radius 284.08 NM; speed 7.06 knots"}
    assert not score("LOC", swapped, expected["LOC"], instance)
    assert not score("CFG", {"tool_ready": True, "line": "aircraft 2 to SHELL 20"}, expected["CFG"], instance)


def test_diagnostics_reject_an_invented_enum_value_like_the_runtime_does():
    from study1.calibrate import diagnostic_row

    class Fake:
        last_rows = []

        def start(self, run_id):
            self.last_rows = [{"empirical": False}]

        def call(self, tier, context, mode):
            payload = {"acknowledged": True, "addressee": "aircraft 2", "instruction": "invented_enum", "report_at_nm": 25.0}
            return {"decided": True, "payload": payload, "confidence": 0.99}, 1, 1

    instance = next(i for i in generate()["instances"] if i["split"] == "heldout")
    context = clean_contexts(instance)["ACK"]
    row = diagnostic_row(("probe", instance, "ACK", context, 0, "T1"), Fake())
    assert row["handoff_valid"] is False and row["candidate_correct"] is False
