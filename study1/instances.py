"""Seeded materials. The answer key is never passed to a model backend."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
from pathlib import Path

from scenarios.simpilot.tools import ORIENTATION_STORE
from scenarios.simpilot.world import Aircraft, Tanker, World

SEED = 20261007
TEST_N = 64
HELDOUT_N = 32


def make_world(instance):
    spec = instance["world"]
    w = World.sample(instance["transmission"], channel_state=spec["channel_state"],
                     jamming_inject_scheduled=spec["jamming_inject_scheduled"],
                     trainee_level=spec["trainee_level"],
                     deviation_inject_scheduled=spec["deviation_inject_scheduled"])
    w.aircraft["aircraft 2"] = Aircraft(**spec["aircraft"])
    w.tanker = Tanker(**spec["tanker"])
    return w


def orientation(instance):
    out = copy.deepcopy(ORIENTATION_STORE)
    out.update(instance["orientation"])
    return out


def make_instance(rng, split, index):
    # Balanced address and communication strata; order shuffled after generation.
    address_kind = ("direct", "resolved_ambiguity", "misaddressed", "direct")[index % 4]
    prefix = {"direct": rng.choice(["Aircraft 2", "Aircraft two"]),
              "resolved_ambiguity": "Aircraft two or three", "misaddressed": "Aircraft 3"}[address_kind]
    distance = rng.choice([10, 15, 20, 25, 30])
    phrasing = rng.choice([
        "make your way to the fuel tanker, contact me when you are {d} nautical miles out",
        "join the tanker and report at {d} NM",
        "proceed to the tanker; call control when {d} nautical miles from it",
        "fly to the fuel tanker and advise control at {d} NM",
    ])
    channel = ("clear", "in_use", "jammed", "jammed")[(index // 4) % 4]
    scheduled = channel == "jammed" and (index // 4) % 4 == 3
    trainee = rng.choice(["initial_employment", "lfe_proficient"])
    deviation = rng.choice([False, True])
    # Two plans can disagree for identical observations: intent must key a policy.
    plan_id = rng.choice(["basic_join", "recover_communications"])
    primary = rng.choice(["Channel 5", "Channel 6"])
    backup = rng.choice(["Channel 7", "Channel 9"])
    radius = rng.uniform(6, 10)
    tanker = dict(callsign=rng.choice(["SHELL 1", "SHELL 2"]), cx=rng.uniform(50, 80),
                  cy=rng.uniform(35, 60), radius_nm=radius, speed_kts=rng.uniform(260, 300),
                  angle_rad=rng.uniform(0, 2 * math.pi))
    aircraft = dict(callsign="aircraft 2", x=rng.uniform(-5, 5), y=rng.uniform(-5, 5),
                    heading_deg=90.0, speed_kts=rng.uniform(420, 480))
    instance = {
        "id": f"{split}-{index:03d}", "split": split, "address_kind": address_kind,
        "transmission": f"{prefix}, " + phrasing.format(d=distance) + ".",
        "world": dict(channel_state=channel, jamming_inject_scheduled=scheduled,
                      trainee_level=trainee, deviation_inject_scheduled=deviation,
                      aircraft=aircraft, tanker=tanker),
        "orientation": {
            "own_callsigns": ["aircraft 1", "aircraft 2"],
            "comms_plan": {"primary": primary, "backup": backup},
            "briefing_material": {"tanker_callsign": tanker["callsign"],
                                  "orbit": f"{radius:.2f} NM circular orbit, {tanker['speed_kts']:.2f} kts"},
            "exercise_plan": {"plan_id": plan_id, "trainee_level": trainee},
            "exercise_injects": [{"type": "jamming", "channel": primary}] if scheduled else [],
        },
        "white_force_plan": {
            "deviation": "misinterpret" if deviation and trainee == "lfe_proficient" else "comply",
            "jammed_call": "deferred" if plan_id == "recover_communications" else "made",
        },
        "answer_key": {"ACK": {"addressee": "aircraft 3" if address_kind == "misaddressed" else "aircraft 2",
                               "instruction": "join_tanker", "report_at_nm": float(distance),
                               "acknowledged": address_kind != "misaddressed"}},
    }
    return instance


def generate(test_n=TEST_N, heldout_n=HELDOUT_N, seed=SEED):
    materials = []
    for split, count, split_seed in [("heldout", heldout_n, seed + 1), ("test", test_n, seed + 2)]:
        rng = random.Random(split_seed)
        rows = [make_instance(rng, split, i) for i in range(count)]
        rng.shuffle(rows)
        materials.extend(rows)
    return {"protocol_version": "1.0", "seed": seed, "test_n": test_n,
            "heldout_n": heldout_n, "instances": materials}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    material = generate()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise SystemExit("Refusing to overwrite frozen materials")
    args.out.write_text(json.dumps(material, indent=2) + "\n")
    args.out.with_suffix(".sha256").write_text(digest(material) + "\n")
    print(f"Wrote {len(material['instances'])} synthetic instances; sha256={digest(material)}")


if __name__ == "__main__":
    main()
