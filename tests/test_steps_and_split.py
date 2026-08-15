from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tokenaware.data import load_math_train, stratified_split
from tokenaware.mc import mc_label, select_prefix_states
from tokenaware.scoring import answers_equal, gold_from_solution
from tokenaware.steps import extract_boxed, parse_steps


def test_parse_steps_skips_headers():
    text = (
        "Reasoning Steps:\n"
        "- Step 1: Compute 2+2=4\n"
        "- Step 2: Square it to get 16\n"
        "\\boxed{16}\n"
    )
    steps = parse_steps(text)
    assert [s.text for s in steps] == ["Compute 2+2=4", "Square it to get 16"]
    assert extract_boxed(text) == "16"


def test_gold_from_solution():
    sol = "We have $x=3$.\n\\boxed{3}"
    assert gold_from_solution(sol) == "3"
    assert answers_equal("3", "3")


def test_select_prefix_states_and_mc_label():
    steps = [
        {"char_end": 5, "tokens_remaining_this_trace": 30},
        {"char_end": 10, "tokens_remaining_this_trace": 20},
        {"char_end": 15, "tokens_remaining_this_trace": 10},
        {"char_end": 20, "tokens_remaining_this_trace": 0},
    ]
    rollouts = [
        {"sample_id": i, "text": "x" * 20, "correct": i == 0, "steps": steps}
        for i in range(3)
    ]
    states = select_prefix_states(rollouts)
    assert len(states) == 6
    assert {s["source_sample_id"] for s in states} == {0, 1}
    assert [s["prefix_char_end"] for s in states[:3]] == [5, 10, 15]

    label = mc_label(
        [
            {"correct": True, "n_tokens": 8, "truncated": False},
            {"correct": False, "n_tokens": 12, "truncated": True},
        ]
    )
    assert label["v_mc"] == 0.5
    assert label["t_mc_mean"] == 10
    assert label["n_truncated"] == 1


def test_split_sizes_and_no_leak():
    rows = load_math_train()
    # Official MATH train is 7500; two geometry rows have "Level ?" and are dropped.
    assert 7498 <= len(rows) <= 7500
    split = stratified_split(rows)
    assert len(split["val"]) == 500
    assert len(split["train"]) == 2000
    val_ids = {r["problem_id"] for r in split["val"]}
    train_ids = {r["problem_id"] for r in split["train"]}
    assert val_ids.isdisjoint(train_ids)
    assert set(split["level_counts"]["train"]) == {"1", "2", "3", "4", "5"}
