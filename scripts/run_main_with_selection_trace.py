#!/usr/bin/env python
"""Run the unmodified main_fed.py while tracing only FL client selections.

This wrapper is used solely by the 10-round parity check.  It replaces
``np.random.choice`` with a transparent forwarding function and records calls
whose population/size/replace tuple is exactly the fixed diagnostic protocol.
The wrapper itself performs no random draws.
"""

from __future__ import annotations

import hashlib
import json
import runpy
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    if "--selection_trace" not in sys.argv:
        raise SystemExit("--selection_trace PATH is required")
    forwarded = list(sys.argv[1:])

    def pop_option(name: str, required: bool = False):
        if name not in forwarded:
            if required:
                raise SystemExit(f"{name} PATH is required")
            return None
        position = forwarded.index(name)
        try:
            value = forwarded[position + 1]
        except IndexError as exc:
            raise SystemExit(f"{name} PATH is required") from exc
        del forwarded[position : position + 2]
        return value

    trace_path = Path(pop_option("--selection_trace", required=True)).resolve()
    hash_value = pop_option("--global_hash_trace")
    hash_path = Path(hash_value).resolve() if hash_value else None
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]

    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    if hash_path is not None:
        hash_path.parent.mkdir(parents=True, exist_ok=True)
    original_choice = np.random.choice
    selection_index = 0

    def traced_choice(a, size=None, replace=True, p=None):
        nonlocal selection_index
        result = original_choice(a, size=size, replace=replace, p=p)
        try:
            population_size = len(a)
        except TypeError:
            population_size = int(a)
        if (
            population_size == 100
            and size is not None
            and int(size) == 10
            and replace is False
            and p is None
        ):
            selection_index += 1
            with trace_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "selection_index": selection_index,
                            "selected_clients": [int(item) for item in result.tolist()],
                        }
                    )
                    + "\n"
                )
        return result

    np.random.choice = traced_choice
    if hash_path is not None:
        import models.test as test_module

        original_evaluate = test_module.evaluate_round_accuracy

        def traced_evaluate(net_glob, dataset_test, args, round_number):
            digest = hashlib.sha256()
            for name, tensor in net_glob.state_dict().items():
                digest.update(name.encode("utf-8"))
                contiguous = tensor.detach().to("cpu").contiguous()
                digest.update(contiguous.numpy().tobytes())
            with hash_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {"round": int(round_number), "global_state_sha256": digest.hexdigest()}
                    )
                    + "\n"
                )
            return original_evaluate(net_glob, dataset_test, args, round_number)

        test_module.evaluate_round_accuracy = traced_evaluate
    sys.argv = [str(root / "main_fed.py"), *forwarded]
    runpy.run_path(str(root / "main_fed.py"), run_name="__main__")


if __name__ == "__main__":
    main()
