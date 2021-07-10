import datetime
import time
from typing import Callable, List, Tuple


def time_steps(steps: List[Tuple[Callable, ...]]):
    print(f"Timing {len(steps)} steps")

    results = []
    start = time.time()

    for func, *args in steps:
        print("Stepping..")
        if len(args) == 0:
            results.append(func())
        else:
            results.append(func(*args))

        print(round(time.time() - start, 1), "s elapsed")

    duration = round(time.time() - start, 1)
    print(
        f"{len(steps)} steps completed in {str(datetime.timedelta(seconds=duration))}"
    )
    return results


def flatten_list(lst):
    return [item for sublist in lst for item in sublist]
