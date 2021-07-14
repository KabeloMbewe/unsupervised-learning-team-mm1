import datetime
import time
from multiprocessing import cpu_count
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pandas.api.types import CategoricalDtype
from scipy.sparse import csr_matrix


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


def runParallel(funcs: List[Callable]):
    jobs = min(len(funcs), cpu_count())
    Parallel(n_jobs=jobs)(delayed(func)() for func in funcs)


def applyParallel(grouped, func):
    results = Parallel(n_jobs=cpu_count())(delayed(func)(group) for _, group in grouped)
    return pd.concat(results)


def sparse_pivot(df, index, columns, values) -> csr_matrix:
    index_c = CategoricalDtype(sorted(df[index].unique()), ordered=True)
    colum_c = CategoricalDtype(sorted(df[columns].unique()), ordered=True)

    row = df[index].astype(index_c).cat.codes
    col = df[columns].astype(colum_c).cat.codes

    return csr_matrix(
        (df[values], (row, col)),
        shape=(index_c.categories.size, colum_c.categories.size),
    )
