from typing import NamedTuple

from sinri.drawer.BatchDataBase import BatchDataBase
from sinri.drawer.BatchDataExt import BatchDataExt


class BatchData(NamedTuple):
    return_latents: bool
    base: BatchDataBase
    ext: BatchDataExt
