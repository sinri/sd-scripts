from typing import NamedTuple, Any


class BatchDataBase(NamedTuple):
    # バッチ分割が必要ないデータ
    step: int
    prompt: str
    negative_prompt: str
    seed: int
    init_image: Any
    mask_image: Any
    clip_prompt: str
    guide_image: Any