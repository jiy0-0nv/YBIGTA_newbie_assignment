# 구현하세요!
from datasets import load_dataset
from typing import Optional

def load_corpus() -> list[str]:
    """
    Wikitext-2 코퍼스를 로드하여
    학습 가능한 문장 리스트를 반환하는 함수.

    Returns:
        corpus (list[str]): 빈 문장이 제거된 문장 단위 코퍼스
    """
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    corpus: list[str] = [
        line for line in ds["text"]
        if line is not None and line.strip() != ""
    ]
    # 구현하세요!
    return corpus