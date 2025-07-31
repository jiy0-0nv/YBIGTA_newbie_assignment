from datasets import load_dataset
from typing import Optional

def load_corpus(max_sentences: Optional[int] = 5000) -> list[str]:
    """
    Args:
        max_sentences (Optional[int]): 불러올 최대 문장 수

    Returns:
        list[str]: 빈 줄 제거 후 문장 단위로 추출
    """
    # train split에서 앞 max_sentences개 문장만 로드
    split = f"train[:{max_sentences}]" if max_sentences is not None else "train"
    ds = load_dataset("sms_spam", split="train")

    # 빈 문자열/공백만 있는 줄 제거
    corpus = [
        line for line in ds["sms"]
        if line is not None and line.strip() != ""
    ]
    # 구현하세요!
    return corpus