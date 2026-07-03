import re
from collections import Counter

import pandas as pd

STOPWORDS = {
    "관한",
    "대한",
    "하여",
    "하며",
    "하고",
    "한다",
    "하였다",
    "있다",
    "있는",
    "없는",
    "없다",
    "한다",
    "경우",
    "여부",
    "판단",
    "사건",
    "원고",
    "피고",
    "법원",
    "따라",
    "통해",
    "이를",
    "이유",
    "피해자를",
    "피해자",
    "것으로",
    "그리고",
    "또는",
}


def extract_issue_keywords(results: pd.DataFrame, limit: int = 10) -> list[tuple[str, int]]:
    text = " ".join((results["issues"] + " " + results["summary"]).tolist())
    words = re.findall(r"[가-힣A-Za-z]{2,}", text)
    normalized = [word.lower() for word in words if word.lower() not in STOPWORDS]
    return Counter(normalized).most_common(limit)


def category_counts(results: pd.DataFrame) -> pd.DataFrame:
    counts = results["category"].replace("", "미분류").value_counts()
    return counts.rename_axis("category").reset_index(name="count")
