"""Cross-validation k-fold theo topic để tuning tham số mà không học trên tập chấm (ADR 0010).

Chỉ có 35 topic có qrels và không có tập dev riêng. Chọn tham số trên cả 35 topic rồi
báo điểm trên chính 35 topic đó là báo điểm bị thổi lên. Ở đây mỗi fold chọn tham số
tốt nhất trên các fold còn lại rồi chấm trên fold của mình; `cv_mean` là điểm trung thực.
"""

from __future__ import annotations

import random
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass


@dataclass
class CVResult:
    cv_mean: float  # trung bình trên mọi topic, mỗi topic chấm bằng tham số do fold của nó chọn
    best_param_all: Hashable  # tham số tốt nhất trên toàn bộ topic (giá trị ghi vào config)
    fold_params: list[Hashable]  # tham số được chọn ở từng fold


def make_folds(topics: Sequence[str], n_folds: int = 5, seed: int = 42) -> list[list[str]]:
    """Chia topic thành `n_folds` phần tất định: sort theo số, shuffle bằng seed, chia vòng."""
    ordered = sorted(topics, key=int)
    random.Random(seed).shuffle(ordered)
    return [ordered[i::n_folds] for i in range(n_folds)]


def _best(per_topic: Mapping[Hashable, Mapping[str, float]], topics: Sequence[str]) -> Hashable:
    """Tham số có điểm trung bình cao nhất trên `topics`; hoà thì chọn tham số nhỏ nhất."""
    params = sorted(per_topic)
    means = [sum(per_topic[p][t] for t in topics) / len(topics) for p in params]
    return params[means.index(max(means))]


def cross_validate(
    per_topic: Mapping[Hashable, Mapping[str, float]], n_folds: int = 5, seed: int = 42
) -> CVResult:
    """CV theo topic trên bảng `{tham_số: {topic_id: điểm}}`."""
    topic_sets = {frozenset(scores) for scores in per_topic.values()}
    if len(topic_sets) != 1:
        raise ValueError("Mọi tham số phải được chấm trên cùng một tập topic")
    topics = sorted(next(iter(topic_sets)), key=int)

    folds = make_folds(topics, n_folds, seed)
    fold_params: list[Hashable] = []
    test_scores: list[float] = []
    for i, test in enumerate(folds):
        train = [t for j, f in enumerate(folds) if j != i for t in f]
        chosen = _best(per_topic, train)
        fold_params.append(chosen)
        test_scores.extend(per_topic[chosen][t] for t in test)

    return CVResult(
        cv_mean=sum(test_scores) / len(test_scores),
        best_param_all=_best(per_topic, topics),
        fold_params=fold_params,
    )
