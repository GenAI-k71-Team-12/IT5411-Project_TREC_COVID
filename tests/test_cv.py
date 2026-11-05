"""Test cross-validation theo topic dùng để tuning tham số fusion (ADR 0010)."""

from __future__ import annotations

import pytest

from trec_covid.evaluate.cv import cross_validate, make_folds

TOPICS = [str(t) for t in range(1, 36)]


def test_moi_topic_thuoc_dung_mot_fold():
    folds = make_folds(TOPICS, n_folds=5, seed=42)
    assert len(folds) == 5
    flat = [t for f in folds for t in f]
    assert sorted(flat, key=int) == TOPICS
    assert {len(f) for f in folds} == {7}


def test_cung_seed_cung_ket_qua_khac_seed_khac_fold():
    assert make_folds(TOPICS, 5, 42) == make_folds(TOPICS, 5, 42)
    assert make_folds(TOPICS, 5, 42) != make_folds(TOPICS, 5, 7)


def test_tham_so_thang_moi_topic_duoc_chon_moi_fold():
    per_topic = {
        0.2: {t: 0.5 for t in TOPICS},
        0.7: {t: 0.6 for t in TOPICS},
    }
    res = cross_validate(per_topic, n_folds=5, seed=42)
    assert res.best_param_all == 0.7
    assert res.fold_params == [0.7] * 5
    assert res.cv_mean == pytest.approx(0.6)


def test_cv_cham_tren_fold_test_khong_phai_fold_train():
    # Tham số A tốt ở mọi topic trừ đúng các topic của fold 0, nơi B tốt hơn hẳn.
    folds = make_folds(TOPICS, 5, 42)
    a = {t: (0.0 if t in folds[0] else 1.0) for t in TOPICS}
    b = {t: (0.9 if t in folds[0] else 0.5) for t in TOPICS}
    res = cross_validate({"A": a, "B": b}, n_folds=5, seed=42)
    # Fold 0 chọn A (từ train), nên chấm A trên test fold 0 -> 0; các fold khác cũng A -> 1.
    assert res.fold_params == ["A"] * 5
    assert res.cv_mean == pytest.approx(28 / 35)


def test_hoa_diem_chon_tham_so_nho_nhat():
    per_topic = {0.9: {t: 0.5 for t in TOPICS}, 0.1: {t: 0.5 for t in TOPICS}}
    assert cross_validate(per_topic).best_param_all == 0.1


def test_tham_so_thieu_topic_thi_raise():
    with pytest.raises(ValueError, match="topic"):
        cross_validate({1: {"1": 0.5}, 2: {"1": 0.5, "2": 0.4}})
