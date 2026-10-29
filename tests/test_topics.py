"""Test đọc topic của cuộc thi — đầu vào của mọi phép đo."""

from __future__ import annotations

import pandas as pd
import pytest

from trec_covid.io.topics import TOPIC_FIELDS, Topic, build_queries, load_topics

ROWS = [
    {
        "topic_id": 2,
        "query": "coronavirus remdesivir",
        "question": "is remdesivir effective",
        "narrative": "Seeking evidence about remdesivir",
    },
    {
        "topic_id": 1,
        "query": "coronavirus origin",
        "question": "what is the origin of COVID-19",
        "narrative": "Seeking information about the origin",
    },
    {
        "topic_id": 3,
        "query": "coronavirus social distancing",
        "question": "how does distancing work",
        "narrative": "Seeking information about distancing",
    },
]


def _write(tmp_path, rows, name="topics.csv", encoding="utf-8"):
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, index=False, encoding=encoding)
    return path


class TestLoadTopics:
    def test_doc_du_ba_truong(self, tmp_path):
        topics = load_topics(_write(tmp_path, ROWS))
        assert len(topics) == 3
        assert isinstance(topics[0], Topic)
        assert topics[0].query == "coronavirus origin"

    def test_sap_tang_dan_theo_topic_id(self, tmp_path):
        assert [t.topic_id for t in load_topics(_write(tmp_path, ROWS))] == [1, 2, 3]

    def test_topic_id_la_so_nguyen(self, tmp_path):
        assert all(isinstance(t.topic_id, int) for t in load_topics(_write(tmp_path, ROWS)))

    def test_doc_duoc_file_co_bom(self, tmp_path):
        # Export từ Windows/Kaggle thường có BOM; đọc utf-8 thuần sẽ hỏng tên cột.
        topics = load_topics(_write(tmp_path, ROWS, encoding="utf-8-sig"))
        assert len(topics) == 3

    def test_gioi_han_so_topic(self, tmp_path):
        assert len(load_topics(_write(tmp_path, ROWS), n_topics=2)) == 2

    def test_thieu_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_topics(tmp_path / "khong_co.csv")

    def test_thieu_cot_topic_id(self, tmp_path):
        path = _write(tmp_path, [{"query": "x"}], name="bad.csv")
        with pytest.raises(ValueError, match="topic_id"):
            load_topics(path)

    def test_thieu_cot_query(self, tmp_path):
        path = _write(tmp_path, [{"topic_id": 1}], name="bad.csv")
        with pytest.raises(ValueError, match="query"):
            load_topics(path)

    def test_file_rong_thi_raise(self, tmp_path):
        path = _write(tmp_path, [{"topic_id": 1, "query": "x"}], name="one.csv")
        df = pd.read_csv(path).iloc[0:0]
        df.to_csv(path, index=False)
        with pytest.raises(ValueError, match="không có topic"):
            load_topics(path)

    def test_thieu_cot_narrative_thi_rong(self, tmp_path):
        path = _write(tmp_path, [{"topic_id": 1, "query": "x"}], name="short.csv")
        assert load_topics(path)[0].narrative == ""

    def test_alias_ten_cot(self, tmp_path):
        rows = [{"topic": 7, "title": "coronavirus vaccine", "description": "d"}]
        topics = load_topics(_write(tmp_path, rows, name="alias.csv"))
        assert topics[0].topic_id == 7
        assert topics[0].query == "coronavirus vaccine"
        assert topics[0].narrative == "d"


class TestTopic:
    def test_text_theo_truong(self):
        t = Topic(topic_id=1, query="a", question="b", narrative="c")
        assert [t.text(f) for f in TOPIC_FIELDS] == ["a", "b", "c"]

    def test_text_strip(self):
        assert Topic(topic_id=1, query="  a  ").text() == "a"

    def test_truong_la_thi_raise(self):
        with pytest.raises(ValueError, match="không hợp lệ"):
            Topic(topic_id=1, query="a").text("narrative_full")


class TestBuildQueries:
    def test_mac_dinh_dung_query(self, tmp_path):
        pairs = build_queries(load_topics(_write(tmp_path, ROWS)))
        assert pairs[0] == (1, "coronavirus origin")

    def test_doi_truong_sang_narrative(self, tmp_path):
        pairs = build_queries(load_topics(_write(tmp_path, ROWS)), field="narrative")
        assert pairs[0][1].startswith("Seeking information")

    def test_bo_topic_co_van_ban_rong(self, tmp_path):
        rows = [
            {"topic_id": 1, "query": "x", "question": "", "narrative": ""},
            {"topic_id": 2, "query": "", "question": "", "narrative": ""},
        ]
        pairs = build_queries(load_topics(_write(tmp_path, rows, name="e.csv")))
        assert [tid for tid, _ in pairs] == [1]
