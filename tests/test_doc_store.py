"""Test Doc Store — hợp đồng dữ liệu mà cả 4 người cùng dựa vào."""

from __future__ import annotations

import pandas as pd
import pytest

from trec_covid.io.doc_store import (
    ALL_COLUMNS,
    CORE_COLUMNS,
    build_doc_store,
    build_docid_order,
    doc_store_stats,
    iter_texts,
    load_cord19,
    load_doc_store,
    load_docid_order,
    save_doc_store,
    validate_doc_store,
)


class TestLoadCord19:
    def test_doc_dung_cot(self, mini_metadata_csv):
        df = load_cord19(mini_metadata_csv)
        assert list(df.columns) == list(ALL_COLUMNS)
        assert len(df) == 6

    def test_abstract_rong_khong_thanh_nan(self, mini_metadata_csv):
        df = load_cord19(mini_metadata_csv)
        assert not df["abstract"].isna().any()
        assert (df["abstract"] == "").any()

    def test_thieu_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_cord19(tmp_path / "khong_ton_tai.csv")

    def test_thieu_cot_cord_uid(self, tmp_path):
        bad = tmp_path / "bad.csv"
        pd.DataFrame({"title": ["x"]}).to_csv(bad, index=False)
        with pytest.raises(ValueError, match="cord_uid"):
            load_cord19(bad)

    def test_cot_year_duoc_doi_thanh_publish_time(self, tmp_path):
        # Bản CORD-19 cũ dùng `year`; Doc Store chuẩn hoá về `publish_time`.
        path = tmp_path / "old.csv"
        pd.DataFrame(
            {"cord_uid": ["a"], "title": ["T"], "abstract": ["A"], "year": ["2020"]}
        ).to_csv(path, index=False)
        df = load_cord19(path)
        assert "publish_time" in df.columns
        assert df.loc[0, "publish_time"] == "2020"

    def test_chi_lay_cot_yeu_cau(self, tmp_path):
        path = tmp_path / "full.csv"
        pd.DataFrame(
            {
                "cord_uid": ["a"],
                "title": ["T"],
                "abstract": ["A"],
                "sha": ["deadbeef"],
                "license": ["cc-by"],
            }
        ).to_csv(path, index=False)
        df = load_cord19(path, columns=CORE_COLUMNS)
        assert list(df.columns) == list(CORE_COLUMNS)


class TestBuildDocStore:
    def test_du_8_cot(self, mini_doc_store):
        assert list(mini_doc_store.columns) == list(ALL_COLUMNS)

    def test_minimal_chi_3_cot(self, raw_metadata_df):
        df = build_doc_store(raw_metadata_df, minimal=True)
        assert list(df.columns) == list(CORE_COLUMNS)

    def test_khu_trung_cord_uid(self, mini_doc_store):
        assert mini_doc_store["cord_uid"].is_unique
        assert len(mini_doc_store) == 5  # 6 dòng thô, 1 cặp trùng

    def test_giu_ban_co_abstract_day_du_nhat(self, mini_doc_store):
        row = mini_doc_store.loc[mini_doc_store["cord_uid"] == "ug7v899j"].iloc[0]
        assert "lymphopenia" in row["abstract"]

    def test_thu_tu_tat_dinh(self, raw_metadata_df):
        a = build_doc_store(raw_metadata_df)
        b = build_doc_store(raw_metadata_df.sample(frac=1.0, random_state=1))
        assert a["cord_uid"].tolist() == b["cord_uid"].tolist()
        # Xáo trộn đầu vào không được đổi nội dung chọn ra (kể cả khi hoà độ dài)
        pd.testing.assert_frame_equal(a, b)

    def test_hoa_do_dai_abstract_van_chon_tat_dinh(self):
        df = pd.DataFrame(
            {
                "cord_uid": ["x", "x"],
                "title": ["T", "T"],
                "abstract": ["beta", "alfa"],  # cùng độ dài 4
            }
        )
        assert build_doc_store(df).loc[0, "abstract"] == "alfa"

    def test_index_lien_tuc_tu_0(self, mini_doc_store):
        assert mini_doc_store.index.tolist() == list(range(len(mini_doc_store)))

    def test_da_chuan_hoa_nbsp_va_gach_noi(self, mini_doc_store):
        row = mini_doc_store.loc[mini_doc_store["cord_uid"] == "ug7v899j"].iloc[0]
        assert "\u00a0" not in row["abstract"]
        assert "\u2011" not in row["abstract"]
        assert "2019-nCoV" in row["abstract"]

    def test_bo_placeholder_va_ban_quyen(self, mini_doc_store):
        row = mini_doc_store.loc[mini_doc_store["cord_uid"] == "ug7v899j"].iloc[0]
        assert "[Figure 1]" not in row["abstract"]
        assert "Elsevier" not in row["abstract"]

    def test_giu_bai_chi_co_title(self, mini_doc_store):
        assert "a1b2c3d4" in set(mini_doc_store["cord_uid"])
        row = mini_doc_store.loc[mini_doc_store["cord_uid"] == "a1b2c3d4"].iloc[0]
        assert row["abstract"] == ""
        assert row["title"] != ""

    def test_bo_dong_cord_uid_rong(self):
        df = pd.DataFrame(
            {
                "cord_uid": ["a", "", "b"],
                "title": ["T1", "T2", "T3"],
                "abstract": ["A1", "A2", "A3"],
            }
        )
        assert build_doc_store(df)["cord_uid"].tolist() == ["a", "b"]

    def test_rong_hoan_toan_thi_raise(self):
        df = pd.DataFrame({"cord_uid": [""], "title": ["T"], "abstract": ["A"]})
        with pytest.raises(ValueError, match="rỗng"):
            build_doc_store(df)

    def test_thieu_cot_metadata_thi_dien_chuoi_rong(self):
        df = pd.DataFrame({"cord_uid": ["a"], "title": ["T"], "abstract": ["A"]})
        out = build_doc_store(df)
        assert list(out.columns) == list(ALL_COLUMNS)
        assert out.loc[0, "doi"] == ""
        assert out.loc[0, "pmcid"] == ""

    def test_nan_thanh_chuoi_rong(self):
        df = pd.DataFrame(
            {"cord_uid": ["a"], "title": ["T"], "abstract": [None], "journal": [None]}
        )
        out = build_doc_store(df)
        assert out.loc[0, "abstract"] == ""
        assert out.loc[0, "journal"] == ""


class TestValidate:
    def test_doc_store_hop_le(self, mini_doc_store):
        validate_doc_store(mini_doc_store)  # không raise

    def test_thieu_cot(self, mini_doc_store):
        with pytest.raises(ValueError, match="thiếu cột"):
            validate_doc_store(mini_doc_store.drop(columns=["doi"]))

    def test_con_nan(self, mini_doc_store):
        broken = mini_doc_store.copy()
        broken.loc[0, "journal"] = None
        with pytest.raises(ValueError, match="NaN"):
            validate_doc_store(broken)

    def test_trung_cord_uid(self, mini_doc_store):
        broken = pd.concat([mini_doc_store, mini_doc_store.iloc[[0]]], ignore_index=True)
        with pytest.raises(ValueError, match="trùng"):
            validate_doc_store(broken)


class TestRoundTrip:
    def test_ghi_doc_lai_khong_doi(self, mini_doc_store, tmp_path):
        path = save_doc_store(mini_doc_store, tmp_path / "ds.parquet")
        back = load_doc_store(path)
        pd.testing.assert_frame_equal(back, mini_doc_store)

    def test_doc_theo_cot(self, mini_doc_store, tmp_path):
        path = save_doc_store(mini_doc_store, tmp_path / "ds.parquet")
        back = load_doc_store(path, columns=["cord_uid", "title"])
        assert list(back.columns) == ["cord_uid", "title"]

    def test_docid_order_dung_thu_tu(self, mini_doc_store, tmp_path):
        path = tmp_path / "docid_order.parquet"
        build_docid_order(mini_doc_store, path)
        assert load_docid_order(path) == mini_doc_store["cord_uid"].tolist()

    def test_docid_order_position_lien_tuc(self, mini_doc_store):
        order = build_docid_order(mini_doc_store)
        assert order["position"].tolist() == list(range(len(mini_doc_store)))


class TestStats:
    def test_dem_bai_co_abstract(self, mini_doc_store):
        stats = doc_store_stats(mini_doc_store)
        assert stats["n_docs"] == 5
        assert stats["n_with_abstract"] == 4
        assert stats["n_title_only"] == 1
        assert stats["n_unique_cord_uid"] == 5


class TestIterTexts:
    def test_ghep_title_va_abstract_dung_thu_tu(self, mini_doc_store):
        texts = list(iter_texts(mini_doc_store))
        assert len(texts) == len(mini_doc_store)
        first = mini_doc_store.iloc[0]
        assert texts[0].startswith(first["title"])
        assert first["abstract"] in texts[0]

    def test_bai_chi_co_title_khong_co_khoang_trang_thua(self, mini_doc_store):
        texts = dict(zip(mini_doc_store["cord_uid"], iter_texts(mini_doc_store), strict=True))
        assert not texts["a1b2c3d4"].endswith(" ")
