# 0010: Tuning fusion bằng 5-fold CV theo topic, chọn theo nDCG@10 condensed

- **Ngày:** 2026-11-05
- **Người:** Người 3 (Fusion · Rerank · Đánh giá)
- **Trạng thái:** chấp nhận (số liệu đo với run dense tạm, cần chạy lại khi có run của Người 2)

## Bối cảnh

Weighted RRF có `w`, Convex có `α`. Cả hai cần "tập dev", nhưng cuộc thi chỉ có 35 topic
có qrels và không có tập dev riêng. Có hai vấn đề:

1. **Học trên tập chấm.** Chọn tham số trên 35 topic rồi báo điểm trên chính 35 topic đó
   là điểm bị thổi lên, và rất dễ bị hỏi vặn khi bảo vệ.
2. **qrels thưa và lệch pool.** `qrels.csv` chỉ có phán định vòng 1–2 (ADR 0008). Tài liệu
   chưa chấm bị coi là không liên quan. Đo trên run tạm ngày 2026-11-05:

| run | nDCG@10 | judged@10 | nDCG@10 condensed | bpref |
|---|---|---|---|---|
| bm25 | 0.2155 | 0.43 | 0.4601 | 0.2648 |
| dense (tạm) | 0.1387 | 0.21 | 0.5943 | 0.2960 |

   Theo nDCG@10 thường, BM25 hơn hẳn dense. Nhưng chỉ 21% top-10 của dense có nhãn: dense
   tìm ra nhiều bài mà pool vòng 1–2 chưa từng thấy, nên bị phạt oan. Bỏ bài chưa chấm
   rồi chấm lại (run "condensed", Sakai 2007) thì thứ tự đảo ngược. Tuning theo nDCG@10
   thường chọn `w_bm25 = 1.0`, tức là **bỏ hẳn nhánh dense**. Đó là hệ quả của pool, không
   phải của chất lượng.

## Phương án đã cân nhắc

| Phương án | Ưu | Nhược |
|---|---|---|
| Tune trên cả 35 topic, báo luôn số đó | đơn giản | số bị thổi lên |
| Chia đôi topic (lẻ tune, chẵn báo) | không rò rỉ | mỗi nửa ~17 topic, rất nhiễu |
| **5-fold CV theo topic** | không rò rỉ, dùng hết 35 topic | chạy lưới 5 lần (~5 giây) |
| Chọn theo nDCG@10 thường | đúng độ đo chính của TREC-COVID | thiên vị BM25 vì pool, chọn bỏ dense |
| **Chọn theo nDCG@10 condensed** | công bằng giữa các nhánh khi qrels thưa | không phải số chính thức; bỏ qua việc trả bài chưa chấm |

## Quyết định

- Tuning bằng **5-fold CV theo topic** (`evaluate/cv.py`, seed 42). Cột `ndcg10_cv5` là số
  báo cáo cho công thức có tham số. Giá trị ghi vào `configs/fusion.yaml` là tham số tốt
  nhất trên cả 35 topic. Hoà điểm thì chọn tham số nhỏ hơn để kết quả tất định.
- Độ đo để **chọn** tham số và công thức là **nDCG@10 condensed** (`--metric condensed`,
  mặc định). nDCG@10 thường vẫn là cột chính thức trong mọi bảng, kèm judged@10 và bpref
  để người đọc thấy độ lệch.
- RRF chuẩn giữ `k = 60`, không tune.

## Kết quả (2026-11-05, run dense tạm)

`results/2026-11-05_fusion-comparison.csv` (mọi run cắt top-1000 theo hợp đồng):

| run | nDCG@10 | condensed | CV 5-fold | bpref | tham số |
|---|---|---|---|---|---|
| bm25 | 0.2155 | 0.4601 | — | 0.2648 | |
| dense | 0.1387 | 0.5943 | — | 0.2960 | |
| rrf | 0.2068 | 0.5730 | 0.5730 | 0.3248 | k = 60 |
| weighted_rrf | 0.1869 | 0.6019 | 0.5899 | 0.3288 | w_bm25 = 0.4 |
| convex | 0.1801 | 0.6014 | **0.5937** | **0.3380** | α = 0.3 |

`fusion.active = convex`. Cần nói thẳng: trên nDCG@10 condensed, fusion chỉ **ngang** dense
đơn lẻ (0.5937 so với 0.5943; thắng 13 topic, thua 18). Fusion chỉ hơn rõ ở bpref và MAP.
Cổng "vượt cả hai baseline" của ROADMAP Mốc 2 **chưa đạt** với run tạm này.

## Hệ quả

- Chạy lại `scripts/07_fusion_experiments.py` khi có run dense chính thức (Người 2). Tham
  số và cả kết luận ở trên có thể đổi.
- Chương thực nghiệm phải trình bày cả hai độ đo và giải thích pool bias. Đây cũng là câu
  trả lời cho câu hỏi phản biện "nDCG@10 = x nghĩa là gì".
- Muốn chọn theo nDCG@10 thường thì chạy `--metric raw` và sửa config. ADR này phải được
  thay bằng ADR mới.
