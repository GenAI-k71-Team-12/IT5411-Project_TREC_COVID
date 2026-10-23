"""Entry point thống nhất cho mọi bước.

Chưa cài đặt. Dự kiến:

    python -m trec_covid build-docstore
    python -m trec_covid build-index --branch bm25
    python -m trec_covid build-index --branch dense
    python -m trec_covid search --query "does remdesivir shorten recovery"
    python -m trec_covid make-run --config configs/fusion.yaml
    python -m trec_covid evaluate --run runs/rrf.txt
"""
