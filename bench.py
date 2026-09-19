"""
bench.py — Benchmark đo lường chất lượng truy xuất và câu trả lời của RAG Agent
Nhóm: RAUMAMIENTAY — Chủ đề: Thư viện UTH
Thành viên: Nguyễn Đức Minh (2A202602891) — Chiến lược: SentenceChunker
"""

from __future__ import annotations

import csv
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.agent import KnowledgeBaseAgent
from src.chunking import (
    FixedSizeChunker,
    RecursiveChunker,
    SentenceChunker,
    compute_similarity,
)
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    GEMINI_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    GeminiEmbedder,
    MockEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)
from src.models import Document
from src.store import EmbeddingStore

# ==============================================================================
# CẤU HÌNH CHIẾN LƯỢC VÀ DỮ LIỆU
# ==============================================================================
# Chiến lược của Nguyễn Đức Minh: "sentence"
STRATEGY = "sentence"
DATA_DIR = Path("data/thu-vien-uth")

# 5 CÂU HỎI BENCHMARK THỐNG NHẤT CỦA NHÓM RAUMAMIENTAY
QUERIES = [
    {
        "id": 1,
        "query": "Trả sách quá hạn thì bị phạt bao nhiêu tiền?",
        "gold_doc": "phuc-vu-muon-tra-tai-lieu",
        "gold_snippet": "1000 đồng",
        "filter": None,
    },
    {
        "id": 2,
        "query": "Sách mượn về nhà được gia hạn mấy lần, mỗi lần bao lâu?",
        "gold_doc": "phuc-vu-muon-tra-tai-lieu",
        "gold_snippet": "01 lần với thời gian 45 ngày",
        "filter": None,
    },
    {
        "id": 3,
        "query": "Mỗi bạn đọc được mượn tối đa bao nhiêu tài liệu về nhà?",
        "gold_doc": "quy-dinh-muon-tra-sinh-vien",
        "gold_snippet": "05 tài liệu",
        "filter": {"audience": "student"},
    },
    {
        "id": 4,
        "query": "Khi vào phòng đọc được mang theo những gì?",
        "gold_doc": "noi-quy-thu-vien",
        "gold_snippet": "máy tính cá nhân",
        "filter": None,
    },
    {
        "id": 5,
        "query": "Muốn kiểm tra tỉ lệ trùng lặp cho khóa luận, đồ án thì dùng dịch vụ nào?",
        "gold_doc": "dich-vu-quet-trung-lap",
        "gold_snippet": "Turnitin",
        "filter": None,
    },
]

# 5 CẶP CÂU DỰ ĐOÁN COSINE SIMILARITY
SIMILARITY_PAIRS = [
    (
        "Tôi muốn mượn sách về nhà.",
        "Tôi muốn gia hạn sách đã mượn.",
        "cao",
    ),
    (
        "Thời hạn mượn sách của sinh viên là bao lâu?",
        "Mức phạt quá hạn mượn sách là bao nhiêu?",
        "cao",
    ),
    (
        "Giảng viên được mượn 10 tài liệu.",
        "Sinh viên được mượn 5 tài liệu.",
        "cao",
    ),
    (
        "Tôi muốn mượn sách.",
        "Tôi không muốn mượn sách.",
        "cao",
    ),
    (
        "Quy định phòng đọc thư viện.",
        "Công thức nấu món phở bò truyền thống.",
        "thấp",
    ),
]


def parse_markdown(file_path: Path) -> tuple[dict[str, str], str]:
    """Tách frontmatter YAML và nội dung thân văn bản."""
    text = file_path.read_text(encoding="utf-8")
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            fm_raw = parts[1]
            body = parts[2].strip()
            metadata = {}
            for line in fm_raw.splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    metadata[k.strip()] = v.strip().strip('"').strip("'")
            return metadata, body
    return {}, text.strip()


def get_chunker(strategy: str):
    if strategy == "sentence":
        return SentenceChunker(max_sentences_per_chunk=3)
    elif strategy == "fixed":
        return FixedSizeChunker(chunk_size=400, overlap=80)
    elif strategy == "recursive":
        return RecursiveChunker(chunk_size=400)
    else:
        return SentenceChunker(max_sentences_per_chunk=3)


def build_store(embedder) -> tuple[EmbeddingStore, int, float]:
    store = EmbeddingStore(collection_name="uth_library", embedding_fn=embedder)
    chunker = get_chunker(STRATEGY)

    all_docs: list[Document] = []
    lengths = []

    md_files = sorted(DATA_DIR.glob("*.md"))
    for p in md_files:
        meta, body = parse_markdown(p)
        doc_id = meta.get("doc_id", p.stem)
        meta["doc_id"] = doc_id

        raw_chunks = chunker.chunk(body)
        for i, chunk_text in enumerate(raw_chunks):
            lengths.append(len(chunk_text))
            chunk_doc = Document(
                id=f"{doc_id}#{i}",
                content=chunk_text,
                metadata=dict(meta),
            )
            all_docs.append(chunk_doc)

    store.add_documents(all_docs)
    avg_len = sum(lengths) / len(lengths) if lengths else 0.0
    return store, len(all_docs), avg_len


def main():
    load_dotenv(override=False)
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()

    if provider == "openai":
        try:
            embedder = OpenAIEmbedder(model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL))
            print("Đang sử dụng OpenAI Embedding (text-embedding-3-small)")
        except Exception as e:
            print(f"Không thể khởi tạo OpenAI Embedder ({e}), chuyển về MockEmbedder.")
            embedder = _mock_embed
    elif provider == "gemini":
        try:
            embedder = GeminiEmbedder(model_name=os.getenv("GEMINI_EMBEDDING_MODEL", GEMINI_EMBEDDING_MODEL))
            print("Đang sử dụng Gemini Embedding (gemini-embedding-001)")
        except Exception as e:
            print(f"Không thể khởi tạo Gemini Embedder ({e}), chuyển về MockEmbedder.")
            embedder = _mock_embed
    else:
        embedder = _mock_embed
        print("Đang sử dụng MockEmbedder.")

    store, total_chunks, avg_len = build_store(embedder)
    print(f"\nChiến lược: {STRATEGY} | Tổng số chunks: {total_chunks} | Độ dài TB: {avg_len:.1f} ký tự")

    # Mock/Local LLM function for demonstration
    def simple_llm(prompt: str) -> str:
        return "Căn cứ theo tài liệu cung cấp [1], thông tin được quy định rõ trong nội quy thư viện."

    agent = KnowledgeBaseAgent(store=store, llm_fn=simple_llm)

    output_lines = []
    output_lines.append(f"BENCHMARK RESULT - NGUYỄN ĐỨC MINH (2A202602891)")
    output_lines.append(f"Chiến lược: {STRATEGY} | Chunks: {total_chunks} | Độ dài TB: {avg_len:.1f}")
    output_lines.append("=" * 70)

    doc_id_correct = 0
    content_score = 0
    top3_relevant_count = 0

    print("\n--- CHẠY 5 CÂU HỎI BENCHMARK ---")
    for item in QUERIES:
        qid = item["id"]
        q = item["query"]
        filt = item["filter"]
        gold_doc = item["gold_doc"]

        results = store.search_with_filter(q, top_k=3, metadata_filter=filt)

        has_gold_doc_top3 = any(r.get("metadata", {}).get("doc_id") == gold_doc for r in results)
        if has_gold_doc_top3:
            doc_id_correct += 2  # Ngây thơ theo doc_id

        # Kiểm tra nội dung
        top1 = results[0] if results else None
        top1_has_ans = item["gold_snippet"].lower() in (top1["content"].lower() if top1 else "")

        has_ans_top3 = any(item["gold_snippet"].lower() in r["content"].lower() for r in results)
        if has_ans_top3:
            top3_relevant_count += 1

        score_item = 0
        if top1_has_ans:
            score_item = 2
        elif has_ans_top3:
            score_item = 1
        content_score += score_item

        info_top1 = f"{top1['metadata'].get('doc_id')} (score: {top1['score']:.3f})" if top1 else "None"
        output_lines.append(f"Q{qid}: {q}")
        output_lines.append(f"  Filter: {filt}")
        output_lines.append(f"  Top-1: {info_top1} | Điểm nội dung: {score_item}/2")
        for rank, r in enumerate(results, 1):
            snippet = r['content'][:90].replace('\n', ' ')
            output_lines.append(f"    [{rank}] {r['metadata'].get('doc_id')} ({r['score']:.3f}): {snippet}...")
        output_lines.append("-" * 70)

    summary_line = f"TỔNG KẾT: theo doc_id={doc_id_correct}/10 · theo nội dung={content_score}/10 · top-3: {top3_relevant_count}/5"
    output_lines.append(summary_line)
    print(summary_line)

    print("\n--- ĐO ĐỘ TƯƠNG ĐỒNG 5 CẶP CÂU (MỤC 4 BÁO CÁO CÁ NHÂN) ---")
    output_lines.append("\n5 CẶP CÂU DỰ ĐOÁN COSINE SIMILARITY:")
    for idx, (sent_a, sent_b, pred) in enumerate(SIMILARITY_PAIRS, 1):
        vec_a = embedder(sent_a)
        vec_b = embedder(sent_b)
        sim = compute_similarity(vec_a, vec_b)
        line = f"Cặp {idx}: sim = {sim:.3f} | Dự đoán: {pred} | A: '{sent_a}' vs B: '{sent_b}'"
        print(line)
        output_lines.append(line)

    # Ghi file kết quả
    Path("ket_qua_benchmark.txt").write_text("\n".join(output_lines), encoding="utf-8")
    print("\nĐã lưu toàn bộ kết quả vào ket_qua_benchmark.txt")


if __name__ == "__main__":
    main()
