# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Đức Minh  
**MSSV:** 2A202602891  
**Nhóm:** RAUMAMIENTAY  
**Ngày:** 2026-09-19  

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**  
> *Viết 1-2 câu:*  
> Độ tương tự cosine cao (tiệm cận 1.0) nghĩa là hai vector embedding chỉ cùng một hướng trong không gian đa chiều, thể hiện hai đoạn văn bản có sự tương đồng lớn về ngữ nghĩa và chủ đề bất kể độ dài hay từ ngữ bề mặt có thể khác nhau.

**Ví dụ có độ tương tự CAO:**  
- Câu A: Quy định về việc mượn sách về nhà của sinh viên.
- Câu B: Chính sách cho phép người học mang tài liệu thư viện ra ngoài.
- Tại sao tương đồng: Hai câu dùng từ vựng hoàn toàn khác nhau ("sinh viên" vs "người học", "mượn sách về nhà" vs "mang tài liệu ra ngoài") nhưng diễn đạt cùng một ý định và bản chất ngữ nghĩa.

**Ví dụ có độ tương tự THẤP:**  
- Câu A: Thời hạn trả sách và mức phạt quá hạn tại thư viện.
- Câu B: Hướng dẫn cách nấu món phở bò truyền thống của Hà Nội.
- Tại sao khác: Hai câu thuộc hai miền chủ đề hoàn toàn độc lập (dịch vụ học thuật vs ẩm thực), không chia sẻ ngữ cảnh hay khái niệm chung.

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**  
> *Viết 1-2 câu:*  
> Khoảng cách Euclid bị phụ thuộc nặng nề vào độ dài của vector (độ dài văn bản), khiến hai đoạn văn cùng nghĩa nhưng một đoạn dài và một đoạn ngắn bị coi là xa nhau. Ngược lại, Cosine similarity chỉ đo góc giữa hai vector (`cos(θ) = (A · B) / (|A| × |B|)`), chuẩn hóa độ dài nên tập trung thuần túy vào hướng ngữ nghĩa của văn bản.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**  
> *Trình bày phép tính:*  
> Bước nhảy giữa các chunk liên tiếp: `step = chunk_size - overlap = 500 - 50 = 450` ký tự.  
> Chunk 1 bắt đầu từ 0 đến 500 (còn lại 9500 ký tự cần phủ).  
> Số chunk bổ sung: `ceil((10000 - 500) / 450) = ceil(9500 / 450) = ceil(21.111...) = 22`.  
> Hoặc áp dụng công thức: `ceil((10000 - 50) / (500 - 50)) = ceil(9950 / 450) = ceil(22.11) = 23` nếu tính cả phần rơi ra cuối cùng, hoặc chạy chính xác theo vòng lặp `range(0, 10000, 450)` sẽ sinh đúng 22 chunks (start = 0, 450, 900, ..., 9450; tại start=9450, `start + 500 = 9950 < 10000` nên thêm 1 bước cuối hoặc dừng tùy điều kiện). Kiểm tra trực tiếp với code vòng lặp repo ra 22 chunks.  
> *Đáp án:* 22 chunks.

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**  
> *Viết 1-2 câu:*  
> Khi overlap tăng lên 100, bước nhảy giảm xuống `500 - 100 = 400`, làm số lượng chunk tăng lên thành `ceil(9500 / 400) + 1 = 25` chunks. Ta muốn overlap lớn hơn để bảo toàn ngữ cảnh ở ranh giới cắt, tránh việc một câu văn, số liệu hoặc mệnh đề quan trọng bị chia đôi làm mất ý nghĩa khi tìm kiếm.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi lập trình (implement) các phần chính trong gói `src`.

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:  
> Sử dụng regex `(?<=[.!?])\s+` (positive lookbehind) để tách câu ở vị trí khoảng trắng ngay sau dấu chấm, chấm than hoặc hỏi chấm mà không làm mất dấu câu của câu trước. Sau đó dùng list comprehension lọc bỏ câu rỗng và gom nhóm `max_sentences_per_chunk` câu liền kề thành một chunk trọn vẹn; xử lý edge case chuỗi rỗng trả về `[]`.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:  
> Thuật toán đệ quy 2 chiều: thử lần lượt danh sách phân cách ưu tiên `["\n\n", "\n", ". ", " ", ""]`. Nếu mảnh văn bản lớn hơn `chunk_size` thì đệ quy với separator cấp thấp hơn; đồng thời có bước gom (merge) các mảnh nhỏ liên tiếp sao cho tổng độ dài tiệm cận `chunk_size` để tránh sinh ra các chunk vụn. Base case dừng khi văn bản nhỏ hơn `chunk_size` hoặc hết danh sách separators (cắt cứng theo độ dài).

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:  
> Khởi tạo bộ nhớ thuần in-memory (list các dictionary) để an toàn và nhất quán trên mọi môi trường chấm bài. Hàm `_make_record` chuẩn hóa `Document`, tự động gán `doc_id` vào metadata và nhúng embedding; hàm `search` tính tích vô hướng (dot product) giữa vector câu hỏi đã chuẩn hóa và các vector lưu trữ, sắp xếp giảm dần và lấy ra top-k kèm `score`.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:  
> `search_with_filter` thực hiện pre-filtering (lọc trước): quét lọc các bản ghi khớp toàn bộ điều kiện trong `metadata_filter` rồi mới đưa danh sách ứng viên vào tính similarity, đảm bảo không bị mất slot top-k bởi tài liệu sai đối tượng. Hàm `delete_document` lọc bỏ mọi bản ghi có `id` hoặc `metadata['doc_id']` trùng với `doc_id` cần xóa và trả về `True` nếu số lượng phần tử giảm đi.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:  
> Truy xuất top-k chunk từ store, kiểm tra nếu store rỗng thì thông báo không có dữ liệu mà không gọi LLM. Xây dựng prompt chứa các khối ngữ cảnh được đánh số `[1]`, `[2]` kèm tên nguồn tài liệu (`doc_id`), yêu cầu LLM chỉ trả lời dựa vào ngữ cảnh và trích dẫn số thứ tự nguồn để phục vụ việc truy vết nguồn gốc (traceability).

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```
============================= test session starts =============================
platform win32 -- Python 3.10.11, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\AI Vin\K4-L3A-Data-Foundations
collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

============================== 42 passed in 0.18s ==============================
```

**Số lượng bài test vượt qua (pass):** 42 / 42

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Tôi muốn mượn sách về nhà. | Tôi muốn gia hạn sách đã mượn. | cao | 0.794 | Đúng |
| 2 | Thời hạn mượn sách của sinh viên là bao lâu? | Mức phạt quá hạn mượn sách là bao nhiêu? | cao | 0.813 | Đúng |
| 3 | Giảng viên được mượn 10 tài liệu. | Sinh viên được mượn 5 tài liệu. | cao | 0.880 | Đúng |
| 4 | Tôi muốn mượn sách. | Tôi không muốn mượn sách. | thấp | 0.852 | Sai (Bất ngờ) |
| 5 | Quy định phòng đọc thư viện. | Công thức nấu món phở bò truyền thống. | thấp | 0.565 | Đúng |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**  
> *Viết 2-3 câu:*  
> Kết quả bất ngờ nhất là Cặp 4 ("Tôi muốn mượn sách" và "Tôi không muốn mượn sách") có độ tương đồng đạt tới 0.852 — cao hơn cả cặp đồng nghĩa thực sự ở Cặp 1 (0.794). Điều này phản ánh rõ ràng rằng mô hình embedding mã hóa theo "chủ đề và ngữ cảnh từ vựng" (cùng nói về việc mượn sách) chứ không thực sự hiểu được các từ phủ định ngữ pháp mang tính đảo ngược ngữ nghĩa ("không"), chứng minh tại sao retrieval cần kết hợp thêm LLM để suy luận logic.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** trên mã nguồn cá nhân của bạn trong gói `src`. **5 câu hỏi này phải trùng với các thành viên cùng nhóm** (xem `REPORT_NHOM.md`).

Chiến lược cá nhân: **`SentenceChunker(max_sentences_per_chunk=3)`** (41 chunks, độ dài trung bình 222.7 ký tự, mô hình `gemini-embedding-001`).

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Trả sách quá hạn thì bị phạt bao nhiêu tiền? | `phuc-vu-muon-tra-tai-lieu`: "## Quá hạn  Trả sách quá hạn phạt 1000 đồng/1 cuốn/1 ngày..." | 0.900 | Có (hạng 1) | "phạt 1000 đồng/1 cuốn/1 ngày [1]" — đúng (2đ) |
| 2 | Sách mượn về nhà được gia hạn mấy lần, mỗi lần bao lâu? | `phuc-vu-muon-tra-tai-lieu`: "Thời gian mượn về nhà: 45 ngày. Số lần được gia hạn: 01 lần với thời gian 45 ngày..." | 0.849 | Có (hạng 1) | "gia hạn 01 lần với thời gian 45 ngày [1]" — đúng (2đ) |
| 3 | Mỗi bạn đọc được mượn tối đa bao nhiêu tài liệu về nhà? (filter `audience: student`) | `quy-dinh-muon-tra-sinh-vien`: "– Kiểm tra số lượng và tình trạng tài liệu khi mượn. ## 2. Số lượng tài liệu được mượn..." | 0.848 | Có (hạng 1) | "Sinh viên: 05 tài liệu tiếng Việt, 03 tài liệu tiếng Anh [1]" — đúng (2đ) |
| 4 | Khi vào phòng đọc được mang theo những gì? | `noi-quy-thu-vien`: "Nội quy phòng đọc  - Xuất trình thẻ Thư viện khi vào phòng đọc. - Chỉ được mang vào phòng đọc: Máy tính cá nhân, Sách..." | 0.850 | Có (hạng 1) | "Chỉ được mang vào phòng đọc máy tính cá nhân, sách, tập vở và dụng cụ học tập [1]" — đúng (2đ) |
| 5 | Muốn kiểm tra tỉ lệ trùng lặp cho khóa luận, đồ án thì dùng dịch vụ nào? | `dich-vu-quet-trung-lap`: "# Dịch vụ quét trùng lặp  ## Dịch vụ  Xác định tỉ lệ trùng lắp nội dung... Turnitin..." | 0.840 | Có (hạng 1) | "Dịch vụ quét trùng lặp bằng phần mềm chống đạo văn Turnitin [1]" — đúng (2đ) |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** 5 / 5 (Cả 5 câu đều nằm ở Top-1, tổng điểm nội dung: 10/10)

**A/B Test câu hỏi 3 (Lọc Metadata Audience):**
- **Khi CÓ filter `audience: student`:** Top-1 đến Top-3 tập trung vào đúng quy định sinh viên, loại bỏ hoàn toàn quy định của giảng viên; agent trả lời đúng chuẩn xác đối tượng sinh viên.
- **Khi KHÔNG CÓ filter:** Top-1 và Top-2 bị chiếm bởi tài liệu giảng viên (score 0.880), dẫn tới agent trả lời: "tối đa 10 tài liệu", **sai hoàn toàn về đối tượng**. Điều này chứng minh metadata filter là bắt buộc khi xử lý tài liệu đa đối tượng.

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**  
> Tôi học được từ bạn Tài (chiến lược `HeadingChunker`) cách gắn ngược đường dẫn tiêu đề (`Breadcrumb heading`) vào từng chunk con sau khi tách. Việc này giúp các đoạn văn ngắn trong các tiểu mục không bao giờ bị mất ngữ cảnh của "Điều" hay "Chương" lớn phía trên, giải quyết triệt để nhược điểm bị đứt gãy ý nghĩa của `SentenceChunker` khi gặp các văn bản hành chính nhiều danh sách gạch đầu dòng.

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 10 / 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 5 / 5 |
| Kết quả truy xuất của tôi (Competition Results) | 10 / 10 |
| **Tổng phần cá nhân** | **60 / 60** |
