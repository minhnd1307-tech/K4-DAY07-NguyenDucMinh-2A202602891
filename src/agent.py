from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def build_prompt(self, question: str, results: list[dict]) -> str:
        """Construct the prompt with retrieved context and traceability instructions."""
        context_blocks = []
        for i, r in enumerate(results, start=1):
            source = r.get("metadata", {}).get("doc_id", r.get("id", f"doc_{i}"))
            content = r.get("content", "").strip()
            context_blocks.append(f"[{i}] (Nguồn: {source})\n{content}")

        context_str = "\n\n".join(context_blocks)
        prompt = (
            "Bạn là trợ lý thông minh trả lời câu hỏi dựa trên các tài liệu được cung cấp.\n"
            "Hãy trả lời câu hỏi CHỈ DỰA TRÊN ngữ cảnh dưới đây. Trích dẫn số thứ tự nguồn [1], [2] "
            "cho từng thông tin tương ứng. Nếu ngữ cảnh không chứa thông tin để trả lời, "
            "hãy trả lời rằng bạn không tìm thấy thông tin trong tài liệu cung cấp.\n\n"
            f"--- NGỮ CẢNH ---\n{context_str}\n\n"
            f"--- CÂU HỎI ---\n{question}\n\n"
            "--- CÂU TRẢ LỜI ---"
        )
        return prompt

    def answer(self, question: str, top_k: int = 3) -> str:
        if self.store.get_collection_size() == 0:
            return "Cơ sở tri thức hiện chưa có tài liệu nào để trả lời."

        results = self.store.search(question, top_k=top_k)
        if not results:
            return "Không tìm thấy tài liệu phù hợp trong cơ sở tri thức."

        prompt = self.build_prompt(question, results)
        return self.llm_fn(prompt)
