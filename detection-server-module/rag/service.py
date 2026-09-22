from app.schemas import RagDocument

from .retriever import retrieve_with_scores


class RAGService:
    """Retrieve domain documents from the persistent Chroma index."""

    async def retrieve(
        self,
        retrieval_query: str,
        resolved=None,
    ) -> list[RagDocument]:
        del resolved  # Reserved for metadata-aware retrieval extensions.

        documents = retrieve_with_scores(retrieval_query)
        return [
            RagDocument(
                id=f"rag_{index}",
                title=document.metadata.get("source", "unknown"),
                content=document.page_content,
                source=document.metadata.get("source", "unknown"),
                score=float(score),
            )
            for index, (document, score) in enumerate(documents)
        ]
