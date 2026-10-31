"""
Base interfaces and definitions for reranking engines, which judge how well
each of some texts answers a query.
"""
from typing import List, Optional


class RerankEngine:
    """
    Abstract base class for all reranking engines.
    """
    MODEL_ID = ""

    def get_model_id(self) -> str:
        return self.MODEL_ID

    def rank(self, query: str, documents: List[str], instruction: Optional[str] = None) -> List[float]:
        """
        How relevant each document is to the query, from 0 to 1, in the
        documents' order. instruction says what relevant means, if not
        answering the query.
        """
        raise NotImplementedError

    @classmethod
    def setup(cls):
        """
        Performs first-time setup, such as downloading default models.
        """
        pass
