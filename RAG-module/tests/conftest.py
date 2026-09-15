"""Fixtures dung chung cho bo test offline.

Tat ca dependency nang (embedding model va Gemini) deu duoc thay bang fake. Chroma
van chay local trong ``tmp_path`` de kiem tra persistence that ma khong can mang.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Khong gui Chroma telemetry trong luc chay test.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")


class KeywordEmbeddings(Embeddings):
    """Embedding nho, xac dinh va du de test retrieval ma khong tai model."""

    def __init__(self) -> None:
        self.document_calls = 0
        self.query_calls = 0

    @staticmethod
    def _embed(text: str) -> list[float]:
        normalized = text.casefold()
        scab_score = sum(
            normalized.count(term)
            for term in ("ghẻ", "scab", "venturia", "sẹo táo")
        )
        rot_score = sum(
            normalized.count(term)
            for term in ("thối đen", "black rot", "diplodia", "mắt ếch")
        )
        safety_score = sum(
            normalized.count(term)
            for term in ("thuốc", "hoạt chất", "nhãn", "an toàn")
        )
        # Chieu cuoi luon khac 0 de cosine distance hop le voi moi van ban.
        return [float(scab_score), float(rot_score), float(safety_score), 0.1]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls += 1
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls += 1
        return self._embed(text)


@pytest.fixture
def fake_embeddings() -> KeywordEmbeddings:
    return KeywordEmbeddings()


@pytest.fixture
def apple_data_dir(tmp_path: Path) -> Path:
    """Tao corpus co lap tu dung hai file Apple that cua du an."""

    source_dir = PROJECT_ROOT / "data" / "apple"
    target_dir = tmp_path / "data" / "apple"
    target_dir.mkdir(parents=True)
    for filename in ("apple_black_rot.txt", "apple_scab.txt"):
        shutil.copy2(source_dir / filename, target_dir / filename)
    return tmp_path / "data"

