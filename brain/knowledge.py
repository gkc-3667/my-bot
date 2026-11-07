"""Persistent user-taught knowledge and instruction examples.

Stored facts are retrieval memory, not model-weight updates. Curated
question/answer examples can be exported to the existing SFT JSONL format.
"""
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class KnowledgeItem:
    id: int
    topic: str
    content: str
    source: str
    created_at: str


def _terms(text):
    return {
        token for token in re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE)
        if len(token) > 1
    }


class KnowledgeBase:
    def __init__(self, database_path="data/knowledge.sqlite3"):
        self.path = Path(database_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic TEXT NOT NULL DEFAULT '',
                    content TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                )
                """
            )

    def _connect(self):
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def add(self, content, topic="", source=""):
        content = str(content).strip()
        if not content:
            raise ValueError("Knowledge content must not be empty.")
        if len(content) > 20000:
            raise ValueError("Knowledge content is too long (maximum 20,000 characters).")
        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO knowledge(topic, content, source, created_at) VALUES (?, ?, ?, ?)",
                (str(topic).strip()[:200], content, str(source).strip()[:1000], created_at),
            )
            item_id = cursor.lastrowid
        return item_id

    def search(self, query, limit=5):
        limit = max(1, min(int(limit), 20))
        terms = _terms(query)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, topic, content, source, created_at FROM knowledge "
                "ORDER BY id DESC LIMIT 500"
            ).fetchall()

        scored = []
        for row in rows:
            item_terms = _terms(row["topic"] + " " + row["content"])
            score = len(terms & item_terms) if terms else 0
            if not terms or score:
                scored.append((score, row["id"], row))
        scored.sort(key=lambda entry: (entry[0], entry[1]), reverse=True)
        return [
            KnowledgeItem(
                id=row["id"],
                topic=row["topic"],
                content=row["content"],
                source=row["source"],
                created_at=row["created_at"],
            )
            for _, _, row in scored[:limit]
        ]

    def count(self):
        """Return the number of stored knowledge notes."""
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS total FROM knowledge").fetchone()
        return int(row["total"])

    @staticmethod
    def add_training_example(prompt, answer, path="data/learned_examples.jsonl"):
        prompt = str(prompt).strip()
        answer = str(answer).strip()
        if not prompt or not answer:
            raise ValueError("Both prompt and answer are required.")
        if len(prompt) > 10000 or len(answer) > 20000:
            raise ValueError("Prompt or answer is too long.")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        example = {"prompt": prompt, "answer": answer}
        encoded = json.dumps(example, ensure_ascii=False)
        if target.exists():
            for line in target.read_text(encoding="utf-8").splitlines():
                try:
                    if json.loads(line) == example:
                        return False
                except json.JSONDecodeError:
                    continue
        with target.open("a", encoding="utf-8") as file:
            file.write(encoded + "\n")
        return True
