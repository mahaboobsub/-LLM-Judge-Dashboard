"""
src/logger.py
=============
Audit logging infrastructure for the LLM-as-Judge pipeline.

Provides:
  - JSONLinesLogger  — append-only .jsonl log files (immutable audit trail)
  - log_result()     — convenience wrapper to log a single evaluation result
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

from src.schema import AuditLogEntry


class JSONLinesLogger:
    """
    Append-only JSON Lines logger for audit trails.

    Each line in the .jsonl file is a complete, self-contained JSON object
    representing one judge evaluation call (prompt, raw response, verdict,
    tokens, cost, latency).

    Parameters
    ----------
    log_dir : str — Directory where log files will be stored.
    run_name : str — Base name for the log file (without extension).
    """

    def __init__(self, log_dir: str = "logs", run_name: Optional[str] = None) -> None:
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        if run_name is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            run_name = f"run_{timestamp}"

        self.log_file = os.path.join(log_dir, f"{run_name}.jsonl")
        self._entry_count = 0

    def log_audit_entry(self, entry: AuditLogEntry) -> None:
        """Append a single AuditLogEntry to the log file."""
        record = entry.model_dump()
        self._write_record(record)

    def log_dict(self, record: Dict[str, Any]) -> None:
        """Append a raw dict to the log file."""
        self._write_record(record)

    def _write_record(self, record: Dict[str, Any]) -> None:
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
        self._entry_count += 1

    @property
    def entry_count(self) -> int:
        """Number of entries written in this session."""
        return self._entry_count

    @property
    def log_path(self) -> str:
        """Full path to the current log file."""
        return self.log_file

    def read_all(self) -> list:
        """Read and parse all entries from the log file."""
        entries = []
        if not os.path.exists(self.log_file):
            return entries
        with open(self.log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return entries


def log_result(
    logger: JSONLinesLogger,
    audit: AuditLogEntry,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Convenience function to log an audit entry with optional extra metadata.

    Parameters
    ----------
    logger : JSONLinesLogger
    audit  : AuditLogEntry
    extra  : dict — Additional fields to merge into the log record
    """
    record = audit.model_dump()
    if extra:
        record.update(extra)
    logger.log_dict(record)
