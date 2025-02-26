"""Utilities for handling task-related operations in EthoPy."""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

log = logging.getLogger(__name__)


@dataclass
class Task:
    """Represents a task configuration with its path and identifier."""

    path: Optional[Path]
    id: Optional[int]

    @property
    def is_manual(self) -> bool:
        """Return True if either path or id is specified, indicating manual
        task selection"""
        return bool(self.path or self.id)


def resolve_task(
    task_path: Optional[Union[str, Path]] = None, task_id: Optional[int] = None
) -> Task:
    """
    Resolve task configuration from either a path or ID.

    Args:
        task_path: Optional path to task file
        task_id: Optional task ID from database

    Returns:
        Task object containing resolved path and ID

    Raises:
        FileNotFoundError: If task path doesn't exist
        ValueError: If task ID doesn't exist in database
    """
    if task_path and task_id:
        raise ValueError("Cannot specify both task path and ID")

    if task_path:
        path = Path(task_path)
        # If only filename provided, look in default config directory
        if not path.parent.name:
            path = Path(__file__).parent.parent / "task" / path

        if not path.is_file():
            raise FileNotFoundError(f"Task file not found: {path}")

        return Task(path=path, id=None)

    if task_id:
        # Import here to avoid circular dependency
        from ethopy.core.logger import experiment

        task_query = experiment.Task() & {"task_idx": task_id}
        if not len(task_query):
            raise ValueError(f"No task found with ID: {task_id}")

        task_path = Path(task_query.fetch1("task"))
        path, filename = os.path.split(task_path)
        if not path:
            task_path = Path(
                os.path.join(
                    str(Path(__file__).parent.absolute()), "..", "task", filename
                )
            )
        if not task_path.is_file():
            raise FileNotFoundError(f"Task file from database not found: {task_path}")

        return Task(path=task_path, id=task_id)

    return Task(path=None, id=None)
