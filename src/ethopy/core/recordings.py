"""Create tables for the lab_recordings schema."""

import datajoint as dj

from ethopy.core.logger import (  # noqa: F401
    experiment,
    recording,
)


@recording.schema
class Software(dj.Lookup):
    """Acquisition software and version."""
    definition = """
    software: varchar(64)
    version: varchar(16)
    ---
    description: varchar(248)
    """


@recording.schema
class Aim(dj.Lookup):
    """Recording aim."""
    definition = """
    rec_aim: varchar(16)
    ---
    rec_time: enum('functional','structural','behavior','other','sync')
    description: varchar(2048)
    """


@recording.schema
class Recording(dj.Manual):
    """Recording metadata."""
    definition = """
    -> experiment.Session
    rec_idx: smallint
    ---
    -> Software
    -> Aim
    filename: varchar(255)
    source_path: varchar(512)
    target_path: varchar(512)
    timestamp=CURRENT_TIMESTAMP : timestamp
    """
