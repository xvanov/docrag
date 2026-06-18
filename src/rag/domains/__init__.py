"""Pluggable use-case domains layered over the source-agnostic core.

Each domain binds a corpus to its ingest path, its chat backend, its retrieval
tuning, and its citation rendering. See ``base.Domain`` for the interface and
``registry`` for lookup.
"""
