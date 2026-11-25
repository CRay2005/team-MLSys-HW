"""Explicitly enabled HW3 diagnostic observers in every spawned Python rank."""
import os

if os.environ.get("HW3_TRACE_DIR"):
    from runtime_observer import install

    install()
