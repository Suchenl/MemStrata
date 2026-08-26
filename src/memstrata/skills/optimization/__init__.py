"""Production self-optimization skill (agent-in-the-loop).

Turns a live/finished production run into a decision-oriented checkpoint report so an
agent can observe → diagnose → tune/rerun during a long rollout. See ``SKILL.md`` for the
protocol and ``monitor.py`` for the deterministic signal extractor.
"""

from .monitor import SKILL_KNOBS, analyze

__all__ = ["analyze", "SKILL_KNOBS"]
