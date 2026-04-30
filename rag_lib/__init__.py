"""rag_lib — Personal Research Radar backend library.

Phase 1A ships the Selector and Gatherer protocols, the Paper / Profile
data classes, the Profile builder, and the compliance test harness.
Phase 1B will populate the selector/gatherer bodies and PDF ingestion.
"""

__version__ = "0.1.0a1"

from .paper import Paper, Topic, TopicNode
from .profile import Profile

__all__ = ["Paper", "Topic", "TopicNode", "Profile"]
