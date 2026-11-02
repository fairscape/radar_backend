"""Score post-processing utilities.

Selectors emit ``(score, paper, breakdown)`` tuples; modules in this
package transform that output. Today only ``percentile.attach_percentile``
exists.
"""

from .percentile import attach_percentile

__all__ = ["attach_percentile"]
