"""Compatibility import for the public ATOM RLHF model runner.

Custom all-reduce policy is an ATOM configuration concern. AtomRL must not
patch aiter modules or ATOM's private distributed helpers from this module.
"""


class NoCustomARModelRunner:
    """Backward-compatible name for ATOM's public RLHF runner."""

    def __new__(cls, *args, **kwargs):
        from atom.rollout.model_runner_ext import RLHFModelRunner

        return RLHFModelRunner(*args, **kwargs)
