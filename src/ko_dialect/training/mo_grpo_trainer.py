"""MO-GRPO trainer: per-objective group-normalize, then sum (Arm 1 aggregation).

MO-GRPO (Ichihara et al., CyberAgentAILab, 2025; arXiv:2509.22047;
github.com/CyberAgentAILab/MO-GRPO) fixes a multi-objective failure of vanilla GRPO:
when several reward functions are *summed* and the group advantage is normalized once,
the highest-variance reward axis dominates the advantage and the policy reward-hacks
that single axis (exactly the style-axis domination we observed). The method:

    For each prompt group (size = ``num_generations``), z-normalize EACH reward
    function's values *within the group* (per-objective mean/std), THEN sum the
    normalized values to form the scalar advantage — i.e. it swaps the order of
    "sum" and "normalize". This equalizes every objective's contribution
    regardless of native scale/variance while preserving each objective's
    preference ordering, so no single axis can dominate.

TRL 1.5.1 implements this algorithm natively in
``GRPOTrainer._generate_and_score_completions`` under
``multi_objective_aggregation="normalize_then_sum"`` (z-normalize each reward over the
group via ``(grouped - mean_k) / (std_k + eps)``, then weighted ``nansum`` → advantage).

We also force ``scale_rewards="none"``, but NOT for the reason an earlier version of this
docstring claimed. Reading the TRL 1.5.1 source: the ``normalize_then_sum`` branch does
**not reference ``scale_rewards`` at all** — it unconditionally re-standardizes the summed
advantage across the batch (``(rewards - mean) / (std + 1e-4)``). So ``scale_rewards="none"``
is a harmless **no-op** in this branch (it does not prevent any "double-scaling"); we set it
only for clarity / forward-compatibility and to match the paper-faithful intent. Note this
means TRL actually does per-objective-z-norm → sum → *a second batch-level normalize*,
which is slightly beyond the pure paper ("normalize then sum"); the extra step is benign
(a monotonic rescale of the whole advantage) and does not reintroduce axis domination.

Rather than re-implement the deep, version-fragile ``_generate_and_score_completions``,
this subclass simply *enforces* the two TRL settings that realize MO-GRPO and validates
the group-divisibility precondition, so a caller cannot silently fall back to the
sum-then-normalize behavior. If a future TRL drops the ``normalize_then_sum`` branch,
construction fails loudly here instead of degrading to vanilla GRPO at runtime.
"""

from __future__ import annotations

import logging

from trl import GRPOTrainer

logger = logging.getLogger(__name__)


class MOGRPOTrainer(GRPOTrainer):
    """GRPOTrainer that enforces MO-GRPO aggregation (normalize-then-sum, no re-scale).

    Overrides ``args.multi_objective_aggregation`` to ``"normalize_then_sum"`` and
    ``args.scale_rewards`` to ``"none"`` before delegating to ``GRPOTrainer.__init__``,
    so the per-objective-then-sum advantage path (arXiv:2509.22047) is always taken.
    """

    def __init__(self, *args, **kwargs):
        train_args = kwargs.get("args")
        if train_args is None and args:
            # GRPOTrainer signature is (model, args, ...); args[1] is the config.
            train_args = args[1] if len(args) > 1 else None
        if train_args is None:
            raise ValueError("MOGRPOTrainer requires a GRPOConfig via the `args=` keyword.")

        # Enforce the MO-GRPO algorithm. Both TRL branches live in
        # `_generate_and_score_completions`; this pair selects normalize-then-sum and
        # prevents TRL from re-normalizing the already per-objective-normalized sum.
        if getattr(train_args, "multi_objective_aggregation", None) != "normalize_then_sum":
            logger.info(
                "MOGRPOTrainer: forcing multi_objective_aggregation='normalize_then_sum' (was %r)",
                getattr(train_args, "multi_objective_aggregation", None),
            )
            train_args.multi_objective_aggregation = "normalize_then_sum"
        # scale_rewards is post-init-normalized by GRPOConfig to a str ("none"/"group"/
        # "batch"); set the canonical string so it survives any further validation.
        if getattr(train_args, "scale_rewards", None) != "none":
            logger.info(
                "MOGRPOTrainer: forcing scale_rewards='none' (was %r)",
                getattr(train_args, "scale_rewards", None),
            )
            train_args.scale_rewards = "none"

        # Guard the TRL precondition: per_device_train_batch_size must be divisible by
        # num_generations (each device batch must hold whole prompt groups). MO-GRPO's
        # per-group z-normalization is only well-defined over complete groups.
        bs = getattr(train_args, "per_device_train_batch_size", None)
        ng = getattr(train_args, "num_generations", None)
        if bs and ng and bs % ng != 0:
            raise ValueError(
                f"MO-GRPO requires per_device_train_batch_size ({bs}) to be divisible "
                f"by num_generations ({ng}); each device batch must contain whole prompt "
                "groups for per-group z-normalization."
            )

        super().__init__(*args, **kwargs)
