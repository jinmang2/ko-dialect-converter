from __future__ import annotations

import os
from contextlib import contextmanager

from omegaconf import DictConfig


@contextmanager
def setup_tracking(logger_cfg: DictConfig, experiment_cfg: DictConfig):
    """Set tracking env vars so HF Trainer callbacks (MLflow/Wandb) pick them up."""
    name = str(logger_cfg.get("name", "disabled"))

    if name == "mlflow":
        tracking_uri = str(logger_cfg.get("tracking_uri", "mlruns"))
        os.environ.setdefault("MLFLOW_TRACKING_URI", tracking_uri)
        os.environ["MLFLOW_EXPERIMENT_NAME"] = str(experiment_cfg.name)
        if experiment_cfg.get("run_name"):
            os.environ["MLFLOW_RUN_NAME"] = str(experiment_cfg.run_name)
        try:
            yield
        finally:
            os.environ.pop("MLFLOW_RUN_NAME", None)

    elif name == "wandb":
        os.environ["WANDB_PROJECT"] = str(logger_cfg.get("project", "ko-dialect"))
        if logger_cfg.get("entity"):
            os.environ["WANDB_ENTITY"] = str(logger_cfg.entity)
        if experiment_cfg.get("run_name"):
            os.environ["WANDB_NAME"] = str(experiment_cfg.run_name)
        tags = list(logger_cfg.get("tags", []))
        if tags:
            os.environ["WANDB_TAGS"] = ",".join(tags)
        if logger_cfg.get("notes"):
            os.environ["WANDB_NOTES"] = str(logger_cfg.notes)
        try:
            yield
        finally:
            for _key in (
                "WANDB_PROJECT",
                "WANDB_ENTITY",
                "WANDB_NAME",
                "WANDB_TAGS",
                "WANDB_NOTES",
            ):
                os.environ.pop(_key, None)

    else:
        os.environ["WANDB_DISABLED"] = "true"
        try:
            yield
        finally:
            os.environ.pop("WANDB_DISABLED", None)
