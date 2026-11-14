from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class RunStatus(StrEnum):
    CREATED = "CREATED"
    DATA_VALIDATED = "DATA_VALIDATED"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    INTERRUPTED = "INTERRUPTED"
    BLOCKED = "BLOCKED"


class ValidationIntegrity(StrEnum):
    CLEAN = "CLEAN"
    TAINTED = "TAINTED"
    UNKNOWN = "UNKNOWN"


class ReproductionAssessment(StrEnum):
    NOT_ASSESSED = "NOT_ASSESSED"
    MATCHES_REFERENCE_WITH_TOLERANCE = "MATCHES_REFERENCE_WITH_TOLERANCE"
    DEVIATES = "DEVIATES"
    NOT_COMPARABLE = "NOT_COMPARABLE"
    BLOCKED = "BLOCKED"


ALLOWED_VARIANTS = {"released_code", "evaluation_corrected", "research_variant", "smoke_only"}


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    name: str
    dataset_name: str
    dataset_version: str
    dataset_fingerprint: str
    variant: str
    seed: int
    rank: int
    learning_rate: float
    batch_size: int
    max_epochs: int
    validation_interval_epochs: int
    regularization_strength: float
    reconstruction_coefficient: float
    device: str
    evaluation_ranking: str
    checkpoint_selection: str
    validation_source: str
    validation_holdout_fraction: float
    validation_query_limit: int
    test_query_limit: int
    training_example_limit: int
    config_text: str
    config_fingerprint: str
    implementation_fingerprint: str
    provenance_status: str

    def as_parameters(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "validation_interval_epochs": self.validation_interval_epochs,
            "regularization_strength": self.regularization_strength,
            "reconstruction_coefficient": self.reconstruction_coefficient,
            "device": self.device,
            "evaluation_ranking": self.evaluation_ranking,
            "checkpoint_selection": self.checkpoint_selection,
            "validation_source": self.validation_source,
            "validation_holdout_fraction": self.validation_holdout_fraction,
            "validation_query_limit": self.validation_query_limit,
            "test_query_limit": self.test_query_limit,
            "training_example_limit": self.training_example_limit,
        }
