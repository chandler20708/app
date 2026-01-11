"""Unified access point for AED analytics pipelines."""

from dataclasses import dataclass
from typing import Dict, Optional

from models.aed_insights.config import AEDConfig
from models.aed_insights.descriptive import DescriptiveArtifacts
from models.aed_insights.inference import InferenceArtifacts
from models.aed_insights.modelling import ModellingArtifacts
from models.aed_insights.pipelines import (
    run_descriptive_pipeline,
    run_inference_pipeline,
    run_modelling_pipeline,
)


@dataclass
class AnalyticsRegistry:
    """Registry that exposes analytics pipelines."""
    config: AEDConfig

    def descriptive(self) -> DescriptiveArtifacts:
        return run_descriptive_pipeline(self.config)

    def inference(self) -> InferenceArtifacts:
        return run_inference_pipeline(self.config)

    def modelling(self) -> ModellingArtifacts:
        return run_modelling_pipeline(self.config)

    def all_artifacts(self) -> Dict[str, object]:
        return {
            "descriptive": self.descriptive(),
            "inference": self.inference(),
            "modelling": self.modelling(),
        }


def load_registry(config: Optional[AEDConfig] = None) -> AnalyticsRegistry:
    """Create a registry with the provided configuration."""
    return AnalyticsRegistry(config=config or AEDConfig())
