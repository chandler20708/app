"""Pipelines for descriptive, inference, and modelling AED outputs."""

from typing import Optional

from models.aed_insights.config import AEDConfig
from models.aed_insights.core import AEDRepository, build_featured_df, get_schema
from models.aed_insights.descriptive import (
    DescriptiveArtifacts,
    build_descriptive_figures,
    build_descriptive_tables,
)
from models.aed_insights.inference import InferenceArtifacts, build_inference_tables
from models.aed_insights.modelling import ModellingArtifacts, train_models


def run_descriptive_pipeline(config: Optional[AEDConfig] = None) -> DescriptiveArtifacts:
    """Run the descriptive analytics pipeline."""
    config = config or AEDConfig()
    schema = get_schema()
    repo = AEDRepository(config=config, schema=schema)
    df = repo.load_sampled()
    df = build_featured_df(df, config, schema)

    tables = build_descriptive_tables(df, config, schema)
    figures = build_descriptive_figures(df, config, schema)
    metadata = {
        "sample_size": len(df),
        "data_path": str(config.data.data_path),
    }
    return DescriptiveArtifacts(tables=tables, figures=figures, metadata=metadata)


def run_inference_pipeline(config: Optional[AEDConfig] = None) -> InferenceArtifacts:
    """Run the inference analytics pipeline."""
    config = config or AEDConfig()
    schema = get_schema()
    repo = AEDRepository(config=config, schema=schema)
    df = repo.load_sampled()
    df = build_featured_df(df, config, schema)

    tables = build_inference_tables(df, config, schema)
    metadata = {
        "sample_size": len(df),
        "data_path": str(config.data.data_path),
    }
    return InferenceArtifacts(tables=tables, figures={}, metadata=metadata)


def run_modelling_pipeline(config: Optional[AEDConfig] = None) -> ModellingArtifacts:
    """Run the modelling pipeline and return performance outputs."""
    config = config or AEDConfig()
    schema = get_schema()
    repo = AEDRepository(config=config, schema=schema)
    df = repo.load_sampled()
    df = build_featured_df(df, config, schema)

    models, tables, metadata = train_models(df, config)
    metadata = dict(metadata)
    metadata["models"] = models
    metadata["data_path"] = str(config.data.data_path)
    return ModellingArtifacts(tables=tables, figures={}, metadata=metadata)
