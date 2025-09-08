from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import yaml


class ConfigParsingFailed(Exception):
    pass


@dataclass
class DataConfig:
    version: str
    dataset_name: str
    onehot_encode_columns: List[str]
    label_encode_columns: List[str]


@dataclass
class FeaturesConfig:
    enable_duration_features: bool
    enable_balance_features: bool
    enable_campaign_features: bool
    enable_seasonal_features: bool
    enable_previous_outcome_features: bool


@dataclass
class ModelEvalConfig:
    cv_folds: int
    stratified: bool
    primary_metric: str
    random_state: int


@dataclass
class ModelConfig:
    model_type: str
    model_params: Dict[str, Any]


@dataclass
class Config:
    data_prep: DataConfig
    feature_prep: FeaturesConfig
    model_evaluation: ModelEvalConfig
    model: ModelConfig

    @staticmethod
    def from_yaml(config_file: str):
        with open(config_file, 'r', encoding='utf-8') as stream:
            try:
                config_data = yaml.safe_load(stream)
                return Config(
                    data_prep=DataConfig(**config_data['data_prep']),
                    feature_prep=FeaturesConfig(**config_data['feature_prep']),
                    model_evaluation=ModelEvalConfig(**config_data['model_evaluation']),
                    model=ModelConfig(**config_data['model'])
                )
            except (yaml.YAMLError, OSError) as e:
                raise ConfigParsingFailed from e