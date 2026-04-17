import os
from enum import Enum
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
import yaml

class Environment(str, Enum):
    DEVELOPMENT = "DEVELOPMENT"
    STAGING = "STAGING"
    PRODUCTION = "PRODUCTION"

class ProductionConfig(BaseModel):
    """
    Institutional configuration model using Pydantic for validation.
    """
    model_config = ConfigDict(protected_namespaces=())
    
    environment: Environment = Environment.DEVELOPMENT
    trading_enabled: bool = False
    log_level: str = "INFO"
    data_cache_dir: str = "data/parquet"
    
    backtest: Dict[str, Any] = {
        "initial_cash": 100000.0,
        "commission_bps": 1.0,
        "slippage_coeff": 0.1
    }
    
    risk: Dict[str, Any] = {
        "max_order_value": 100000.0,
        "max_drawdown_pct": 0.10,
        "max_sector_weight": 0.25
    }

class ConfigManager:
    """
    Handles loading and validation of configuration files.
    """
    def get_config(self, path: Optional[str] = None) -> ProductionConfig:
        if not path or not os.path.exists(path):
            return ProductionConfig()
            
        with open(path, "r") as f:
            data = yaml.safe_load(f)
            
        return ProductionConfig(**data)

config_manager = ConfigManager()
