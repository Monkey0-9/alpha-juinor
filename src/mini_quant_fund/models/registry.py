"""
models/registry.py

Section E: Model Registry.
Tracks lifecycle of models (CANDIDATE -> SHADOW -> PROD -> RETIRED).
"""
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, Optional
from datetime import datetime

STATUS_CANDIDATE = "CANDIDATE"
STATUS_SHADOW = "SHADOW"
STATUS_PROD = "PROD"
STATUS_RETIRED = "RETIRED"

@dataclass
class ModelMetadata:
    model_id: str
    name: str
    version: str
    status: str
    created_at: str
    metrics: Dict[str, float]
    approved_by: Optional[str] = None
    shadow_start_date: Optional[str] = None
    prod_start_date: Optional[str] = None

class ModelRegistry:
    def __init__(self, registry_path="runtime/model_registry.json"):
        self.path = registry_path
        self.models: Dict[str, ModelMetadata] = {}
        self.load()

    def load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, 'r') as f:
                    data = json.load(f)
                    self.models = {k: ModelMetadata(**v) for k, v in data.items()}
            except Exception:
                self.models = {}

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, 'w') as f:
            data = {k: asdict(v) for k, v in self.models.items()}
            json.dump(data, f, indent=2)

    def register(self, name: str, version: str, metrics: Dict) -> str:
        mid = f"{name}_{version}"
        meta = ModelMetadata(
            model_id=mid,
            name=name,
            version=version,
            status=STATUS_CANDIDATE,
            created_at=datetime.utcnow().isoformat(),
            metrics=metrics
        )
        self.models[mid] = meta
        self.save()
        return mid

    def update_status(self, model_id: str, new_status: str, user: str = "SYSTEM"):
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")

        m = self.models[model_id]
        m.status = new_status
        if new_status == STATUS_SHADOW:
            m.shadow_start_date = datetime.utcnow().isoformat()
        elif new_status == STATUS_PROD:
            m.prod_start_date = datetime.utcnow().isoformat()
            m.approved_by = user

        self.save()

    def get_prod_model(self, name: str) -> Optional[ModelMetadata]:
        for m in self.models.values():
            if m.name == name and m.status == STATUS_PROD:
                return m
        return None
