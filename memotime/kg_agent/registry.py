

# =============================
# file: kg_agent/registry.py
# =============================
import json
from pathlib import Path
from typing import Any, Dict, List, Union, Optional

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

from .cards_seed import SEED_TEMPLATES

TEMPLATES_DIR = Path("templates")
INDEX_PATH = Path("templates_index.json")

class TemplateCard:
    def __init__(self, workflow_id: str, spec: Dict[str, Any]):
        self.workflow_id = workflow_id
        self.spec = spec

    @staticmethod
    def from_yaml_text(text: str) -> "TemplateCard":
        if yaml is None:
            raise RuntimeError("pyyaml not installed")
        spec = yaml.safe_load(text)
        return TemplateCard(workflow_id=spec.get("workflow_id", "unknown"), spec=spec)

    def to_yaml_text(self) -> str:
        if yaml is None:
            raise RuntimeError("pyyaml not installed")
        return yaml.safe_dump(self.spec, sort_keys=False, allow_unicode=True)

class TemplateRegistry:
    def __init__(self, dir_path: Path = TEMPLATES_DIR, index_path: Path = INDEX_PATH):
        self.dir = dir_path
        self.index_path = index_path
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index: Dict[str, Dict[str, Any]] = {}
        self._load_index()
        self._ensure_seed_templates()

    def _load_index(self) -> None:
        if self.index_path.exists():
            self.index = json.loads(self.index_path.read_text(encoding="utf-8"))
        else:
            self.index = {}

    def _save_index(self) -> None:
        self.index_path.write_text(json.dumps(self.index, ensure_ascii=False, indent=2), encoding="utf-8")

    def _ensure_seed_templates(self) -> None:
        for fname, content in SEED_TEMPLATES.items():
            path = self.dir / fname
            if not path.exists():
                if yaml is None:
                    raise RuntimeError("pyyaml required to init templates")
                path.write_text(yaml.safe_dump(content, sort_keys=False, allow_unicode=True), encoding="utf-8")
                wid = content.get("workflow_id", path.stem)
                if wid not in self.index:
                    self.index[wid] = {"name": wid, "examples": [], "file": str(path)}
        self._save_index()

    def list_templates(self) -> List[str]:
        return list(self.index.keys())

    def load_card(self, workflow_id: str) -> TemplateCard:
        entry = self.index.get(workflow_id)
        if not entry:
            raise KeyError(f"Unknown workflow_id: {workflow_id}")
        path = Path(entry["file"]) if entry.get("file") else (self.dir / f"{workflow_id}.yaml")
        spec = yaml.safe_load(path.read_text(encoding="utf-8"))
        return TemplateCard(workflow_id=workflow_id, spec=spec)

    def add_card(self, workflow_id: str, yaml_text: str, example_question: Optional[str] = None) -> None:
        path = self.dir / f"{workflow_id}.yaml"
        path.write_text(yaml_text, encoding="utf-8")
        self.index[workflow_id] = {"name": workflow_id, "examples": [] if example_question is None else [example_question], "file": str(path)}
        self._save_index()

    def add_example(self, workflow_id: str, example_question: str) -> None:
        if workflow_id not in self.index:
            raise KeyError(f"Unknown workflow_id: {workflow_id}")
        self.index[workflow_id].setdefault("examples", []).append(example_question)
        self._save_index()
