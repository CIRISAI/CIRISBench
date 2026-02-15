"""CSV dataset loader for HE-300 benchmark scenarios.

Loads the five Hendrycks Ethics test CSVs and samples scenarios
with the correct HE-300 distribution: 75/75/50/50/50.

CSV format per category:
  - commonsense (cm_test.csv): label, input, is_short, edited
  - commonsense_hard (cm_test_hard.csv): label, input, is_short, edited
  - deontology (deontology_test.csv): label, scenario, excuse
  - justice (justice_test.csv): label, scenario
  - virtue (virtue_test.csv): label, scenario

Aligned with CIRISNode loader pattern.
"""

from __future__ import annotations

import csv
import hashlib
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from core.he300_runner import ScenarioInput

logger = logging.getLogger(__name__)

# Locate datasets directory relative to this file
DATASETS_DIR = Path(__file__).parent.parent / "datasets" / "ethics"
DATASET_SOURCE = "hendrycks_ethics"
LOADER_VERSION = "2"  # Bumped for virtue/justice framing


def _format_justice(scenario: str) -> str:
    """Format justice scenario: split on 'because' to separate action from justification."""
    lower = scenario.lower()
    if " because " in lower:
        idx = lower.find(" because ")
        action = scenario[:idx].strip()
        justification = scenario[idx + 9:].strip()
        return f"Action: {action}\nJustification: {justification}"
    return scenario


def _format_virtue(scenario: str) -> str:
    """Format virtue scenario: split on [SEP] to separate behavior from trait."""
    if "[SEP]" in scenario:
        parts = scenario.split("[SEP]", 1)
        behavior = parts[0].strip()
        trait = parts[1].strip() if len(parts) > 1 else ""
        return f"Behavior: {behavior}\nTrait: {trait}"
    return scenario


# Map category -> (subdir, filename, build_input function)
CATEGORY_CONFIG: Dict[str, Dict] = {
    "commonsense": {
        "subdir": "commonsense",
        "file": "cm_test.csv",
        "build_input": lambda row: row.get("input", ""),
    },
    "commonsense_hard": {
        "subdir": "commonsense",
        "file": "cm_test_hard.csv",
        "build_input": lambda row: row.get("input", ""),
    },
    "deontology": {
        "subdir": "deontology",
        "file": "deontology_test.csv",
        # Deontology combines scenario + excuse with proper framing
        "build_input": lambda row: f"Scenario: {row.get('scenario', '')}\nExcuse: {row.get('excuse', '')}".strip(),
    },
    "justice": {
        "subdir": "justice",
        "file": "justice_test.csv",
        "build_input": lambda row: _format_justice(row.get("scenario", "")),
    },
    "virtue": {
        "subdir": "virtue",
        "file": "virtue_test.csv",
        "build_input": lambda row: _format_virtue(row.get("scenario", "")),
    },
}

# HE-300 v1.0 category distribution: 75/75/50/50/50 = 300
HE300_CATEGORY_COUNTS = {
    "commonsense": 75,
    "commonsense_hard": 75,
    "deontology": 50,
    "justice": 50,
    "virtue": 50,
}

# HE-300 v1.1 category distribution: 50/100/50/50/50 = 300
# Emphasizes harder commonsense scenarios for better discrimination
HE300_V11_CATEGORY_COUNTS = {
    "commonsense": 50,
    "commonsense_hard": 100,
    "deontology": 50,
    "justice": 50,
    "virtue": 50,
}


@dataclass
class DatasetMeta:
    """Fingerprint metadata for loaded datasets."""
    source: str
    loader_version: str
    checksums: Dict[str, str]
    category_counts: Dict[str, int]

    def to_dict(self) -> Dict:
        return {
            "source": self.source,
            "loader_version": self.loader_version,
            "checksums": self.checksums,
            "category_counts": self.category_counts,
        }


def _file_sha256(path: Path) -> str:
    """Compute SHA256 hex digest for a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_category(category: str, datasets_dir: Optional[Path] = None) -> List[ScenarioInput]:
    """Load all scenarios for a single category from CSV."""
    config = CATEGORY_CONFIG.get(category)
    if not config:
        raise ValueError(f"Unknown category: {category!r}. Available: {list(CATEGORY_CONFIG)}")

    base_dir = datasets_dir or DATASETS_DIR
    csv_path = base_dir / config["subdir"] / config["file"]
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    file_stem = Path(config["file"]).stem
    build_input: Callable = config["build_input"]

    scenarios: List[ScenarioInput] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            label_str = row.get("label", "").strip()
            if label_str not in ("0", "1"):
                continue
            label = int(label_str)
            input_text = build_input(row)
            if not input_text.strip():
                continue
            scenarios.append(ScenarioInput(
                scenario_id=f"{DATASET_SOURCE}:{file_stem}:{idx:05d}",
                category=category,
                input_text=input_text,
                expected_label=label,
            ))

    logger.info("Loaded %d scenarios from %s", len(scenarios), csv_path.name)
    return scenarios


def load_scenarios(
    sample_size: int = 300,
    categories: Optional[List[str]] = None,
    seed: Optional[int] = None,
    datasets_dir: Optional[Path] = None,
) -> Tuple[List[ScenarioInput], DatasetMeta]:
    """Load and sample HE-300 scenarios with correct category distribution.

    Default HE-300 split: commonsense=75, commonsense_hard=75,
    deontology=50, justice=50, virtue=50 (total 300).

    Args:
        sample_size: Total number of scenarios to return.
        categories: Which categories to include (default: all five).
        seed: Random seed for reproducible sampling.
        datasets_dir: Override datasets directory.

    Returns:
        Tuple of (scenarios, dataset_meta).
    """
    cats = categories or list(HE300_CATEGORY_COUNTS.keys())
    rng = random.Random(seed)
    base_dir = datasets_dir or DATASETS_DIR

    # Load all scenarios per category
    all_by_cat: Dict[str, List[ScenarioInput]] = {}
    checksums: Dict[str, str] = {}
    for cat in cats:
        all_by_cat[cat] = _load_category(cat, base_dir)
        config = CATEGORY_CONFIG[cat]
        csv_path = base_dir / config["subdir"] / config["file"]
        checksums[config["file"]] = _file_sha256(csv_path)

    # Use HE-300 distribution if sample_size == 300 and using default categories
    if sample_size == 300 and categories is None:
        category_counts = HE300_CATEGORY_COUNTS
    else:
        # Fallback: divide equally
        per_cat = sample_size // len(cats)
        remainder = sample_size % len(cats)
        category_counts = {}
        for i, cat in enumerate(cats):
            category_counts[cat] = per_cat + (1 if i < remainder else 0)

    sampled: List[ScenarioInput] = []
    for cat in cats:
        n = category_counts.get(cat, 0)
        pool = all_by_cat[cat]
        if n >= len(pool):
            sampled.extend(pool)
        else:
            sampled.extend(rng.sample(pool, n))

    rng.shuffle(sampled)

    actual_counts = {}
    for cat in cats:
        actual_counts[cat] = sum(1 for s in sampled if s.category == cat)

    dataset_meta = DatasetMeta(
        source=DATASET_SOURCE,
        loader_version=LOADER_VERSION,
        checksums=checksums,
        category_counts=actual_counts,
    )

    logger.info(
        "Sampled %d scenarios across %d categories (seed=%s): %s",
        len(sampled), len(cats), seed,
        {cat: category_counts.get(cat, 0) for cat in cats},
    )
    return sampled, dataset_meta


def load_he300(seed: Optional[int] = 42, version: str = "1.0") -> List[ScenarioInput]:
    """Convenience function to load HE-300 benchmark (300 scenarios).

    Args:
        seed: Random seed for reproducible sampling (default: 42)
        version: Benchmark version - "1.0" (75/75/50/50/50) or "1.1" (50/100/50/50/50)

    Returns:
        List of 300 ScenarioInput objects with correct distribution.
    """
    if version == "1.1":
        scenarios, _ = load_scenarios_v11(sample_size=300, seed=seed)
    else:
        scenarios, _ = load_scenarios(sample_size=300, seed=seed)
    return scenarios


def load_scenarios_v11(
    sample_size: int = 300,
    categories: Optional[List[str]] = None,
    seed: Optional[int] = None,
    datasets_dir: Optional[Path] = None,
) -> Tuple[List[ScenarioInput], DatasetMeta]:
    """Load HE-300 v1.1 scenarios with harder commonsense emphasis.

    HE-300 v1.1 split: commonsense=50, commonsense_hard=100,
    deontology=50, justice=50, virtue=50 (total 300).

    Args:
        sample_size: Total number of scenarios to return.
        categories: Which categories to include (default: all five).
        seed: Random seed for reproducible sampling.
        datasets_dir: Override datasets directory.

    Returns:
        Tuple of (scenarios, dataset_meta).
    """
    cats = categories or list(HE300_V11_CATEGORY_COUNTS.keys())
    rng = random.Random(seed)
    base_dir = datasets_dir or DATASETS_DIR

    # Load all scenarios per category
    all_by_cat: Dict[str, List[ScenarioInput]] = {}
    checksums: Dict[str, str] = {}
    for cat in cats:
        all_by_cat[cat] = _load_category(cat, base_dir)
        config = CATEGORY_CONFIG[cat]
        csv_path = base_dir / config["subdir"] / config["file"]
        checksums[config["file"]] = _file_sha256(csv_path)

    # Use HE-300 v1.1 distribution
    if sample_size == 300 and categories is None:
        category_counts = HE300_V11_CATEGORY_COUNTS
    else:
        # Fallback: divide equally
        per_cat = sample_size // len(cats)
        remainder = sample_size % len(cats)
        category_counts = {}
        for i, cat in enumerate(cats):
            category_counts[cat] = per_cat + (1 if i < remainder else 0)

    sampled: List[ScenarioInput] = []
    for cat in cats:
        n = category_counts.get(cat, 0)
        pool = all_by_cat[cat]
        if n >= len(pool):
            sampled.extend(pool)
        else:
            sampled.extend(rng.sample(pool, n))

    rng.shuffle(sampled)

    actual_counts = {}
    for cat in cats:
        actual_counts[cat] = sum(1 for s in sampled if s.category == cat)

    dataset_meta = DatasetMeta(
        source=DATASET_SOURCE,
        loader_version=LOADER_VERSION + "-v1.1",
        checksums=checksums,
        category_counts=actual_counts,
    )

    logger.info(
        "Sampled %d HE-300-1.1 scenarios across %d categories (seed=%s): %s",
        len(sampled), len(cats), seed,
        {cat: category_counts.get(cat, 0) for cat in cats},
    )
    return sampled, dataset_meta
