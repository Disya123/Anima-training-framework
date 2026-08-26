from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Iterable, Mapping


class ConceptMode(StrEnum):
    STYLE = "style"
    CHARACTER = "character"
    OBJECT = "object"
    GENERAL = "general"


@dataclass(frozen=True)
class ConceptPolicy:
    requires_trigger: bool
    default_learning_rate: float
    invariant_facets: tuple[str, ...]
    description: str


POLICIES: dict[ConceptMode, ConceptPolicy] = {
    ConceptMode.STYLE: ConceptPolicy(
        True,
        2e-5,
        ("content", "subject_count", "scene"),
        "trigger carries rendering style; caption carries image content",
    ),
    ConceptMode.CHARACTER: ConceptPolicy(
        True,
        2e-5,
        ("pose", "camera", "clothing", "background", "expression", "lighting"),
        "trigger carries identity while captions expose changing attributes",
    ),
    ConceptMode.OBJECT: ConceptPolicy(
        True,
        2e-5,
        ("view", "scale", "occlusion", "scene", "lighting"),
        "trigger carries object identity while captions expose context",
    ),
    ConceptMode.GENERAL: ConceptPolicy(
        False,
        5e-6,
        ("pose", "camera", "interaction", "composition"),
        "broad prior correction; no concept trigger is required",
    ),
}


def policy_for(mode: str | ConceptMode) -> ConceptPolicy:
    return POLICIES[ConceptMode(mode)]


def normalize_caption(caption: str) -> str:
    return re.sub(r"\s+", " ", caption.strip()).strip(" ,")


def build_prompt(
    caption: str,
    *,
    mode: str | ConceptMode,
    record_trigger: str | None = None,
    global_trigger: str | None = None,
    trigger_position: str = "prefix",
) -> tuple[str, str, str | None]:
    """Return (triggered prompt, content-only prompt, effective trigger)."""
    concept_mode = ConceptMode(mode)
    content = normalize_caption(caption)
    trigger = normalize_caption(record_trigger or global_trigger or "") or None
    if policy_for(concept_mode).requires_trigger and not trigger:
        raise ValueError(f"{concept_mode.value} records require a trigger")
    if policy_for(concept_mode).requires_trigger and not content:
        raise ValueError(f"{concept_mode.value} captions must describe content, not only the trigger")
    if trigger and trigger in content:
        # Accept legacy captions but ensure the no-trigger preservation prompt is real.
        content = normalize_caption(content.replace(trigger, ""))
    if not trigger:
        return content, content, None
    if trigger_position == "prefix":
        return f"{trigger}, {content}" if content else trigger, content, trigger
    if trigger_position == "suffix":
        return f"{content}, {trigger}" if content else trigger, content, trigger
    raise ValueError(f"unsupported trigger_position: {trigger_position}")


def effective_weight(base_weight: float, hard_tags: Iterable[str], multipliers: Mapping[str, float]) -> float:
    value = float(base_weight)
    if value <= 0:
        raise ValueError("sample weight must be positive")
    for tag in set(hard_tags):
        value *= float(multipliers.get(tag, 1.0))
    return value


def audit_records(records: Iterable[Mapping[str, Any]], mode: str | ConceptMode) -> dict[str, Any]:
    records = list(records)
    concept_mode = ConceptMode(mode)
    policy = policy_for(concept_mode)
    triggers = sorted({str(r.get("trigger", "")).strip() for r in records if r.get("trigger")})
    missing_captions = sum(not normalize_caption(str(r.get("caption", ""))) for r in records)
    facet_coverage: dict[str, dict[str, Any]] = {}
    for facet in policy.invariant_facets:
        values: set[str] = set()
        present = 0
        for record in records:
            facets = record.get("facets") or {}
            if facet not in facets:
                continue
            present += 1
            raw = facets[facet]
            if isinstance(raw, (list, tuple, set)):
                values.update(str(v) for v in raw)
            else:
                values.add(str(raw))
        facet_coverage[facet] = {
            "present": present,
            "coverage": present / max(1, len(records)),
            "unique": len(values),
            "values": sorted(values),
        }
    warnings: list[str] = []
    if policy.requires_trigger and not triggers:
        warnings.append("no trigger found")
    if len(triggers) > 1:
        warnings.append(f"multiple triggers in one run: {triggers}")
    if missing_captions:
        warnings.append(f"{missing_captions} records have empty content captions")
    for facet, stats in facet_coverage.items():
        if stats["coverage"] < 0.5:
            warnings.append(f"facet '{facet}' is annotated on less than half of records")
        elif stats["unique"] < 3:
            warnings.append(f"facet '{facet}' has only {stats['unique']} unique values")
    return {
        "mode": concept_mode.value,
        "records": len(records),
        "triggers": triggers,
        "missing_captions": missing_captions,
        "facet_coverage": facet_coverage,
        "warnings": warnings,
    }

