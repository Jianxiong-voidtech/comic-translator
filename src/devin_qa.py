import json
from dataclasses import asdict, dataclass
from typing import List
from src.pipeline import TextRegion


@dataclass
class QAIssue:
    source_text: str
    translated_text: str
    severity: str
    reason: str
    suggested_fix: str


def run_devin_qa(regions: List[TextRegion], max_expansion_ratio: float = 2.6) -> dict:
    """
    Devin Bot QA check:
    Flags overly long translations that risk overflowing speech bubbles
    and suggests code/prompt adjustments for PR review.
    """
    issues: List[QAIssue] = []

    for r in regions:
        src_len = max(1, len(r.source_text))
        dst_len = len(r.translated_text)
        ratio = round(dst_len / src_len, 2)

        if ratio > max_expansion_ratio:
            issues.append(
                QAIssue(
                    source_text=r.source_text,
                    translated_text=r.translated_text,
                    severity="WARN",
                    reason=f"Translation expansion ratio {ratio}x exceeds threshold {max_expansion_ratio}x.",
                    suggested_fix="Tighten LLM system prompt length constraint or lower font size in render_translated_panel().",
                )
            )

    return {
        "agent": "devin-bot-qa",
        "total_regions": len(regions),
        "issues_found": len(issues),
        "status": "NEEDS_ATTENTION" if issues else "PASS",
        "issues": [asdict(i) for i in issues],
    }


def save_qa_report(report: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
