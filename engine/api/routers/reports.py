# -*- coding: utf-8 -*-
"""
Report Generation API Router

Generates signed static reports in Jekyll-compatible formats for GitHub Pages.
Reports include cryptographic signatures for integrity verification.
"""

import logging
import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any
from enum import Enum

from fastapi import APIRouter, HTTPException, status, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Add project root to path for imports
import sys
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/reports",
    tags=["reports", "export"],
    responses={
        404: {"description": "Not found"},
        500: {"description": "Internal server error"},
    },
)

# --- Constants ---
REPORTS_DIR = Path(project_root) / "data" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Benchmark results directory (where HE-300 stores batch results)
BENCHMARK_RESULTS_DIR = Path(project_root) / "data" / "benchmark_results"
BENCHMARK_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --- Schemas ---
class ReportFormat(str, Enum):
    """Available report output formats."""
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"


class CategoryResult(BaseModel):
    """Results for a single category."""
    category: str
    total: int
    correct: int
    accuracy: float
    avg_latency_ms: float
    errors: int = 0


class BenchmarkRunSummary(BaseModel):
    """Summary of a benchmark run for reporting."""
    batch_id: str
    model_name: str
    identity_id: str
    guidance_id: str
    total_scenarios: int
    correct_predictions: int
    overall_accuracy: float
    avg_latency_ms: float
    total_errors: int
    categories: List[CategoryResult]
    started_at: str
    completed_at: str
    processing_time_ms: float
    # Agent metadata for clear identification
    agent_name: str = ""  # Name of the agent being tested
    agent_type: str = ""  # "base_llm", "eee_purple", "ciris_agent"
    protocol: str = ""  # "a2a", "mcp", or "" for direct API
    agent_url: str = ""  # URL of the agent endpoint


class EvaluationDetailReport(BaseModel):
    """Evaluation detail for heuristic or semantic methods."""
    classification: str = ""  # "ethical", "unethical", "unknown"
    label: Optional[int] = None  # 0 = ethical, 1 = unethical
    confidence: float = 0.0
    method: str = ""  # "heuristic" or "semantic"


class ScenarioDetail(BaseModel):
    """Individual scenario result for detailed reports."""
    scenario_id: str  # UID for relating back to source dataset
    category: str  # question type/category from dataset
    input_text: str  # the scenario prompt text
    expected_label: Optional[int]
    predicted_label: Optional[int]
    model_response: str
    is_correct: bool
    latency_ms: float
    error: Optional[str] = None
    trace_id: Optional[str] = None
    trace_url: Optional[str] = None
    # Dual evaluation results per HE-300 spec
    heuristic_eval: Optional[EvaluationDetailReport] = None
    semantic_eval: Optional[EvaluationDetailReport] = None
    evaluations_agree: bool = True
    disagreement_note: Optional[str] = None


class ReportRequest(BaseModel):
    """Request to generate a report."""
    batch_id: str = Field(..., description="Batch ID from benchmark run")
    summary: BenchmarkRunSummary
    scenarios: List[ScenarioDetail] = Field(default_factory=list)
    format: ReportFormat = ReportFormat.MARKDOWN
    include_scenarios: bool = Field(default=True, description="Include individual scenario results")
    sign_report: bool = Field(default=True, description="Add cryptographic signature")
    jekyll_frontmatter: bool = Field(default=True, description="Include Jekyll YAML frontmatter")
    title: Optional[str] = None
    description: Optional[str] = None


class ReportSignature(BaseModel):
    """Cryptographic signature for report integrity."""
    algorithm: str = "sha256"
    content_hash: str
    timestamp: str
    signer_id: str = "ethicsengine-enterprise"
    signature: str  # HMAC-SHA256 of content_hash + timestamp


class ReportMetadata(BaseModel):
    """Metadata for a generated report."""
    report_id: str
    batch_id: str
    model_name: str = ""
    accuracy: float = 0.0
    format: ReportFormat
    created_at: str
    file_path: str
    file_size: int
    signature: Optional[ReportSignature] = None


class ReportResponse(BaseModel):
    """Response after generating a report."""
    status: str
    report_id: str
    metadata: ReportMetadata
    download_url: str
    preview: Optional[str] = None


class ReportListResponse(BaseModel):
    """List of generated reports."""
    reports: List[ReportMetadata]
    total: int


class BenchmarkResultItem(BaseModel):
    """A benchmark result item for the UI results list."""
    id: str
    model_name: str
    report_name: str = "HE-300 Benchmark"
    created_at: str
    scores: Dict[str, float]
    status: str = "completed"


class BenchmarkResultsResponse(BaseModel):
    """Response containing available benchmark results."""
    results: List[BenchmarkResultItem]
    total: int


# --- Signing Utilities ---
def get_signing_key() -> bytes:
    """Get or generate signing key."""
    key = os.environ.get("REPORT_SIGNING_KEY", "")
    if not key:
        # Generate a deterministic key from machine ID for consistency
        # In production, use a proper secret management system
        key = "ethicsengine-default-signing-key-change-in-production"
    return key.encode()


def sign_content(content: str) -> ReportSignature:
    """Create a signature for report content."""
    import hmac
    
    # Hash the content
    content_hash = hashlib.sha256(content.encode()).hexdigest()
    timestamp = datetime.now(timezone.utc).isoformat()
    
    # Create HMAC signature
    message = f"{content_hash}:{timestamp}".encode()
    signature = hmac.new(get_signing_key(), message, hashlib.sha256).hexdigest()
    
    return ReportSignature(
        algorithm="sha256+hmac",
        content_hash=content_hash,
        timestamp=timestamp,
        signature=signature
    )


def verify_signature(content: str, signature: ReportSignature) -> bool:
    """Verify a report signature."""
    import hmac
    
    # Recalculate content hash
    content_hash = hashlib.sha256(content.encode()).hexdigest()
    if content_hash != signature.content_hash:
        return False
    
    # Verify HMAC
    message = f"{signature.content_hash}:{signature.timestamp}".encode()
    expected = hmac.new(get_signing_key(), message, hashlib.sha256).hexdigest()
    
    return hmac.compare_digest(signature.signature, expected)


# --- Report Generation ---
def generate_markdown_report(request: ReportRequest, signature: Optional[ReportSignature]) -> str:
    """Generate a Markdown report with optional Jekyll frontmatter."""
    lines = []
    summary = request.summary
    
    # Jekyll frontmatter
    if request.jekyll_frontmatter:
        title = request.title or f"HE-300 Benchmark Report - {summary.batch_id}"
        description = request.description or f"Ethics benchmark results for {summary.model_name}"
        
        lines.append("---")
        lines.append(f"layout: report")
        lines.append(f"title: \"{title}\"")
        lines.append(f"description: \"{description}\"")
        lines.append(f"date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %z')}")
        lines.append(f"batch_id: \"{summary.batch_id}\"")
        lines.append(f"model: \"{summary.model_name}\"")
        lines.append(f"accuracy: {summary.overall_accuracy:.4f}")
        lines.append(f"categories:")
        for cat in summary.categories:
            lines.append(f"  - name: {cat.category}")
            lines.append(f"    accuracy: {cat.accuracy:.4f}")
        if signature:
            lines.append(f"signature:")
            lines.append(f"  hash: \"{signature.content_hash}\"")
            lines.append(f"  algorithm: \"{signature.algorithm}\"")
            lines.append(f"  timestamp: \"{signature.timestamp}\"")
        lines.append("---")
        lines.append("")
    
    # Title and summary
    lines.append(f"# HE-300 Ethics Benchmark Report")
    lines.append("")
    lines.append(f"**Batch ID:** `{summary.batch_id}`  ")
    lines.append(f"**Model:** `{summary.model_name}`  ")
    lines.append(f"**Identity:** {summary.identity_id}  ")
    lines.append(f"**Guidance:** {summary.guidance_id}  ")
    lines.append(f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append("")
    
    # Overall Results
    lines.append("## Overall Results")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total Scenarios | {summary.total_scenarios} |")
    lines.append(f"| Correct Predictions | {summary.correct_predictions} |")
    lines.append(f"| **Overall Accuracy** | **{summary.overall_accuracy:.2%}** |")
    lines.append(f"| Average Latency | {summary.avg_latency_ms:.1f} ms |")
    lines.append(f"| Total Errors | {summary.total_errors} |")
    lines.append(f"| Processing Time | {summary.processing_time_ms:.1f} ms |")
    lines.append("")
    
    # Category Breakdown
    lines.append("## Results by Category")
    lines.append("")
    lines.append("| Category | Total | Correct | Accuracy | Avg Latency |")
    lines.append("|----------|-------|---------|----------|-------------|")
    for cat in summary.categories:
        lines.append(f"| {cat.category} | {cat.total} | {cat.correct} | {cat.accuracy:.2%} | {cat.avg_latency_ms:.1f} ms |")
    lines.append("")
    
    # Accuracy chart (text-based for Markdown compatibility)
    lines.append("### Accuracy by Category")
    lines.append("")
    lines.append("```")
    max_width = 40
    for cat in summary.categories:
        bar_len = int(cat.accuracy * max_width)
        bar = "█" * bar_len + "░" * (max_width - bar_len)
        lines.append(f"{cat.category:20} |{bar}| {cat.accuracy:.1%}")
    lines.append("```")
    lines.append("")
    
    # Individual scenarios (if requested and available)
    if request.include_scenarios and request.scenarios:
        lines.append("## Scenario Details")
        lines.append("")
        lines.append("<details>")
        lines.append("<summary>Click to expand individual scenario results</summary>")
        lines.append("")
        
        # Group by category
        by_category: Dict[str, List[ScenarioDetail]] = {}
        for s in request.scenarios:
            if s.category not in by_category:
                by_category[s.category] = []
            by_category[s.category].append(s)
        
        for category, scenarios in by_category.items():
            lines.append(f"### {category.title()}")
            lines.append("")
            
            for s in scenarios[:10]:  # Limit to first 10 per category to avoid huge reports
                status_icon = "✅" if s.is_correct else "❌"
                lines.append(f"#### {s.scenario_id} {status_icon}")
                lines.append("")
                lines.append(f"**Input:** {s.input_text[:200]}{'...' if len(s.input_text) > 200 else ''}")
                lines.append("")
                lines.append(f"**Expected:** {'UNETHICAL' if s.expected_label == 1 else 'ETHICAL'} | **Predicted:** {'UNETHICAL' if s.predicted_label == 1 else 'ETHICAL'}")
                lines.append("")
                if s.model_response:
                    lines.append(f"> {s.model_response[:300]}{'...' if len(s.model_response) > 300 else ''}")
                lines.append("")
            
            if len(scenarios) > 10:
                lines.append(f"*...and {len(scenarios) - 10} more scenarios in this category*")
                lines.append("")
        
        lines.append("</details>")
        lines.append("")
    
    # Signature block
    if signature:
        lines.append("---")
        lines.append("")
        lines.append("## Report Integrity")
        lines.append("")
        lines.append("This report is cryptographically signed for integrity verification.")
        lines.append("")
        lines.append("```")
        lines.append(f"Algorithm: {signature.algorithm}")
        lines.append(f"Content Hash: {signature.content_hash}")
        lines.append(f"Timestamp: {signature.timestamp}")
        lines.append(f"Signature: {signature.signature}")
        lines.append("```")
        lines.append("")
        lines.append("To verify: recalculate SHA-256 of report content (excluding this signature block) and compare.")
        lines.append("")
    
    # Footer
    lines.append("---")
    lines.append("")
    lines.append("*Generated by [EthicsEngine Enterprise](https://github.com/rng-ops/ethicsengine_enterprise) HE-300 Benchmark System*")
    
    return "\n".join(lines)


def generate_html_report(request: ReportRequest, signature: Optional[ReportSignature]) -> str:
    """Generate an HTML report with full test data and export capabilities."""
    summary = request.summary
    title = request.title or f"HE-300 Benchmark Report - {summary.batch_id}"
    
    # Calculate category data for chart
    cat_labels = [cat.category for cat in summary.categories]
    cat_accuracies = [cat.accuracy * 100 for cat in summary.categories]
    
    # Build signature block separately to avoid nested f-string issues
    if signature:
        signature_block = f'''
        <div class="signature">
            <strong>&#128272; Report Signature</strong><br>
            Algorithm: {signature.algorithm}<br>
            Content Hash: {signature.content_hash}<br>
            Timestamp: {signature.timestamp}<br>
            Signature: {signature.signature}
        </div>
        '''
        signature_json = json.dumps({
            "algorithm": signature.algorithm,
            "content_hash": signature.content_hash,
            "timestamp": signature.timestamp,
            "signature": signature.signature
        })
    else:
        signature_block = ''
        signature_json = 'null'
    
    # Build scenarios HTML table
    # Label mapping: 0 = ETHICAL, 1 = UNETHICAL (per Hendrycks Ethics dataset convention)
    scenarios_rows = []
    for s in request.scenarios:
        status_class = 'correct' if s.is_correct else 'incorrect'
        expected = 'UNETHICAL' if s.expected_label == 1 else 'ETHICAL' if s.expected_label == 0 else 'N/A'
        predicted = 'UNETHICAL' if s.predicted_label == 1 else 'ETHICAL' if s.predicted_label == 0 else 'ERROR'
        # Escape HTML in text content
        input_text_escaped = s.input_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')
        response_escaped = s.model_response.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;') if s.model_response else ''

        # Build trace link HTML if available
        trace_link_html = ''
        if s.trace_id and s.trace_url:
            trace_link_html = f'<a href="{s.trace_url}" target="_blank" rel="noopener noreferrer" title="View LangSmith trace: {s.trace_id}">&#128279;</a>'
        elif s.trace_id:
            trace_link_html = f'<span title="Trace ID: {s.trace_id}">&#128279;</span>'
        else:
            trace_link_html = '<span style="color: #ccc;">-</span>'

        # Build heuristic/semantic evaluation badges
        heuristic_html = '-'
        semantic_html = '-'
        agree_icon = ''

        if s.heuristic_eval:
            h_class = 'correct' if s.heuristic_eval.classification == ('unethical' if s.expected_label == 1 else 'ethical') else 'incorrect'
            heuristic_html = f'<span class="eval-badge {h_class}" title="Confidence: {s.heuristic_eval.confidence:.0%}">{s.heuristic_eval.classification.upper()}</span>'

        if s.semantic_eval:
            sem_class = 'correct' if s.semantic_eval.classification == ('unethical' if s.expected_label == 1 else 'ethical') else 'incorrect'
            semantic_html = f'<span class="eval-badge {sem_class}" title="Confidence: {s.semantic_eval.confidence:.0%}">{s.semantic_eval.classification.upper()}</span>'

        if s.heuristic_eval and s.semantic_eval:
            if s.evaluations_agree:
                agree_icon = '<span title="Evaluations agree" style="color: #22c55e;">&#10003;</span>'
            else:
                agree_icon = f'<span title="{s.disagreement_note or "Disagreement"}" style="color: #f59e0b;">&#9888;</span>'

        scenarios_rows.append(f'''
        <tr class="scenario-row {status_class}" data-category="{s.category}">
            <td><code>{s.scenario_id}</code></td>
            <td><span class="category-badge">{s.category}</span></td>
            <td class="input-text" title="{input_text_escaped}">{input_text_escaped[:80]}{'...' if len(s.input_text) > 80 else ''}</td>
            <td><span class="label-badge expected">{expected}</span></td>
            <td>{heuristic_html}</td>
            <td>{semantic_html}</td>
            <td>{agree_icon}</td>
            <td><span class="label-badge predicted {status_class}">{predicted}</span></td>
            <td><span class="status-icon">{("&#10003;" if s.is_correct else "&#10007;")}</span></td>
            <td>{s.latency_ms:.0f}ms</td>
            <td style="text-align: center;">{trace_link_html}</td>
            <td>
                <button class="btn-small" onclick="showDetails('{s.scenario_id}')">View</button>
            </td>
        </tr>
        ''')
    
    scenarios_table = ''.join(scenarios_rows)
    
    # Build scenarios JSON for export (properly escaped)
    scenarios_json = json.dumps([{
        "scenario_id": s.scenario_id,
        "category": s.category,
        "input_text": s.input_text,
        "expected_label": s.expected_label,
        "predicted_label": s.predicted_label,
        "model_response": s.model_response,
        "is_correct": s.is_correct,
        "latency_ms": s.latency_ms,
        "error": s.error,
        "trace_id": s.trace_id,
        "trace_url": s.trace_url,
        "heuristic_eval": s.heuristic_eval.model_dump() if s.heuristic_eval else None,
        "semantic_eval": s.semantic_eval.model_dump() if s.semantic_eval else None,
        "evaluations_agree": s.evaluations_agree,
        "disagreement_note": s.disagreement_note
    } for s in request.scenarios], indent=2)
    
    # Build summary JSON
    summary_json = json.dumps({
        "batch_id": summary.batch_id,
        "model_name": summary.model_name,
        "agent_name": summary.agent_name,
        "agent_type": summary.agent_type,
        "protocol": summary.protocol,
        "agent_url": summary.agent_url,
        "identity_id": summary.identity_id,
        "guidance_id": summary.guidance_id,
        "total_scenarios": summary.total_scenarios,
        "correct_predictions": summary.correct_predictions,
        "overall_accuracy": summary.overall_accuracy,
        "avg_latency_ms": summary.avg_latency_ms,
        "total_errors": summary.total_errors,
        "categories": [{
            "category": c.category,
            "total": c.total,
            "correct": c.correct,
            "accuracy": c.accuracy,
            "avg_latency_ms": c.avg_latency_ms,
            "errors": c.errors
        } for c in summary.categories],
        "started_at": summary.started_at,
        "completed_at": summary.completed_at,
        "processing_time_ms": summary.processing_time_ms
    }, indent=2)
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        :root {{
            --primary: #4f46e5;
            --success: #22c55e;
            --warning: #f59e0b;
            --danger: #ef4444;
            --bg: #f8fafc;
            --card-bg: #ffffff;
            --text: #1e293b;
            --text-muted: #64748b;
            --border: #e2e8f0;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
            padding: 2rem;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        .header {{
            text-align: center;
            margin-bottom: 2rem;
            padding: 2rem;
            background: linear-gradient(135deg, var(--primary), #7c3aed);
            color: white;
            border-radius: 1rem;
        }}
        .header h1 {{ font-size: 2rem; margin-bottom: 0.5rem; }}
        .header .agent-info {{
            display: flex;
            gap: 0.75rem;
            justify-content: center;
            align-items: center;
            flex-wrap: wrap;
            margin: 0.75rem 0;
        }}
        .agent-badge, .model-badge, .protocol-badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1rem;
            border-radius: 2rem;
            font-weight: 600;
            font-size: 0.9rem;
        }}
        .agent-badge {{
            background: rgba(255,255,255,0.2);
            border: 2px solid rgba(255,255,255,0.4);
        }}
        .agent-badge.agent-base_llm {{ background: rgba(34, 197, 94, 0.3); border-color: rgba(34, 197, 94, 0.6); }}
        .agent-badge.agent-eee_purple {{ background: rgba(168, 85, 247, 0.3); border-color: rgba(168, 85, 247, 0.6); }}
        .agent-badge.agent-ciris_agent {{ background: rgba(59, 130, 246, 0.3); border-color: rgba(59, 130, 246, 0.6); }}
        .model-badge {{
            background: rgba(0,0,0,0.2);
            border: 1px solid rgba(255,255,255,0.3);
        }}
        .protocol-badge {{
            background: rgba(251, 191, 36, 0.3);
            border: 1px solid rgba(251, 191, 36, 0.5);
            font-size: 0.75rem;
            padding: 0.35rem 0.75rem;
        }}
        .header .meta {{ opacity: 0.9; font-size: 0.9rem; }}
        .export-bar {{
            display: flex;
            gap: 0.5rem;
            justify-content: center;
            margin-top: 1rem;
        }}
        .export-bar button {{
            padding: 0.5rem 1rem;
            border: 2px solid rgba(255,255,255,0.3);
            background: rgba(255,255,255,0.1);
            color: white;
            border-radius: 0.5rem;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.2s;
        }}
        .export-bar button:hover {{
            background: rgba(255,255,255,0.2);
            border-color: rgba(255,255,255,0.5);
        }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 2rem; }}
        .card {{
            background: var(--card-bg);
            border-radius: 0.75rem;
            padding: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            border: 1px solid var(--border);
        }}
        .card h3 {{ font-size: 0.875rem; color: var(--text-muted); text-transform: uppercase; margin-bottom: 0.5rem; }}
        .card .value {{ font-size: 2rem; font-weight: 700; color: var(--primary); }}
        .card .value.success {{ color: var(--success); }}
        .card .value.warning {{ color: var(--warning); }}
        .card .value.danger {{ color: var(--danger); }}
        table {{ width: 100%; border-collapse: collapse; margin-bottom: 1rem; font-size: 0.875rem; }}
        th, td {{ padding: 0.75rem; text-align: left; border-bottom: 1px solid var(--border); }}
        th {{ background: var(--bg); font-weight: 600; position: sticky; top: 0; }}
        .progress-bar {{
            height: 8px;
            background: var(--border);
            border-radius: 4px;
            overflow: hidden;
        }}
        .progress-bar .fill {{
            height: 100%;
            background: var(--primary);
            transition: width 0.3s;
        }}
        .signature {{
            margin-top: 2rem;
            padding: 1rem;
            background: #1e293b;
            color: #94a3b8;
            border-radius: 0.5rem;
            font-family: monospace;
            font-size: 0.8rem;
            overflow-x: auto;
        }}
        .badge, .label-badge, .category-badge {{
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 0.25rem;
            font-size: 0.75rem;
            font-weight: 600;
        }}
        .badge-success, .label-badge.correct {{ background: #dcfce7; color: #166534; }}
        .badge-danger, .label-badge.incorrect {{ background: #fee2e2; color: #991b1b; }}
        .label-badge.expected {{ background: #e0e7ff; color: #3730a3; }}
        .label-badge.predicted.correct {{ background: #dcfce7; color: #166534; }}
        .label-badge.predicted.incorrect {{ background: #fee2e2; color: #991b1b; }}
        .category-badge {{ background: #f3e8ff; color: #7c3aed; }}
        .status-icon {{ font-size: 1.2rem; }}
        .eval-badge {{
            display: inline-block;
            padding: 0.2rem 0.4rem;
            border-radius: 0.25rem;
            font-size: 0.7rem;
            font-weight: 600;
            text-transform: uppercase;
        }}
        .eval-badge.correct {{ background: #dcfce7; color: #166534; }}
        .eval-badge.incorrect {{ background: #fee2e2; color: #991b1b; }}
        .eval-badge.unknown {{ background: #f3f4f6; color: #6b7280; }}
        .scenario-row.correct {{ background: #f0fdf4; }}
        .scenario-row.incorrect {{ background: #fef2f2; }}
        .scenario-row:hover {{ background: #f1f5f9; }}
        .input-text {{ max-width: 300px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
        .btn-small {{
            padding: 0.25rem 0.5rem;
            font-size: 0.75rem;
            background: var(--primary);
            color: white;
            border: none;
            border-radius: 0.25rem;
            cursor: pointer;
        }}
        .btn-small:hover {{ background: #4338ca; }}
        .scenarios-section {{
            margin-top: 2rem;
        }}
        .scenarios-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }}
        .filter-bar {{
            display: flex;
            gap: 0.5rem;
            align-items: center;
        }}
        .filter-bar select, .filter-bar input {{
            padding: 0.5rem;
            border: 1px solid var(--border);
            border-radius: 0.5rem;
            font-size: 0.875rem;
        }}
        .scenarios-table-wrapper {{
            max-height: 600px;
            overflow-y: auto;
            border: 1px solid var(--border);
            border-radius: 0.5rem;
        }}
        .modal {{
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.5);
            z-index: 1000;
            align-items: center;
            justify-content: center;
        }}
        .modal.active {{ display: flex; }}
        .modal-content {{
            background: white;
            border-radius: 1rem;
            padding: 2rem;
            max-width: 800px;
            max-height: 80vh;
            overflow-y: auto;
            width: 90%;
        }}
        .modal-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }}
        .modal-close {{
            background: none;
            border: none;
            font-size: 1.5rem;
            cursor: pointer;
            color: var(--text-muted);
        }}
        .detail-row {{
            display: flex;
            padding: 0.75rem 0;
            border-bottom: 1px solid var(--border);
        }}
        .detail-label {{
            width: 150px;
            font-weight: 600;
            color: var(--text-muted);
        }}
        .detail-value {{
            flex: 1;
        }}
        .response-text {{
            background: #f8fafc;
            padding: 1rem;
            border-radius: 0.5rem;
            font-family: monospace;
            font-size: 0.875rem;
            white-space: pre-wrap;
            max-height: 200px;
            overflow-y: auto;
        }}
        footer {{ text-align: center; margin-top: 2rem; color: var(--text-muted); font-size: 0.875rem; }}
        @media print {{
            .export-bar, .filter-bar, .btn-small, .modal {{ display: none !important; }}
            .scenarios-table-wrapper {{ max-height: none; overflow: visible; }}
        }}
        @media (max-width: 768px) {{
            body {{ padding: 1rem; }}
            .header {{ padding: 1.5rem; }}
            .header h1 {{ font-size: 1.5rem; }}
            .export-bar {{ flex-wrap: wrap; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>&#128300; HE-300 Ethics Benchmark Report</h1>
            <div class="agent-info">
                {f'<span class="agent-badge agent-{summary.agent_type}">{summary.agent_name or summary.agent_type.replace("_", " ").title() or "Direct LLM"}</span>' if summary.agent_type or summary.agent_name else '<span class="agent-badge">Direct LLM Evaluation</span>'}
                <span class="model-badge">&#129302; {summary.model_name}</span>
                {f'<span class="protocol-badge">{summary.protocol.upper()}</span>' if summary.protocol else ''}
            </div>
            <div class="meta">
                <strong>Batch:</strong> {summary.batch_id} |
                <strong>Generated:</strong> {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
                {f' | <strong>Agent URL:</strong> {summary.agent_url}' if summary.agent_url else ''}
            </div>
            <div class="export-bar">
                <button onclick="exportJSON()">&#128190; Export JSON</button>
                <button onclick="exportXML()">&#128196; Export XML</button>
                <button onclick="exportPDF()">&#128462; Export PDF</button>
                <button onclick="exportCSV()">&#128202; Export CSV</button>
                <button onclick="window.print()">&#128424; Print</button>
            </div>
        </div>

        <div class="grid">
            <div class="card">
                <h3>Overall Accuracy</h3>
                <div class="value {'success' if summary.overall_accuracy >= 0.8 else 'warning' if summary.overall_accuracy >= 0.5 else 'danger'}">{summary.overall_accuracy:.1%}</div>
                <div class="progress-bar"><div class="fill" style="width: {summary.overall_accuracy * 100}%"></div></div>
            </div>
            <div class="card">
                <h3>Total Scenarios</h3>
                <div class="value">{summary.total_scenarios}</div>
            </div>
            <div class="card">
                <h3>Correct Predictions</h3>
                <div class="value success">{summary.correct_predictions}</div>
            </div>
            <div class="card">
                <h3>Errors</h3>
                <div class="value {'danger' if summary.total_errors > 0 else ''}">{summary.total_errors}</div>
            </div>
            <div class="card">
                <h3>Avg Latency</h3>
                <div class="value">{summary.avg_latency_ms:.0f}ms</div>
            </div>
        </div>

        <div class="card">
            <h2 style="margin-bottom: 1rem;">Results by Category</h2>
            <table>
                <thead>
                    <tr>
                        <th>Category</th>
                        <th>Total</th>
                        <th>Correct</th>
                        <th>Accuracy</th>
                        <th>Progress</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(f"""
                    <tr>
                        <td><strong>{cat.category}</strong></td>
                        <td>{cat.total}</td>
                        <td>{cat.correct}</td>
                        <td><span class="badge {'badge-success' if cat.accuracy >= 0.7 else 'badge-danger'}">{cat.accuracy:.1%}</span></td>
                        <td><div class="progress-bar"><div class="fill" style="width: {cat.accuracy * 100}%"></div></div></td>
                    </tr>
                    """ for cat in summary.categories)}
                </tbody>
            </table>
        </div>

        <div class="card" style="margin-top: 1rem;">
            <h2 style="margin-bottom: 1rem;">Configuration</h2>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                <div>
                    <h4 style="color: var(--text-muted); font-size: 0.8rem; text-transform: uppercase; margin-bottom: 0.5rem;">Agent Information</h4>
                    <table>
                        <tr><td><strong>Agent Name</strong></td><td>{summary.agent_name or 'N/A'}</td></tr>
                        <tr><td><strong>Agent Type</strong></td><td>{summary.agent_type.replace('_', ' ').title() if summary.agent_type else 'Direct LLM'}</td></tr>
                        <tr><td><strong>Model</strong></td><td>{summary.model_name}</td></tr>
                        <tr><td><strong>Protocol</strong></td><td>{summary.protocol.upper() if summary.protocol else 'Direct API'}</td></tr>
                        {f'<tr><td><strong>Agent URL</strong></td><td><code style="font-size: 0.8rem;">{summary.agent_url}</code></td></tr>' if summary.agent_url else ''}
                    </table>
                </div>
                <div>
                    <h4 style="color: var(--text-muted); font-size: 0.8rem; text-transform: uppercase; margin-bottom: 0.5rem;">Evaluation Settings</h4>
                    <table>
                        <tr><td><strong>Identity</strong></td><td>{summary.identity_id}</td></tr>
                        <tr><td><strong>Guidance</strong></td><td>{summary.guidance_id}</td></tr>
                        <tr><td><strong>Started</strong></td><td>{summary.started_at}</td></tr>
                        <tr><td><strong>Completed</strong></td><td>{summary.completed_at}</td></tr>
                        <tr><td><strong>Processing Time</strong></td><td>{summary.processing_time_ms:.0f} ms</td></tr>
                    </table>
                </div>
            </div>
        </div>

        <!-- Individual Scenario Results -->
        <div class="card scenarios-section">
            <div class="scenarios-header">
                <h2>&#128203; Individual Scenario Results ({len(request.scenarios)} scenarios)</h2>
                <div class="filter-bar">
                    <label>Filter:</label>
                    <select id="categoryFilter" onchange="filterScenarios()">
                        <option value="">All Categories</option>
                        {''.join(f'<option value="{cat.category}">{cat.category}</option>' for cat in summary.categories)}
                    </select>
                    <select id="statusFilter" onchange="filterScenarios()">
                        <option value="">All Results</option>
                        <option value="correct">Correct Only</option>
                        <option value="incorrect">Incorrect Only</option>
                    </select>
                    <input type="text" id="searchFilter" placeholder="Search..." oninput="filterScenarios()">
                </div>
            </div>
            <div class="scenarios-table-wrapper">
                <table id="scenariosTable">
                    <thead>
                        <tr>
                            <th>ID (UID)</th>
                            <th>Category</th>
                            <th>Prompt</th>
                            <th>Ground Truth</th>
                            <th>Heuristic</th>
                            <th>Semantic</th>
                            <th>Agree</th>
                            <th>Final</th>
                            <th>Result</th>
                            <th>Latency</th>
                            <th>Trace</th>
                            <th>Details</th>
                        </tr>
                    </thead>
                    <tbody>
                        {scenarios_table}
                    </tbody>
                </table>
            </div>
        </div>

        {signature_block}

        <footer>
            Generated by EthicsEngine Enterprise HE-300 Benchmark System<br>
            <a href="https://github.com/rng-ops/ethicsengine_enterprise" style="color: var(--primary);">View on GitHub</a>
        </footer>
    </div>

    <!-- Detail Modal -->
    <div id="detailModal" class="modal">
        <div class="modal-content">
            <div class="modal-header">
                <h3>&#128269; Scenario Details</h3>
                <button class="modal-close" onclick="closeModal()">&times;</button>
            </div>
            <div id="modalBody"></div>
        </div>
    </div>

    <!-- Embedded Data for Export -->
    <script id="reportData" type="application/json">
{{
    "report_type": "he300_benchmark",
    "version": "1.0",
    "generated_at": "{datetime.now(timezone.utc).isoformat()}",
    "summary": {summary_json},
    "scenarios": {scenarios_json},
    "signature": {signature_json}
}}
    </script>

    <script>
        // Get embedded report data
        const reportData = JSON.parse(document.getElementById('reportData').textContent);
        
        // Scenario details lookup
        const scenariosMap = new Map();
        reportData.scenarios.forEach(s => scenariosMap.set(s.scenario_id, s));

        function showDetails(scenarioId) {{
            const s = scenariosMap.get(scenarioId);
            if (!s) return;

            // Label mapping: 0 = ETHICAL, 1 = UNETHICAL (per Hendrycks Ethics dataset convention)
            const expected = s.expected_label === 1 ? 'UNETHICAL' : s.expected_label === 0 ? 'ETHICAL' : 'N/A';
            const predicted = s.predicted_label === 1 ? 'UNETHICAL' : s.predicted_label === 0 ? 'ETHICAL' : 'ERROR';

            // Format evaluation details
            const formatEval = (evalData, evalName) => {{
                if (!evalData) return `<span style="color: #9ca3af;">Not available</span>`;
                const isCorrect = evalData.classification === (s.expected_label === 1 ? 'unethical' : 'ethical');
                const badgeClass = isCorrect ? 'correct' : 'incorrect';
                return `
                    <span class="eval-badge ${{badgeClass}}">${{evalData.classification.toUpperCase()}}</span>
                    <span style="color: #6b7280; font-size: 0.85rem; margin-left: 0.5rem;">
                        Confidence: ${{(evalData.confidence * 100).toFixed(0)}}%
                    </span>
                `;
            }};

            document.getElementById('modalBody').innerHTML = `
                <div class="detail-row">
                    <span class="detail-label">Scenario ID (UID)</span>
                    <span class="detail-value"><code>${{s.scenario_id}}</code></span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Question Type</span>
                    <span class="detail-value"><span class="category-badge">${{s.category}}</span></span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Prompt</span>
                    <span class="detail-value">${{s.input_text}}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Ground Truth</span>
                    <span class="detail-value"><span class="label-badge expected">${{expected}}</span></span>
                </div>

                <h4 style="margin: 1.5rem 0 0.75rem; color: var(--text-muted); font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.05em;">Dual Evaluation Results</h4>

                <div class="detail-row">
                    <span class="detail-label">Heuristic Eval</span>
                    <span class="detail-value">${{formatEval(s.heuristic_eval, 'Heuristic')}}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Semantic Eval</span>
                    <span class="detail-value">${{formatEval(s.semantic_eval, 'Semantic')}}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Evaluations Agree</span>
                    <span class="detail-value">
                        ${{s.evaluations_agree !== false
                            ? '<span style="color: #22c55e;">&#10003; Yes</span>'
                            : `<span style="color: #f59e0b;">&#9888; No - ${{s.disagreement_note || 'Methods disagree'}}</span>`
                        }}
                    </span>
                </div>

                <h4 style="margin: 1.5rem 0 0.75rem; color: var(--text-muted); font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.05em;">Final Result</h4>

                <div class="detail-row">
                    <span class="detail-label">Final Prediction</span>
                    <span class="detail-value"><span class="label-badge predicted ${{s.is_correct ? 'correct' : 'incorrect'}}">${{predicted}}</span></span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Result</span>
                    <span class="detail-value">${{s.is_correct ? '&#10003; Correct' : '&#10007; Incorrect'}}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Latency</span>
                    <span class="detail-value">${{s.latency_ms.toFixed(0)}}ms</span>
                </div>
                ${{s.trace_id ? `<div class="detail-row"><span class="detail-label">Trace ID</span><span class="detail-value"><code>${{s.trace_id}}</code>${{s.trace_url ? ` <a href="${{s.trace_url}}" target="_blank" rel="noopener noreferrer" style="color: var(--primary);">&#128279; View in LangSmith</a>` : ''}}</span></div>` : ''}}
                ${{s.error ? `<div class="detail-row"><span class="detail-label">Error</span><span class="detail-value" style="color: var(--danger)">${{s.error}}</span></div>` : ''}}
                <div style="margin-top: 1rem;">
                    <strong>Model Response:</strong>
                    <div class="response-text">${{s.model_response || 'No response recorded'}}</div>
                </div>
            `;
            document.getElementById('detailModal').classList.add('active');
        }}

        function closeModal() {{
            document.getElementById('detailModal').classList.remove('active');
        }}

        // Close modal on escape key
        document.addEventListener('keydown', (e) => {{
            if (e.key === 'Escape') closeModal();
        }});

        // Close modal on backdrop click
        document.getElementById('detailModal').addEventListener('click', (e) => {{
            if (e.target.classList.contains('modal')) closeModal();
        }});

        function filterScenarios() {{
            const category = document.getElementById('categoryFilter').value;
            const status = document.getElementById('statusFilter').value;
            const search = document.getElementById('searchFilter').value.toLowerCase();
            
            document.querySelectorAll('.scenario-row').forEach(row => {{
                const rowCategory = row.dataset.category;
                const isCorrect = row.classList.contains('correct');
                const text = row.textContent.toLowerCase();
                
                let show = true;
                if (category && rowCategory !== category) show = false;
                if (status === 'correct' && !isCorrect) show = false;
                if (status === 'incorrect' && isCorrect) show = false;
                if (search && !text.includes(search)) show = false;
                
                row.style.display = show ? '' : 'none';
            }});
        }}

        // Export to JSON
        function exportJSON() {{
            const blob = new Blob([JSON.stringify(reportData, null, 2)], {{ type: 'application/json' }});
            downloadBlob(blob, `he300-report-${{reportData.summary.batch_id}}.json`);
        }}

        // Export to XML
        function exportXML() {{
            const xml = jsonToXml(reportData);
            const blob = new Blob([xml], {{ type: 'application/xml' }});
            downloadBlob(blob, `he300-report-${{reportData.summary.batch_id}}.xml`);
        }}

        function jsonToXml(obj, rootName = 'he300_report') {{
            let xml = '<?xml version="1.0" encoding="UTF-8"?>\\n';
            xml += `<${{rootName}}>\\n`;
            xml += objectToXml(obj, 1);
            xml += `</${{rootName}}>`;
            return xml;
        }}

        function objectToXml(obj, indent = 0) {{
            let xml = '';
            const spaces = '  '.repeat(indent);
            
            for (const [key, value] of Object.entries(obj)) {{
                const tagName = key.replace(/[^a-zA-Z0-9_]/g, '_');
                
                if (value === null || value === undefined) {{
                    xml += `${{spaces}}<${{tagName}}/>\\n`;
                }} else if (Array.isArray(value)) {{
                    xml += `${{spaces}}<${{tagName}}>\\n`;
                    value.forEach((item, i) => {{
                        xml += `${{spaces}}  <item index="${{i}}">\\n`;
                        if (typeof item === 'object') {{
                            xml += objectToXml(item, indent + 2);
                        }} else {{
                            xml += `${{spaces}}    ${{escapeXml(String(item))}}\\n`;
                        }}
                        xml += `${{spaces}}  </item>\\n`;
                    }});
                    xml += `${{spaces}}</${{tagName}}>\\n`;
                }} else if (typeof value === 'object') {{
                    xml += `${{spaces}}<${{tagName}}>\\n`;
                    xml += objectToXml(value, indent + 1);
                    xml += `${{spaces}}</${{tagName}}>\\n`;
                }} else {{
                    xml += `${{spaces}}<${{tagName}}>${{escapeXml(String(value))}}</${{tagName}}>\\n`;
                }}
            }}
            return xml;
        }}

        function escapeXml(str) {{
            return str.replace(/&/g, '&amp;')
                      .replace(/</g, '&lt;')
                      .replace(/>/g, '&gt;')
                      .replace(/"/g, '&quot;')
                      .replace(/'/g, '&apos;');
        }}

        // Export to CSV
        function exportCSV() {{
            const headers = ['scenario_id', 'category', 'input_text', 'expected_label', 'predicted_label', 'is_correct', 'latency_ms', 'model_response', 'error'];
            let csv = headers.join(',') + '\\n';
            
            reportData.scenarios.forEach(s => {{
                const row = headers.map(h => {{
                    let val = s[h];
                    if (val === null || val === undefined) val = '';
                    val = String(val).replace(/"/g, '""');
                    if (val.includes(',') || val.includes('"') || val.includes('\\n')) {{
                        val = `"${{val}}"`;
                    }}
                    return val;
                }});
                csv += row.join(',') + '\\n';
            }});
            
            const blob = new Blob([csv], {{ type: 'text/csv' }});
            downloadBlob(blob, `he300-report-${{reportData.summary.batch_id}}.csv`);
        }}

        // Export to PDF (using browser print)
        function exportPDF() {{
            // Create a clean print-optimized version
            const printWindow = window.open('', '_blank');
            printWindow.document.write(`
                <!DOCTYPE html>
                <html>
                <head>
                    <title>HE-300 Benchmark Report - ${{reportData.summary.batch_id}}</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; padding: 20px; font-size: 10pt; }}
                        h1 {{ color: #4f46e5; font-size: 18pt; }}
                        h2 {{ color: #1e293b; font-size: 14pt; margin-top: 20px; }}
                        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; font-size: 9pt; }}
                        th, td {{ border: 1px solid #e2e8f0; padding: 6px; text-align: left; }}
                        th {{ background: #f8fafc; font-weight: bold; }}
                        .correct {{ background: #dcfce7; }}
                        .incorrect {{ background: #fee2e2; }}
                        .stats {{ display: flex; gap: 20px; margin: 15px 0; }}
                        .stat {{ text-align: center; }}
                        .stat-value {{ font-size: 24pt; font-weight: bold; color: #4f46e5; }}
                        .stat-label {{ font-size: 9pt; color: #64748b; }}
                        @media print {{ body {{ print-color-adjust: exact; -webkit-print-color-adjust: exact; }} }}
                    </style>
                </head>
                <body>
                    <h1>HE-300 Ethics Benchmark Report</h1>
                    <p><strong>Batch:</strong> ${{reportData.summary.batch_id}} | <strong>Model:</strong> ${{reportData.summary.model_name}} | <strong>Date:</strong> ${{new Date(reportData.generated_at).toLocaleString()}}</p>
                    
                    <div class="stats">
                        <div class="stat"><div class="stat-value">${{(reportData.summary.overall_accuracy * 100).toFixed(1)}}%</div><div class="stat-label">Accuracy</div></div>
                        <div class="stat"><div class="stat-value">${{reportData.summary.total_scenarios}}</div><div class="stat-label">Total</div></div>
                        <div class="stat"><div class="stat-value">${{reportData.summary.correct_predictions}}</div><div class="stat-label">Correct</div></div>
                        <div class="stat"><div class="stat-value">${{reportData.summary.avg_latency_ms.toFixed(0)}}ms</div><div class="stat-label">Avg Latency</div></div>
                    </div>
                    
                    <h2>Results by Category</h2>
                    <table>
                        <tr><th>Category</th><th>Total</th><th>Correct</th><th>Accuracy</th></tr>
                        ${{reportData.summary.categories.map(c => `<tr><td>${{c.category}}</td><td>${{c.total}}</td><td>${{c.correct}}</td><td>${{(c.accuracy * 100).toFixed(1)}}%</td></tr>`).join('')}}
                    </table>
                    
                    <h2>Scenario Results</h2>
                    <table>
                        <tr><th>ID</th><th>Category</th><th>Input</th><th>Expected</th><th>Predicted</th><th>Correct</th></tr>
                        ${{reportData.scenarios.map(s => `
                            <tr class="${{s.is_correct ? 'correct' : 'incorrect'}}">
                                <td>${{s.scenario_id}}</td>
                                <td>${{s.category}}</td>
                                <td>${{s.input_text.substring(0, 60)}}${{s.input_text.length > 60 ? '...' : ''}}</td>
                                <td>${{s.expected_label === 1 ? 'ETHICAL' : 'UNETHICAL'}}</td>
                                <td>${{s.predicted_label === 1 ? 'ETHICAL' : s.predicted_label === 0 ? 'UNETHICAL' : 'ERROR'}}</td>
                                <td>${{s.is_correct ? 'Yes' : 'No'}}</td>
                            </tr>
                        `).join('')}}
                    </table>
                    
                    ${{reportData.signature ? `
                        <h2>Report Signature</h2>
                        <p style="font-family: monospace; font-size: 8pt; background: #f8fafc; padding: 10px; border-radius: 5px;">
                            Algorithm: ${{reportData.signature.algorithm}}<br>
                            Hash: ${{reportData.signature.content_hash}}<br>
                            Signature: ${{reportData.signature.signature}}
                        </p>
                    ` : ''}}
                </body>
                </html>
            `);
            printWindow.document.close();
            setTimeout(() => {{ printWindow.print(); }}, 500);
        }}

        function downloadBlob(blob, filename) {{
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }}
    </script>
</body>
</html>'''
    
    return html


def generate_json_report(request: ReportRequest, signature: Optional[ReportSignature]) -> str:
    """Generate a JSON report."""
    report = {
        "report_type": "he300_benchmark",
        "version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "batch_id": request.batch_id,
        "summary": request.summary.model_dump(),
        "format": request.format.value,
    }
    
    if request.include_scenarios and request.scenarios:
        report["scenarios"] = [s.model_dump() for s in request.scenarios]
    
    if signature:
        report["signature"] = signature.model_dump()
    
    return json.dumps(report, indent=2, default=str)


# --- API Endpoints ---

@router.get("/results", response_model=BenchmarkResultsResponse)
async def get_benchmark_results():
    """
    Get available benchmark results for report generation.
    
    This endpoint returns all available HE-300 benchmark results that can be
    used to generate reports. Results are loaded from the benchmark_results
    directory and from stored traces.
    """
    results: List[BenchmarkResultItem] = []
    
    # Load from benchmark results directory
    for result_file in BENCHMARK_RESULTS_DIR.glob("*.json"):
        try:
            data = json.loads(result_file.read_text())
            
            # Extract batch_id from filename or data
            batch_id = data.get("batch_id", result_file.stem)
            
            # Extract scores from summary
            scores = {}
            if "summary" in data:
                summary = data["summary"]
                scores["overall"] = summary.get("accuracy", 0)
                if "by_category" in summary:
                    for cat, cat_data in summary["by_category"].items():
                        if isinstance(cat_data, dict):
                            scores[cat] = cat_data.get("accuracy", 0)
                        else:
                            scores[cat] = 0
            
            results.append(BenchmarkResultItem(
                id=batch_id,
                model_name=data.get("model_name", "Unknown"),
                report_name="HE-300 Benchmark",
                created_at=data.get("completed_at", data.get("created_at", datetime.now(timezone.utc).isoformat())),
                scores=scores,
                status=data.get("status", "completed"),
            ))
        except Exception as e:
            logger.warning(f"Failed to load benchmark result from {result_file}: {e}")
    
    # Also load from generated reports metadata
    for meta_file in REPORTS_DIR.glob("*.meta.json"):
        try:
            meta_data = json.loads(meta_file.read_text())
            batch_id = meta_data.get("batch_id", "")
            
            # Skip if we already have this batch_id
            if any(r.id == batch_id for r in results):
                continue
            
            results.append(BenchmarkResultItem(
                id=batch_id,
                model_name=meta_data.get("model_name", "Unknown"),
                report_name="HE-300 Benchmark",
                created_at=meta_data.get("created_at", datetime.now(timezone.utc).isoformat()),
                scores={"overall": meta_data.get("accuracy", 0)},
                status="completed",
            ))
        except Exception as e:
            logger.warning(f"Failed to load metadata from {meta_file}: {e}")
    
    # Sort by created_at descending
    results.sort(key=lambda r: r.created_at, reverse=True)
    
    return BenchmarkResultsResponse(results=results, total=len(results))


@router.post("/generate", response_model=ReportResponse)
async def generate_report(request: ReportRequest):
    """
    Generate a signed static report from benchmark results.
    
    Supports multiple output formats:
    - **markdown**: Jekyll-compatible Markdown with YAML frontmatter
    - **html**: Self-contained HTML page for static hosting
    - **json**: Machine-readable JSON format
    
    Reports can be cryptographically signed for integrity verification.
    """
    report_id = str(uuid.uuid4())[:8]
    
    # Generate report content (without signature first)
    if request.format == ReportFormat.MARKDOWN:
        content = generate_markdown_report(request, None)
        ext = "md"
    elif request.format == ReportFormat.HTML:
        content = generate_html_report(request, None)
        ext = "html"
    else:
        content = generate_json_report(request, None)
        ext = "json"
    
    # Sign if requested
    signature = None
    if request.sign_report:
        signature = sign_content(content)
        
        # Regenerate with signature included
        if request.format == ReportFormat.MARKDOWN:
            content = generate_markdown_report(request, signature)
        elif request.format == ReportFormat.HTML:
            content = generate_html_report(request, signature)
        else:
            content = generate_json_report(request, signature)
    
    # Save to disk
    filename = f"report_{request.batch_id}_{report_id}.{ext}"
    file_path = REPORTS_DIR / filename
    file_path.write_text(content, encoding="utf-8")
    
    # Create metadata
    metadata = ReportMetadata(
        report_id=report_id,
        batch_id=request.batch_id,
        model_name=request.summary.model_name,
        accuracy=request.summary.overall_accuracy,
        format=request.format,
        created_at=datetime.now(timezone.utc).isoformat(),
        file_path=str(file_path),
        file_size=file_path.stat().st_size,
        signature=signature,
    )
    
    # Also save metadata
    meta_path = REPORTS_DIR / f"report_{request.batch_id}_{report_id}.meta.json"
    meta_path.write_text(metadata.model_dump_json(indent=2), encoding="utf-8")
    
    logger.info(f"Generated report {report_id} for batch {request.batch_id} in {request.format} format")
    
    return ReportResponse(
        status="generated",
        report_id=report_id,
        metadata=metadata,
        download_url=f"/reports/download/{report_id}",
        preview=content[:500] if len(content) > 500 else content,
    )


@router.get("/", response_model=ReportListResponse)
async def list_reports():
    """List all generated reports."""
    reports = []
    
    for meta_file in REPORTS_DIR.glob("*.meta.json"):
        try:
            meta_data = json.loads(meta_file.read_text())
            # Reconstruct signature if present
            sig_data = meta_data.pop("signature", None)
            if sig_data:
                meta_data["signature"] = ReportSignature(**sig_data)
            reports.append(ReportMetadata(**meta_data))
        except Exception as e:
            logger.warning(f"Failed to load report metadata from {meta_file}: {e}")
    
    # Sort by creation date (newest first)
    reports.sort(key=lambda r: r.created_at, reverse=True)
    
    return ReportListResponse(reports=reports, total=len(reports))


@router.get("/download/{report_id}")
async def download_report(report_id: str):
    """Download a generated report."""
    # Find the report file
    for report_file in REPORTS_DIR.glob(f"report_*_{report_id}.*"):
        if not report_file.name.endswith(".meta.json"):
            # Determine content type
            if report_file.suffix == ".md":
                media_type = "text/markdown"
            elif report_file.suffix == ".html":
                media_type = "text/html"
            else:
                media_type = "application/json"
            
            return FileResponse(
                path=report_file,
                media_type=media_type,
                filename=report_file.name,
            )
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Report {report_id} not found"
    )


@router.get("/{report_id}", response_model=ReportMetadata)
async def get_report_metadata(report_id: str):
    """Get metadata for a specific report."""
    for meta_file in REPORTS_DIR.glob(f"report_*_{report_id}.meta.json"):
        try:
            meta_data = json.loads(meta_file.read_text())
            sig_data = meta_data.pop("signature", None)
            if sig_data:
                meta_data["signature"] = ReportSignature(**sig_data)
            return ReportMetadata(**meta_data)
        except Exception as e:
            logger.warning(f"Failed to load report metadata: {e}")
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Report {report_id} not found"
    )


@router.delete("/{report_id}")
async def delete_report(report_id: str):
    """Delete a generated report."""
    deleted = False
    
    for file in REPORTS_DIR.glob(f"report_*_{report_id}.*"):
        file.unlink()
        deleted = True
    
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Report {report_id} not found"
        )
    
    return {"status": "deleted", "report_id": report_id}


@router.post("/verify")
async def verify_report_signature(content: str, signature: ReportSignature):
    """Verify the integrity of a report using its signature."""
    is_valid = verify_signature(content, signature)
    
    return {
        "is_valid": is_valid,
        "content_hash": hashlib.sha256(content.encode()).hexdigest(),
        "expected_hash": signature.content_hash,
        "message": "Signature is valid" if is_valid else "Signature verification failed"
    }
