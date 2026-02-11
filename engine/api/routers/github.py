# -*- coding: utf-8 -*-
"""
GitHub Pages Deployment API Router

Enables publishing benchmark reports to GitHub Pages for public viewing.
Supports repo management, file deployment, and automatic site generation.
"""

import logging
import json
import os
import base64
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any
from enum import Enum

from fastapi import APIRouter, HTTPException, status, Body
from pydantic import BaseModel, Field
import httpx

# Add project root to path for imports
import sys
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/github",
    tags=["github", "deployment"],
    responses={
        401: {"description": "GitHub authentication required"},
        403: {"description": "Insufficient permissions"},
        404: {"description": "Repository not found"},
        500: {"description": "GitHub API error"},
    },
)

# --- Constants ---
GITHUB_API_BASE = "https://api.github.com"
REPORTS_DIR = Path(project_root) / "data" / "reports"
GITHUB_CONFIG_FILE = Path(project_root) / "data" / "github_config.json"


# --- Schemas ---
class GitHubRepo(BaseModel):
    """GitHub repository information."""
    full_name: str
    name: str
    owner: str
    description: Optional[str] = None
    private: bool = False
    default_branch: str = "main"
    html_url: str
    has_pages: bool = False
    pages_url: Optional[str] = None
    permissions: Dict[str, bool] = {}


class GitHubConfig(BaseModel):
    """GitHub configuration for deployment."""
    token: str = Field(..., description="GitHub Personal Access Token")
    repo_full_name: str = Field(..., description="Repository in owner/repo format")
    target_branch: str = Field(default="gh-pages", description="Branch for GitHub Pages")
    target_path: str = Field(default="reports", description="Directory path in repo for reports")
    auto_enable_pages: bool = Field(default=True, description="Automatically enable GitHub Pages if not configured")


class DeploymentRequest(BaseModel):
    """Request to deploy reports to GitHub Pages."""
    report_ids: List[str] = Field(..., description="List of report IDs to deploy")
    commit_message: Optional[str] = None
    generate_index: bool = Field(default=True, description="Generate index page for reports")


class DeploymentResult(BaseModel):
    """Result of a deployment operation."""
    status: str
    deployed_reports: List[str]
    failed_reports: List[Dict[str, str]] = []
    commit_sha: Optional[str] = None
    pages_url: Optional[str] = None
    index_url: Optional[str] = None


class PublishedReport(BaseModel):
    """A report published to GitHub Pages."""
    report_id: str
    batch_id: str
    model_name: str
    format: str
    accuracy: float
    published_at: str
    pages_url: str
    raw_url: str
    file_name: str


class PublishedReportsIndex(BaseModel):
    """Index of all published reports."""
    repo: str
    branch: str
    total_reports: int
    reports: List[PublishedReport]
    last_updated: str


# --- Helper Functions ---
def load_github_config() -> Optional[GitHubConfig]:
    """Load GitHub configuration from file."""
    if GITHUB_CONFIG_FILE.exists():
        try:
            data = json.loads(GITHUB_CONFIG_FILE.read_text())
            return GitHubConfig(**data)
        except Exception as e:
            logger.warning(f"Failed to load GitHub config: {e}")
    return None


def save_github_config(config: GitHubConfig):
    """Save GitHub configuration to file."""
    GITHUB_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    # Don't store full token for security - store a masked version
    data = config.model_dump()
    GITHUB_CONFIG_FILE.write_text(json.dumps(data, indent=2))


async def github_request(
    method: str,
    endpoint: str,
    token: str,
    data: Optional[Dict] = None,
    accept: str = "application/vnd.github.v3+json"
) -> httpx.Response:
    """Make a request to GitHub API."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": accept,
        "X-GitHub-Api-Version": "2022-11-28",
    }

    url = f"{GITHUB_API_BASE}{endpoint}"

    async with httpx.AsyncClient(timeout=30.0) as client:
        if method.upper() == "GET":
            response = await client.get(url, headers=headers)
        elif method.upper() == "POST":
            response = await client.post(url, headers=headers, json=data)
        elif method.upper() == "PUT":
            response = await client.put(url, headers=headers, json=data)
        elif method.upper() == "DELETE":
            response = await client.delete(url, headers=headers)
        elif method.upper() == "PATCH":
            response = await client.patch(url, headers=headers, json=data)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")

    return response


def generate_index_html(reports: List[PublishedReport], repo: str) -> str:
    """Generate a HuggingFace-style index page for published reports with model cards."""
    reports_json = json.dumps([r.model_dump() for r in reports], indent=2)

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HE-300 Ethics Benchmark Reports</title>
    <meta name="description" content="AI Ethics Benchmark Leaderboard - Model evaluation results for the HE-300 benchmark suite">
    <style>
        :root {{
            --hf-yellow: #ffd21e;
            --hf-orange: #ff9d00;
            --primary: #6366f1;
            --primary-dark: #4f46e5;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --bg: #0d1117;
            --bg-secondary: #161b22;
            --bg-card: #21262d;
            --text: #f0f6fc;
            --text-secondary: #8b949e;
            --border: #30363d;
            --accent: #58a6ff;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.5;
        }}
        .navbar {{
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border);
            padding: 0.75rem 1.5rem;
            display: flex;
            align-items: center;
            gap: 1.5rem;
            position: sticky;
            top: 0;
            z-index: 100;
        }}
        .navbar-brand {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-weight: 600;
            font-size: 1.1rem;
            color: var(--text);
            text-decoration: none;
        }}
        .navbar-brand .logo {{
            font-size: 1.5rem;
        }}
        .nav-links {{
            display: flex;
            gap: 1rem;
            margin-left: auto;
        }}
        .nav-link {{
            color: var(--text-secondary);
            text-decoration: none;
            padding: 0.5rem 0.75rem;
            border-radius: 0.375rem;
            font-size: 0.875rem;
            transition: all 0.2s;
        }}
        .nav-link:hover, .nav-link.active {{
            color: var(--text);
            background: var(--bg-card);
        }}
        .hero {{
            background: linear-gradient(135deg, var(--primary) 0%, #8b5cf6 50%, #ec4899 100%);
            padding: 3rem 2rem;
            text-align: center;
        }}
        .hero h1 {{
            font-size: 2.25rem;
            margin-bottom: 0.75rem;
            font-weight: 700;
        }}
        .hero p {{
            font-size: 1rem;
            opacity: 0.9;
            max-width: 600px;
            margin: 0 auto;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 1.5rem;
        }}
        .tabs {{
            display: flex;
            gap: 0.25rem;
            background: var(--bg-secondary);
            padding: 0.25rem;
            border-radius: 0.5rem;
            margin-bottom: 1.5rem;
            width: fit-content;
        }}
        .tab {{
            padding: 0.5rem 1rem;
            border: none;
            background: transparent;
            color: var(--text-secondary);
            border-radius: 0.375rem;
            cursor: pointer;
            font-size: 0.875rem;
            font-weight: 500;
            transition: all 0.2s;
        }}
        .tab:hover {{
            color: var(--text);
        }}
        .tab.active {{
            background: var(--bg-card);
            color: var(--text);
        }}
        .stats-row {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 1rem;
            margin-bottom: 1.5rem;
        }}
        .stat-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 0.75rem;
            padding: 1rem;
            text-align: center;
        }}
        .stat-value {{
            font-size: 1.75rem;
            font-weight: 700;
            color: var(--primary);
        }}
        .stat-label {{
            font-size: 0.75rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 0.25rem;
        }}
        .search-filters {{
            display: flex;
            gap: 0.75rem;
            margin-bottom: 1.5rem;
            flex-wrap: wrap;
        }}
        .search-input {{
            flex: 1;
            min-width: 250px;
            background: var(--bg-card);
            border: 1px solid var(--border);
            color: var(--text);
            padding: 0.625rem 1rem;
            border-radius: 0.5rem;
            font-size: 0.875rem;
        }}
        .search-input::placeholder {{
            color: var(--text-secondary);
        }}
        .filter-select {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            color: var(--text);
            padding: 0.625rem 1rem;
            border-radius: 0.5rem;
            font-size: 0.875rem;
            cursor: pointer;
        }}
        .view-toggle {{
            display: flex;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 0.5rem;
            overflow: hidden;
        }}
        .view-btn {{
            padding: 0.625rem 0.875rem;
            border: none;
            background: transparent;
            color: var(--text-secondary);
            cursor: pointer;
            font-size: 1rem;
        }}
        .view-btn.active {{
            background: var(--primary);
            color: white;
        }}

        /* Model Cards Grid */
        .model-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
            gap: 1rem;
        }}
        .model-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 0.75rem;
            overflow: hidden;
            transition: all 0.2s;
        }}
        .model-card:hover {{
            border-color: var(--primary);
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        }}
        .model-card-header {{
            padding: 1rem;
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: flex-start;
            gap: 0.75rem;
        }}
        .model-avatar {{
            width: 48px;
            height: 48px;
            background: linear-gradient(135deg, var(--primary), #8b5cf6);
            border-radius: 0.5rem;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.25rem;
            flex-shrink: 0;
        }}
        .model-info {{
            flex: 1;
            min-width: 0;
        }}
        .model-name {{
            font-weight: 600;
            font-size: 0.95rem;
            color: var(--text);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        .model-provider {{
            font-size: 0.8rem;
            color: var(--text-secondary);
            display: flex;
            align-items: center;
            gap: 0.25rem;
        }}
        .model-card-body {{
            padding: 1rem;
        }}
        .accuracy-bar {{
            height: 8px;
            background: var(--border);
            border-radius: 4px;
            overflow: hidden;
            margin-bottom: 0.5rem;
        }}
        .accuracy-fill {{
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s;
        }}
        .accuracy-fill.high {{ background: var(--success); }}
        .accuracy-fill.medium {{ background: var(--warning); }}
        .accuracy-fill.low {{ background: var(--danger); }}
        .accuracy-text {{
            display: flex;
            justify-content: space-between;
            font-size: 0.8rem;
            color: var(--text-secondary);
        }}
        .accuracy-score {{
            font-weight: 600;
            font-size: 1.5rem;
        }}
        .accuracy-score.high {{ color: var(--success); }}
        .accuracy-score.medium {{ color: var(--warning); }}
        .accuracy-score.low {{ color: var(--danger); }}
        .model-tags {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.375rem;
            margin-top: 0.75rem;
        }}
        .tag {{
            display: inline-flex;
            align-items: center;
            gap: 0.25rem;
            padding: 0.25rem 0.5rem;
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 1rem;
            font-size: 0.7rem;
            color: var(--text-secondary);
        }}
        .model-card-footer {{
            padding: 0.75rem 1rem;
            border-top: 1px solid var(--border);
            display: flex;
            gap: 0.5rem;
        }}
        .btn {{
            flex: 1;
            padding: 0.5rem 0.75rem;
            border: none;
            border-radius: 0.375rem;
            font-weight: 500;
            font-size: 0.8rem;
            cursor: pointer;
            text-decoration: none;
            text-align: center;
            transition: all 0.2s;
        }}
        .btn-primary {{
            background: var(--primary);
            color: white;
        }}
        .btn-primary:hover {{
            background: var(--primary-dark);
        }}
        .btn-secondary {{
            background: var(--bg-secondary);
            color: var(--text);
            border: 1px solid var(--border);
        }}
        .btn-secondary:hover {{
            background: var(--border);
        }}

        /* Leaderboard Table */
        .leaderboard {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 0.75rem;
            overflow: hidden;
        }}
        .leaderboard table {{
            width: 100%;
            border-collapse: collapse;
        }}
        .leaderboard th {{
            background: var(--bg-secondary);
            padding: 0.875rem 1rem;
            text-align: left;
            font-size: 0.75rem;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            border-bottom: 1px solid var(--border);
            cursor: pointer;
        }}
        .leaderboard th:hover {{
            color: var(--text);
        }}
        .leaderboard td {{
            padding: 0.875rem 1rem;
            border-bottom: 1px solid var(--border);
            font-size: 0.875rem;
        }}
        .leaderboard tr:hover {{
            background: var(--bg-secondary);
        }}
        .leaderboard tr:last-child td {{
            border-bottom: none;
        }}
        .rank {{
            font-weight: 700;
            color: var(--hf-yellow);
        }}
        .rank-1 {{ color: #ffd700; }}
        .rank-2 {{ color: #c0c0c0; }}
        .rank-3 {{ color: #cd7f32; }}
        .model-cell {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }}
        .mini-avatar {{
            width: 32px;
            height: 32px;
            background: linear-gradient(135deg, var(--primary), #8b5cf6);
            border-radius: 0.375rem;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.875rem;
        }}

        /* Empty State */
        .empty-state {{
            text-align: center;
            padding: 4rem 2rem;
            color: var(--text-secondary);
        }}
        .empty-icon {{
            font-size: 4rem;
            margin-bottom: 1rem;
            opacity: 0.5;
        }}

        footer {{
            text-align: center;
            padding: 2rem;
            color: var(--text-secondary);
            font-size: 0.8rem;
            border-top: 1px solid var(--border);
            margin-top: 2rem;
        }}
        footer a {{
            color: var(--accent);
            text-decoration: none;
        }}
        footer a:hover {{
            text-decoration: underline;
        }}

        @media (max-width: 768px) {{
            .navbar {{ padding: 0.5rem 1rem; }}
            .hero {{ padding: 2rem 1rem; }}
            .hero h1 {{ font-size: 1.5rem; }}
            .container {{ padding: 1rem; }}
            .model-cards {{ grid-template-columns: 1fr; }}
            .search-filters {{ flex-direction: column; }}
            .search-input {{ min-width: 100%; }}
        }}
    </style>
</head>
<body>
    <nav class="navbar">
        <a href="#" class="navbar-brand">
            <span class="logo">&#128300;</span>
            <span>HE-300 Benchmark</span>
        </a>
        <div class="nav-links">
            <a href="#leaderboard" class="nav-link active">Leaderboard</a>
            <a href="#models" class="nav-link">Models</a>
            <a href="https://github.com/{repo}" class="nav-link" target="_blank">&#128279; Repository</a>
        </div>
    </nav>

    <div class="hero">
        <h1>&#127942; Ethics Benchmark Leaderboard</h1>
        <p>Evaluating AI models on the HE-300 ethics benchmark suite. Compare model performance across moral reasoning tasks.</p>
    </div>

    <div class="container">
        <div class="stats-row" id="statsRow"></div>

        <div class="search-filters">
            <input type="text" class="search-input" id="searchInput" placeholder="&#128269; Search models..." oninput="filterAndRender()">
            <select class="filter-select" id="sortSelect" onchange="filterAndRender()">
                <option value="accuracy-desc">Highest Accuracy</option>
                <option value="accuracy-asc">Lowest Accuracy</option>
                <option value="date-desc">Newest First</option>
                <option value="date-asc">Oldest First</option>
                <option value="model">Model Name</option>
            </select>
            <div class="view-toggle">
                <button class="view-btn active" onclick="setView('cards')" id="viewCards">&#9638;</button>
                <button class="view-btn" onclick="setView('table')" id="viewTable">&#9776;</button>
            </div>
        </div>

        <div id="cardsView" class="model-cards"></div>
        <div id="tableView" class="leaderboard" style="display: none;"></div>
        <div id="emptyState" class="empty-state" style="display: none;">
            <div class="empty-icon">&#128196;</div>
            <h3>No reports found</h3>
            <p>Try adjusting your search criteria</p>
        </div>
    </div>

    <footer>
        <p>Powered by <a href="https://github.com/{repo}">EthicsEngine Enterprise</a> &#8212; HE-300 Benchmark System</p>
        <p>Last updated: <span id="lastUpdated"></span></p>
    </footer>

    <script>
        const reports = {reports_json};
        let filteredReports = [...reports];
        let currentView = 'cards';

        function init() {{
            renderStats();
            filterAndRender();
            document.getElementById('lastUpdated').textContent = new Date().toLocaleString();
        }}

        function renderStats() {{
            const total = reports.length;
            const avgAcc = reports.length > 0 ? reports.reduce((s, r) => s + r.accuracy, 0) / reports.length : 0;
            const models = new Set(reports.map(r => r.model_name)).size;
            const latest = reports.length > 0 ? new Date(Math.max(...reports.map(r => new Date(r.published_at)))).toLocaleDateString() : 'N/A';
            const topAcc = reports.length > 0 ? Math.max(...reports.map(r => r.accuracy)) : 0;

            document.getElementById('statsRow').innerHTML = `
                <div class="stat-card"><div class="stat-value">${{total}}</div><div class="stat-label">Reports</div></div>
                <div class="stat-card"><div class="stat-value">${{models}}</div><div class="stat-label">Models</div></div>
                <div class="stat-card"><div class="stat-value">${{(avgAcc * 100).toFixed(1)}}%</div><div class="stat-label">Avg Score</div></div>
                <div class="stat-card"><div class="stat-value">${{(topAcc * 100).toFixed(1)}}%</div><div class="stat-label">Top Score</div></div>
            `;
        }}

        function filterAndRender() {{
            const search = document.getElementById('searchInput').value.toLowerCase();
            const sort = document.getElementById('sortSelect').value;

            filteredReports = reports.filter(r =>
                r.model_name.toLowerCase().includes(search) ||
                r.batch_id.toLowerCase().includes(search)
            );

            switch(sort) {{
                case 'accuracy-desc': filteredReports.sort((a, b) => b.accuracy - a.accuracy); break;
                case 'accuracy-asc': filteredReports.sort((a, b) => a.accuracy - b.accuracy); break;
                case 'date-desc': filteredReports.sort((a, b) => new Date(b.published_at) - new Date(a.published_at)); break;
                case 'date-asc': filteredReports.sort((a, b) => new Date(a.published_at) - new Date(b.published_at)); break;
                case 'model': filteredReports.sort((a, b) => a.model_name.localeCompare(b.model_name)); break;
            }}

            render();
        }}

        function render() {{
            const empty = document.getElementById('emptyState');
            const cards = document.getElementById('cardsView');
            const table = document.getElementById('tableView');

            if (filteredReports.length === 0) {{
                empty.style.display = 'block';
                cards.style.display = 'none';
                table.style.display = 'none';
                return;
            }}

            empty.style.display = 'none';

            if (currentView === 'cards') {{
                cards.style.display = 'grid';
                table.style.display = 'none';
                renderCards();
            }} else {{
                cards.style.display = 'none';
                table.style.display = 'block';
                renderTable();
            }}
        }}

        function getAccClass(acc) {{
            return acc >= 0.7 ? 'high' : acc >= 0.5 ? 'medium' : 'low';
        }}

        function getModelIcon(name) {{
            const n = name.toLowerCase();
            if (n.includes('gpt')) return '&#129302;';
            if (n.includes('claude')) return '&#128172;';
            if (n.includes('llama')) return '&#129433;';
            if (n.includes('mistral')) return '&#127786;';
            if (n.includes('gemma')) return '&#128142;';
            if (n.includes('phi')) return '&#966;';
            if (n.includes('qwen')) return '&#127968;';
            return '&#129302;';
        }}

        function getProvider(name) {{
            const n = name.toLowerCase();
            if (n.includes('gpt')) return 'OpenAI';
            if (n.includes('claude')) return 'Anthropic';
            if (n.includes('llama')) return 'Meta';
            if (n.includes('mistral')) return 'Mistral AI';
            if (n.includes('gemma')) return 'Google';
            if (n.includes('phi')) return 'Microsoft';
            if (n.includes('qwen')) return 'Alibaba';
            return 'Unknown';
        }}

        function renderCards() {{
            document.getElementById('cardsView').innerHTML = filteredReports.map((r, i) => {{
                const acc = r.accuracy;
                const accPct = (acc * 100).toFixed(1);
                const accClass = getAccClass(acc);
                const date = new Date(r.published_at).toLocaleDateString();
                const icon = getModelIcon(r.model_name);
                const provider = getProvider(r.model_name);

                return `
                    <div class="model-card">
                        <div class="model-card-header">
                            <div class="model-avatar">${{icon}}</div>
                            <div class="model-info">
                                <div class="model-name">${{r.model_name}}</div>
                                <div class="model-provider">&#127970; ${{provider}}</div>
                            </div>
                        </div>
                        <div class="model-card-body">
                            <div class="accuracy-bar">
                                <div class="accuracy-fill ${{accClass}}" style="width: ${{accPct}}%"></div>
                            </div>
                            <div class="accuracy-text">
                                <span>Ethics Score</span>
                                <span class="accuracy-score ${{accClass}}">${{accPct}}%</span>
                            </div>
                            <div class="model-tags">
                                <span class="tag">&#128197; ${{date}}</span>
                                <span class="tag">&#128196; ${{r.format.toUpperCase()}}</span>
                                <span class="tag">&#128200; Rank #${{i + 1}}</span>
                            </div>
                        </div>
                        <div class="model-card-footer">
                            <a href="${{r.pages_url}}" class="btn btn-primary" target="_blank">View Report</a>
                            <a href="${{r.raw_url}}" class="btn btn-secondary" download>Download</a>
                        </div>
                    </div>
                `;
            }}).join('');
        }}

        function renderTable() {{
            const rows = filteredReports.map((r, i) => {{
                const rank = i + 1;
                const rankClass = rank <= 3 ? 'rank-' + rank : '';
                const acc = (r.accuracy * 100).toFixed(1);
                const accClass = getAccClass(r.accuracy);
                const date = new Date(r.published_at).toLocaleDateString();
                const icon = getModelIcon(r.model_name);
                const provider = getProvider(r.model_name);

                return `<tr>
                    <td><span class="rank ${{rankClass}}">#${{rank}}</span></td>
                    <td><div class="model-cell"><div class="mini-avatar">${{icon}}</div><div><strong>${{r.model_name}}</strong><br><small style="color:var(--text-secondary)">${{provider}}</small></div></div></td>
                    <td><span class="accuracy-score ${{accClass}}">${{acc}}%</span></td>
                    <td>${{date}}</td>
                    <td><a href="${{r.pages_url}}" class="btn btn-primary" target="_blank" style="padding: 0.25rem 0.5rem; font-size: 0.75rem;">View</a></td>
                </tr>`;
            }}).join('');

            document.getElementById('tableView').innerHTML = `
                <table>
                    <thead>
                        <tr>
                            <th>Rank</th>
                            <th>Model</th>
                            <th>Score</th>
                            <th>Date</th>
                            <th>Report</th>
                        </tr>
                    </thead>
                    <tbody>${{rows}}</tbody>
                </table>
            `;
        }}

        function setView(view) {{
            currentView = view;
            document.getElementById('viewCards').classList.toggle('active', view === 'cards');
            document.getElementById('viewTable').classList.toggle('active', view === 'table');
            render();
        }}

        init();
    </script>
</body>
</html>'''


def generate_root_index_html(repo: str, reports_path: str) -> str:
    """Generate a HuggingFace-style root index with model hub, popular models, and leaderboard.

    Features:
    - Model cards with ethics scores
    - Leaderboard table view
    - Baked-in list of popular models from HuggingFace/Ollama
    - Reports grouped by model
    - Search and filtering
    """
    owner, repo_name = repo.split("/")

    # Baked-in popular models list (from HuggingFace and Ollama)
    popular_models = [
        {"name": "gpt-4o", "provider": "OpenAI", "category": "Proprietary", "icon": "&#129302;", "params": "1.7T", "tags": ["chat", "reasoning"]},
        {"name": "gpt-4-turbo", "provider": "OpenAI", "category": "Proprietary", "icon": "&#129302;", "params": "1.7T", "tags": ["chat", "code"]},
        {"name": "gpt-3.5-turbo", "provider": "OpenAI", "category": "Proprietary", "icon": "&#129302;", "params": "175B", "tags": ["chat", "fast"]},
        {"name": "claude-3-opus", "provider": "Anthropic", "category": "Proprietary", "icon": "&#128172;", "params": "Unknown", "tags": ["reasoning", "safe"]},
        {"name": "claude-3-sonnet", "provider": "Anthropic", "category": "Proprietary", "icon": "&#128172;", "params": "Unknown", "tags": ["balanced"]},
        {"name": "claude-3-haiku", "provider": "Anthropic", "category": "Proprietary", "icon": "&#128172;", "params": "Unknown", "tags": ["fast", "efficient"]},
        {"name": "gemini-pro", "provider": "Google", "category": "Proprietary", "icon": "&#128142;", "params": "Unknown", "tags": ["multimodal"]},
        {"name": "gemini-1.5-pro", "provider": "Google", "category": "Proprietary", "icon": "&#128142;", "params": "Unknown", "tags": ["long-context"]},
        {"name": "llama3.2:3b", "provider": "Meta", "category": "Open Source", "icon": "&#129433;", "params": "3B", "tags": ["small", "efficient"]},
        {"name": "llama3.2:8b", "provider": "Meta", "category": "Open Source", "icon": "&#129433;", "params": "8B", "tags": ["balanced"]},
        {"name": "llama3.1:70b", "provider": "Meta", "category": "Open Source", "icon": "&#129433;", "params": "70B", "tags": ["large", "reasoning"]},
        {"name": "llama3.1:405b", "provider": "Meta", "category": "Open Source", "icon": "&#129433;", "params": "405B", "tags": ["flagship"]},
        {"name": "mistral:7b", "provider": "Mistral AI", "category": "Open Source", "icon": "&#127786;", "params": "7B", "tags": ["efficient"]},
        {"name": "mistral-nemo", "provider": "Mistral AI", "category": "Open Source", "icon": "&#127786;", "params": "12B", "tags": ["balanced"]},
        {"name": "mixtral:8x7b", "provider": "Mistral AI", "category": "Open Source", "icon": "&#127786;", "params": "46.7B", "tags": ["moe", "efficient"]},
        {"name": "mixtral:8x22b", "provider": "Mistral AI", "category": "Open Source", "icon": "&#127786;", "params": "141B", "tags": ["moe", "large"]},
        {"name": "qwen2.5:7b", "provider": "Alibaba", "category": "Open Source", "icon": "&#127968;", "params": "7B", "tags": ["multilingual"]},
        {"name": "qwen2.5:32b", "provider": "Alibaba", "category": "Open Source", "icon": "&#127968;", "params": "32B", "tags": ["coding", "math"]},
        {"name": "qwen2.5:72b", "provider": "Alibaba", "category": "Open Source", "icon": "&#127968;", "params": "72B", "tags": ["flagship"]},
        {"name": "gemma2:9b", "provider": "Google", "category": "Open Source", "icon": "&#128142;", "params": "9B", "tags": ["efficient"]},
        {"name": "gemma2:27b", "provider": "Google", "category": "Open Source", "icon": "&#128142;", "params": "27B", "tags": ["balanced"]},
        {"name": "phi-3:mini", "provider": "Microsoft", "category": "Open Source", "icon": "&#966;", "params": "3.8B", "tags": ["tiny", "efficient"]},
        {"name": "phi-3:medium", "provider": "Microsoft", "category": "Open Source", "icon": "&#966;", "params": "14B", "tags": ["balanced"]},
        {"name": "codellama:13b", "provider": "Meta", "category": "Open Source", "icon": "&#129433;", "params": "13B", "tags": ["code"]},
        {"name": "codellama:34b", "provider": "Meta", "category": "Open Source", "icon": "&#129433;", "params": "34B", "tags": ["code", "large"]},
        {"name": "deepseek-coder:33b", "provider": "DeepSeek", "category": "Open Source", "icon": "&#128187;", "params": "33B", "tags": ["code"]},
        {"name": "deepseek-v2.5", "provider": "DeepSeek", "category": "Open Source", "icon": "&#128187;", "params": "236B", "tags": ["moe", "flagship"]},
        {"name": "command-r-plus", "provider": "Cohere", "category": "Proprietary", "icon": "&#128640;", "params": "104B", "tags": ["rag", "enterprise"]},
        {"name": "yi:34b", "provider": "01.AI", "category": "Open Source", "icon": "&#127383;", "params": "34B", "tags": ["bilingual"]},
        {"name": "solar:10.7b", "provider": "Upstage", "category": "Open Source", "icon": "&#9728;", "params": "10.7B", "tags": ["efficient"]},
    ]

    popular_models_json = json.dumps(popular_models)

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HE-300 Ethics Benchmark Hub</title>
    <meta name="description" content="AI Ethics Benchmark Leaderboard - Compare model performance on moral reasoning tasks">
    <meta property="og:title" content="HE-300 Ethics Benchmark Hub">
    <meta property="og:description" content="Evaluating AI models on ethics and moral reasoning">
    <meta property="og:type" content="website">
    <style>
        :root {{
            --hf-yellow: #ffd21e;
            --primary: #6366f1;
            --primary-dark: #4f46e5;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --bg: #0d1117;
            --bg-secondary: #161b22;
            --bg-card: #21262d;
            --text: #f0f6fc;
            --text-secondary: #8b949e;
            --border: #30363d;
            --accent: #58a6ff;
            --gold: #ffd700;
            --silver: #c0c0c0;
            --bronze: #cd7f32;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.5;
        }}

        /* Navbar */
        .navbar {{
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border);
            padding: 0.75rem 1.5rem;
            display: flex;
            align-items: center;
            gap: 1.5rem;
            position: sticky;
            top: 0;
            z-index: 100;
        }}
        .navbar-brand {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-weight: 600;
            font-size: 1.1rem;
            color: var(--text);
            text-decoration: none;
        }}
        .navbar-brand .logo {{ font-size: 1.5rem; }}
        .nav-links {{
            display: flex;
            gap: 0.25rem;
            margin-left: auto;
        }}
        .nav-link {{
            color: var(--text-secondary);
            text-decoration: none;
            padding: 0.5rem 0.75rem;
            border-radius: 0.375rem;
            font-size: 0.875rem;
            transition: all 0.2s;
        }}
        .nav-link:hover {{ color: var(--text); background: var(--bg-card); }}
        .nav-link.active {{ color: var(--text); background: var(--primary); }}

        /* Hero */
        .hero {{
            background: linear-gradient(135deg, var(--primary) 0%, #8b5cf6 50%, #ec4899 100%);
            padding: 3rem 2rem;
            text-align: center;
        }}
        .hero h1 {{ font-size: 2.25rem; margin-bottom: 0.75rem; font-weight: 700; }}
        .hero p {{ font-size: 1rem; opacity: 0.9; max-width: 700px; margin: 0 auto 1rem; }}
        .hero-badges {{
            display: flex;
            gap: 0.5rem;
            justify-content: center;
            flex-wrap: wrap;
        }}
        .hero-badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.25rem;
            background: rgba(255,255,255,0.2);
            padding: 0.375rem 0.75rem;
            border-radius: 2rem;
            font-size: 0.8rem;
        }}

        /* Container */
        .container {{ max-width: 1400px; margin: 0 auto; padding: 1.5rem; }}

        /* Section Headers */
        .section-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1rem;
            padding-bottom: 0.75rem;
            border-bottom: 1px solid var(--border);
        }}
        .section-title {{
            font-size: 1.25rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        .section-count {{
            background: var(--bg-card);
            padding: 0.25rem 0.625rem;
            border-radius: 1rem;
            font-size: 0.75rem;
            color: var(--text-secondary);
        }}

        /* Stats Row */
        .stats-row {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 1rem;
            margin-bottom: 1.5rem;
        }}
        .stat-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 0.75rem;
            padding: 1rem;
            text-align: center;
        }}
        .stat-value {{ font-size: 1.75rem; font-weight: 700; color: var(--primary); }}
        .stat-label {{ font-size: 0.7rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.05em; }}

        /* Tabs */
        .tabs {{
            display: flex;
            gap: 0.25rem;
            background: var(--bg-secondary);
            padding: 0.25rem;
            border-radius: 0.5rem;
            margin-bottom: 1.5rem;
            overflow-x: auto;
        }}
        .tab {{
            padding: 0.5rem 1rem;
            border: none;
            background: transparent;
            color: var(--text-secondary);
            border-radius: 0.375rem;
            cursor: pointer;
            font-size: 0.875rem;
            font-weight: 500;
            white-space: nowrap;
            transition: all 0.2s;
        }}
        .tab:hover {{ color: var(--text); }}
        .tab.active {{ background: var(--bg-card); color: var(--text); }}

        /* Search & Filters */
        .search-filters {{
            display: flex;
            gap: 0.75rem;
            margin-bottom: 1.5rem;
            flex-wrap: wrap;
        }}
        .search-input {{
            flex: 1;
            min-width: 250px;
            background: var(--bg-card);
            border: 1px solid var(--border);
            color: var(--text);
            padding: 0.625rem 1rem;
            border-radius: 0.5rem;
            font-size: 0.875rem;
        }}
        .search-input::placeholder {{ color: var(--text-secondary); }}
        .filter-select {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            color: var(--text);
            padding: 0.625rem 1rem;
            border-radius: 0.5rem;
            font-size: 0.875rem;
        }}

        /* Model Cards Grid */
        .model-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 1rem;
        }}
        .model-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 0.75rem;
            overflow: hidden;
            transition: all 0.2s;
        }}
        .model-card:hover {{
            border-color: var(--primary);
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        }}
        .model-card.has-report {{ border-left: 3px solid var(--success); }}
        .model-card.no-report {{ opacity: 0.7; }}
        .model-card-header {{
            padding: 1rem;
            display: flex;
            align-items: flex-start;
            gap: 0.75rem;
        }}
        .model-avatar {{
            width: 44px;
            height: 44px;
            background: linear-gradient(135deg, var(--primary), #8b5cf6);
            border-radius: 0.5rem;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.25rem;
            flex-shrink: 0;
        }}
        .model-info {{ flex: 1; min-width: 0; }}
        .model-name {{
            font-weight: 600;
            font-size: 0.9rem;
            color: var(--text);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        .model-provider {{
            font-size: 0.75rem;
            color: var(--text-secondary);
            display: flex;
            align-items: center;
            gap: 0.25rem;
        }}
        .model-score {{
            text-align: right;
        }}
        .score-value {{
            font-size: 1.25rem;
            font-weight: 700;
        }}
        .score-value.high {{ color: var(--success); }}
        .score-value.medium {{ color: var(--warning); }}
        .score-value.low {{ color: var(--danger); }}
        .score-value.pending {{ color: var(--text-secondary); font-size: 0.8rem; }}
        .model-card-body {{ padding: 0 1rem 1rem; }}
        .model-tags {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.25rem;
            margin-bottom: 0.75rem;
        }}
        .tag {{
            padding: 0.2rem 0.5rem;
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 1rem;
            font-size: 0.65rem;
            color: var(--text-secondary);
        }}
        .tag.category {{ background: rgba(99, 102, 241, 0.2); border-color: var(--primary); color: var(--primary); }}
        .model-actions {{
            display: flex;
            gap: 0.5rem;
        }}
        .btn {{
            flex: 1;
            padding: 0.5rem 0.75rem;
            border: none;
            border-radius: 0.375rem;
            font-weight: 500;
            font-size: 0.75rem;
            cursor: pointer;
            text-decoration: none;
            text-align: center;
            transition: all 0.2s;
        }}
        .btn-primary {{ background: var(--primary); color: white; }}
        .btn-primary:hover {{ background: var(--primary-dark); }}
        .btn-secondary {{ background: var(--bg-secondary); color: var(--text); border: 1px solid var(--border); }}
        .btn-secondary:hover {{ background: var(--border); }}
        .btn:disabled {{ opacity: 0.5; cursor: not-allowed; }}

        /* Leaderboard Table */
        .leaderboard {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 0.75rem;
            overflow: hidden;
        }}
        .leaderboard table {{ width: 100%; border-collapse: collapse; }}
        .leaderboard th {{
            background: var(--bg-secondary);
            padding: 0.75rem 1rem;
            text-align: left;
            font-size: 0.7rem;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            border-bottom: 1px solid var(--border);
        }}
        .leaderboard td {{
            padding: 0.75rem 1rem;
            border-bottom: 1px solid var(--border);
            font-size: 0.875rem;
        }}
        .leaderboard tr:hover {{ background: var(--bg-secondary); }}
        .leaderboard tr:last-child td {{ border-bottom: none; }}
        .rank {{ font-weight: 700; }}
        .rank-1 {{ color: var(--gold); }}
        .rank-2 {{ color: var(--silver); }}
        .rank-3 {{ color: var(--bronze); }}
        .model-cell {{ display: flex; align-items: center; gap: 0.5rem; }}
        .mini-avatar {{
            width: 28px;
            height: 28px;
            background: linear-gradient(135deg, var(--primary), #8b5cf6);
            border-radius: 0.25rem;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.8rem;
        }}

        /* Popular Models Section */
        .popular-section {{
            margin-top: 2rem;
            padding-top: 2rem;
            border-top: 1px solid var(--border);
        }}

        /* Loading State */
        .loading {{
            text-align: center;
            padding: 3rem;
            color: var(--text-secondary);
        }}
        .loading-spinner {{
            width: 40px;
            height: 40px;
            border: 3px solid var(--border);
            border-top-color: var(--primary);
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 1rem;
        }}
        @keyframes spin {{ to {{ transform: rotate(360deg); }} }}

        /* Empty State */
        .empty-state {{
            text-align: center;
            padding: 3rem;
            background: var(--bg-card);
            border: 2px dashed var(--border);
            border-radius: 0.75rem;
        }}
        .empty-icon {{ font-size: 3rem; margin-bottom: 0.75rem; opacity: 0.5; }}

        /* Footer */
        footer {{
            text-align: center;
            padding: 2rem;
            color: var(--text-secondary);
            font-size: 0.8rem;
            border-top: 1px solid var(--border);
            margin-top: 3rem;
        }}
        footer a {{ color: var(--accent); text-decoration: none; }}
        footer a:hover {{ text-decoration: underline; }}

        @media (max-width: 768px) {{
            .navbar {{ flex-wrap: wrap; padding: 0.5rem 1rem; }}
            .nav-links {{ width: 100%; justify-content: center; margin: 0.5rem 0 0 0; }}
            .hero {{ padding: 2rem 1rem; }}
            .hero h1 {{ font-size: 1.5rem; }}
            .container {{ padding: 1rem; }}
            .model-grid {{ grid-template-columns: 1fr; }}
            .search-filters {{ flex-direction: column; }}
            .search-input {{ min-width: 100%; }}
            .tabs {{ width: 100%; }}
        }}
    </style>
</head>
<body>
    <nav class="navbar">
        <a href="#" class="navbar-brand">
            <span class="logo">&#128300;</span>
            <span>HE-300 Benchmark</span>
        </a>
        <div class="nav-links">
            <a href="#" class="nav-link active" onclick="showTab('leaderboard')">&#127942; Leaderboard</a>
            <a href="#" class="nav-link" onclick="showTab('models')">&#129302; Models</a>
            <a href="#" class="nav-link" onclick="showTab('popular')">&#11088; Popular</a>
            <a href="https://github.com/{repo}" class="nav-link" target="_blank">&#128279; GitHub</a>
        </div>
    </nav>

    <div class="hero">
        <h1>&#127942; AI Ethics Benchmark Leaderboard</h1>
        <p>Evaluating AI models on moral reasoning and ethical decision-making using the HE-300 benchmark suite. Compare scores across 300 curated ethics scenarios.</p>
        <div class="hero-badges">
            <span class="hero-badge">&#128200; <span id="reportCount">0</span> Reports</span>
            <span class="hero-badge">&#129302; <span id="modelCount">0</span> Models Tested</span>
            <span class="hero-badge">&#128279; {repo}</span>
        </div>
    </div>

    <div class="container">
        <div class="stats-row" id="statsRow">
            <div class="stat-card"><div class="stat-value">-</div><div class="stat-label">Reports</div></div>
            <div class="stat-card"><div class="stat-value">-</div><div class="stat-label">Models</div></div>
            <div class="stat-card"><div class="stat-value">-</div><div class="stat-label">Avg Score</div></div>
            <div class="stat-card"><div class="stat-value">-</div><div class="stat-label">Top Score</div></div>
        </div>

        <!-- Leaderboard Tab -->
        <div id="leaderboardTab">
            <div class="section-header">
                <h2 class="section-title">&#127942; Ethics Leaderboard <span class="section-count" id="leaderboardCount">0</span></h2>
            </div>
            <div class="search-filters">
                <input type="text" class="search-input" id="searchInput" placeholder="&#128269; Search models..." oninput="filterAndRender()">
                <select class="filter-select" id="sortSelect" onchange="filterAndRender()">
                    <option value="accuracy-desc">Highest Score</option>
                    <option value="accuracy-asc">Lowest Score</option>
                    <option value="date-desc">Newest First</option>
                    <option value="model">Model Name</option>
                </select>
            </div>
            <div id="leaderboardContent" class="leaderboard"></div>
        </div>

        <!-- Models Tab -->
        <div id="modelsTab" style="display: none;">
            <div class="section-header">
                <h2 class="section-title">&#129302; Tested Models</h2>
            </div>
            <div id="modelsContent" class="model-grid"></div>
        </div>

        <!-- Popular Models Tab -->
        <div id="popularTab" style="display: none;">
            <div class="section-header">
                <h2 class="section-title">&#11088; Popular Models <span class="section-count">{len(popular_models)}</span></h2>
            </div>
            <p style="color: var(--text-secondary); margin-bottom: 1rem; font-size: 0.875rem;">
                Common models from HuggingFace & Ollama. Green border indicates the model has been benchmarked.
            </p>
            <div class="search-filters">
                <input type="text" class="search-input" id="popularSearchInput" placeholder="&#128269; Search popular models..." oninput="filterPopular()">
                <select class="filter-select" id="categoryFilter" onchange="filterPopular()">
                    <option value="">All Categories</option>
                    <option value="Open Source">Open Source</option>
                    <option value="Proprietary">Proprietary</option>
                </select>
                <select class="filter-select" id="providerFilter" onchange="filterPopular()">
                    <option value="">All Providers</option>
                    <option value="OpenAI">OpenAI</option>
                    <option value="Anthropic">Anthropic</option>
                    <option value="Meta">Meta</option>
                    <option value="Google">Google</option>
                    <option value="Mistral AI">Mistral AI</option>
                    <option value="Alibaba">Alibaba</option>
                    <option value="Microsoft">Microsoft</option>
                    <option value="DeepSeek">DeepSeek</option>
                </select>
            </div>
            <div id="popularContent" class="model-grid"></div>
        </div>

        <div id="loadingState" class="loading">
            <div class="loading-spinner"></div>
            <p>Loading benchmark reports...</p>
        </div>

        <div id="emptyState" class="empty-state" style="display: none;">
            <div class="empty-icon">&#128196;</div>
            <h3>No Reports Published Yet</h3>
            <p>Run the HE-300 benchmark and deploy reports to see results here.</p>
        </div>
    </div>

    <footer>
        <p>Powered by <a href="https://github.com/{repo}">EthicsEngine Enterprise</a> &#8212; HE-300 Benchmark System</p>
        <p style="margin-top: 0.5rem;"><a href="{reports_path}/reports.json">&#128190; Raw Data (JSON)</a></p>
    </footer>

    <script>
        const popularModels = {popular_models_json};
        let reports = [];
        let filteredReports = [];
        let currentTab = 'leaderboard';

        async function init() {{
            try {{
                const response = await fetch('{reports_path}/reports.json');
                if (response.ok) {{
                    const data = await response.json();
                    reports = data.reports || [];
                    filteredReports = [...reports];
                }}
            }} catch (e) {{
                console.log('No reports yet');
            }}

            document.getElementById('loadingState').style.display = 'none';
            renderStats();
            filterAndRender();
            renderPopular();
        }}

        function renderStats() {{
            const total = reports.length;
            const models = new Set(reports.map(r => r.model_name)).size;
            const avgAcc = total > 0 ? reports.reduce((s, r) => s + r.accuracy, 0) / total : 0;
            const topAcc = total > 0 ? Math.max(...reports.map(r => r.accuracy)) : 0;

            document.getElementById('reportCount').textContent = total;
            document.getElementById('modelCount').textContent = models;

            document.getElementById('statsRow').innerHTML = `
                <div class="stat-card"><div class="stat-value">${{total}}</div><div class="stat-label">Reports</div></div>
                <div class="stat-card"><div class="stat-value">${{models}}</div><div class="stat-label">Models</div></div>
                <div class="stat-card"><div class="stat-value">${{total > 0 ? (avgAcc * 100).toFixed(1) + '%' : '-'}}</div><div class="stat-label">Avg Score</div></div>
                <div class="stat-card"><div class="stat-value">${{total > 0 ? (topAcc * 100).toFixed(1) + '%' : '-'}}</div><div class="stat-label">Top Score</div></div>
            `;
        }}

        function showTab(tab) {{
            currentTab = tab;
            document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
            event.target.classList.add('active');

            ['leaderboardTab', 'modelsTab', 'popularTab'].forEach(t => {{
                document.getElementById(t).style.display = 'none';
            }});
            document.getElementById(tab + 'Tab').style.display = 'block';

            if (tab === 'models') renderModels();
        }}

        function filterAndRender() {{
            const search = document.getElementById('searchInput').value.toLowerCase();
            const sort = document.getElementById('sortSelect').value;

            filteredReports = reports.filter(r => r.model_name.toLowerCase().includes(search));

            switch(sort) {{
                case 'accuracy-desc': filteredReports.sort((a, b) => b.accuracy - a.accuracy); break;
                case 'accuracy-asc': filteredReports.sort((a, b) => a.accuracy - b.accuracy); break;
                case 'date-desc': filteredReports.sort((a, b) => new Date(b.published_at) - new Date(a.published_at)); break;
                case 'model': filteredReports.sort((a, b) => a.model_name.localeCompare(b.model_name)); break;
            }}

            document.getElementById('leaderboardCount').textContent = filteredReports.length;

            if (filteredReports.length === 0) {{
                document.getElementById('leaderboardContent').innerHTML = '';
                document.getElementById('emptyState').style.display = 'block';
                return;
            }}

            document.getElementById('emptyState').style.display = 'none';
            renderLeaderboard();
        }}

        function getAccClass(acc) {{ return acc >= 0.7 ? 'high' : acc >= 0.5 ? 'medium' : 'low'; }}

        function getModelIcon(name) {{
            const n = name.toLowerCase();
            if (n.includes('gpt')) return '&#129302;';
            if (n.includes('claude')) return '&#128172;';
            if (n.includes('llama')) return '&#129433;';
            if (n.includes('mistral') || n.includes('mixtral')) return '&#127786;';
            if (n.includes('gemma') || n.includes('gemini')) return '&#128142;';
            if (n.includes('phi')) return '&#966;';
            if (n.includes('qwen')) return '&#127968;';
            if (n.includes('deepseek')) return '&#128187;';
            return '&#129302;';
        }}

        function getProvider(name) {{
            const n = name.toLowerCase();
            if (n.includes('gpt')) return 'OpenAI';
            if (n.includes('claude')) return 'Anthropic';
            if (n.includes('llama') || n.includes('codellama')) return 'Meta';
            if (n.includes('mistral') || n.includes('mixtral')) return 'Mistral AI';
            if (n.includes('gemma') || n.includes('gemini')) return 'Google';
            if (n.includes('phi')) return 'Microsoft';
            if (n.includes('qwen')) return 'Alibaba';
            if (n.includes('deepseek')) return 'DeepSeek';
            if (n.includes('command')) return 'Cohere';
            return 'Unknown';
        }}

        function renderLeaderboard() {{
            const rows = filteredReports.map((r, i) => {{
                const rank = i + 1;
                const rankClass = rank <= 3 ? 'rank-' + rank : '';
                const acc = (r.accuracy * 100).toFixed(1);
                const accClass = getAccClass(r.accuracy);
                const date = new Date(r.published_at).toLocaleDateString();
                const icon = getModelIcon(r.model_name);
                const provider = getProvider(r.model_name);

                return `<tr>
                    <td><span class="rank ${{rankClass}}">#${{rank}}</span></td>
                    <td><div class="model-cell"><div class="mini-avatar">${{icon}}</div><div><strong>${{r.model_name}}</strong><br><small style="color:var(--text-secondary)">${{provider}}</small></div></div></td>
                    <td><span class="score-value ${{accClass}}">${{acc}}%</span></td>
                    <td style="color: var(--text-secondary)">${{date}}</td>
                    <td><a href="${{r.pages_url}}" class="btn btn-primary" target="_blank">View</a></td>
                </tr>`;
            }}).join('');

            document.getElementById('leaderboardContent').innerHTML = `
                <table>
                    <thead><tr><th>Rank</th><th>Model</th><th>Ethics Score</th><th>Date</th><th></th></tr></thead>
                    <tbody>${{rows}}</tbody>
                </table>
            `;
        }}

        function renderModels() {{
            // Group reports by model, take best score
            const modelMap = {{}};
            reports.forEach(r => {{
                if (!modelMap[r.model_name] || r.accuracy > modelMap[r.model_name].accuracy) {{
                    modelMap[r.model_name] = r;
                }}
            }});

            const models = Object.values(modelMap).sort((a, b) => b.accuracy - a.accuracy);

            document.getElementById('modelsContent').innerHTML = models.map(r => {{
                const acc = (r.accuracy * 100).toFixed(1);
                const accClass = getAccClass(r.accuracy);
                const icon = getModelIcon(r.model_name);
                const provider = getProvider(r.model_name);

                return `
                    <div class="model-card has-report">
                        <div class="model-card-header">
                            <div class="model-avatar">${{icon}}</div>
                            <div class="model-info">
                                <div class="model-name">${{r.model_name}}</div>
                                <div class="model-provider">&#127970; ${{provider}}</div>
                            </div>
                            <div class="model-score">
                                <div class="score-value ${{accClass}}">${{acc}}%</div>
                            </div>
                        </div>
                        <div class="model-card-body">
                            <div class="model-actions">
                                <a href="${{r.pages_url}}" class="btn btn-primary" target="_blank">View Report</a>
                                <a href="${{r.raw_url}}" class="btn btn-secondary" download>Download</a>
                            </div>
                        </div>
                    </div>
                `;
            }}).join('');
        }}

        function filterPopular() {{
            const search = document.getElementById('popularSearchInput').value.toLowerCase();
            const category = document.getElementById('categoryFilter').value;
            const provider = document.getElementById('providerFilter').value;

            const filtered = popularModels.filter(m => {{
                const matchSearch = m.name.toLowerCase().includes(search) || m.provider.toLowerCase().includes(search);
                const matchCategory = !category || m.category === category;
                const matchProvider = !provider || m.provider === provider;
                return matchSearch && matchCategory && matchProvider;
            }});

            renderPopularModels(filtered);
        }}

        function renderPopular() {{ renderPopularModels(popularModels); }}

        function renderPopularModels(models) {{
            // Check which models have reports
            const testedModels = new Set(reports.map(r => r.model_name.toLowerCase()));

            document.getElementById('popularContent').innerHTML = models.map(m => {{
                const tested = testedModels.has(m.name.toLowerCase());
                const report = reports.find(r => r.model_name.toLowerCase() === m.name.toLowerCase());

                let scoreHtml = '<div class="score-value pending">Not tested</div>';
                if (tested && report) {{
                    const acc = (report.accuracy * 100).toFixed(1);
                    const accClass = getAccClass(report.accuracy);
                    scoreHtml = `<div class="score-value ${{accClass}}">${{acc}}%</div>`;
                }}

                return `
                    <div class="model-card ${{tested ? 'has-report' : 'no-report'}}">
                        <div class="model-card-header">
                            <div class="model-avatar">${{m.icon}}</div>
                            <div class="model-info">
                                <div class="model-name">${{m.name}}</div>
                                <div class="model-provider">&#127970; ${{m.provider}}</div>
                            </div>
                            <div class="model-score">${{scoreHtml}}</div>
                        </div>
                        <div class="model-card-body">
                            <div class="model-tags">
                                <span class="tag category">${{m.category}}</span>
                                <span class="tag">${{m.params}}</span>
                                ${{m.tags.map(t => `<span class="tag">${{t}}</span>`).join('')}}
                            </div>
                            <div class="model-actions">
                                ${{tested && report ?
                                    `<a href="${{report.pages_url}}" class="btn btn-primary" target="_blank">View Report</a>` :
                                    `<button class="btn btn-secondary" disabled>Not Benchmarked</button>`
                                }}
                            </div>
                        </div>
                    </div>
                `;
            }}).join('');
        }}

        init();
    </script>
</body>
</html>'''


def generate_jekyll_index(reports: List[PublishedReport], repo: str) -> str:
    """Generate a Jekyll-compatible index for reports."""
    return f'''---
layout: default
title: HE-300 Ethics Benchmark Reports
description: Public benchmark results for AI ethics evaluation
---

# HE-300 Ethics Benchmark Reports

This page contains published benchmark results from the EthicsEngine Enterprise HE-300 benchmark suite.

## Reports

| Date | Model | Accuracy | Report |
|------|-------|----------|--------|
{chr(10).join(f"| {r.published_at[:10]} | {r.model_name} | {r.accuracy:.1%} | [View]({r.pages_url}) |" for r in sorted(reports, key=lambda x: x.published_at, reverse=True))}

---

*Generated by [EthicsEngine Enterprise](https://github.com/{repo})*
'''


# --- API Endpoints ---

@router.get("/config")
async def get_config():
    """Get current GitHub configuration (token masked)."""
    config = load_github_config()
    if not config:
        return {"configured": False}

    return {
        "configured": True,
        "repo_full_name": config.repo_full_name,
        "target_branch": config.target_branch,
        "target_path": config.target_path,
        "token_set": bool(config.token),
        "token_preview": f"{config.token[:4]}...{config.token[-4:]}" if config.token and len(config.token) > 8 else "***"
    }


@router.post("/config")
async def set_config(config: GitHubConfig):
    """Set GitHub configuration for deployment."""
    # Validate token by making a test request
    try:
        response = await github_request("GET", "/user", config.token)
        if response.status_code != 200:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid GitHub token"
            )
        user_data = response.json()

        # Validate repo access
        owner, repo = config.repo_full_name.split("/")
        response = await github_request("GET", f"/repos/{owner}/{repo}", config.token)
        if response.status_code == 404:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Repository {config.repo_full_name} not found or not accessible"
            )
        elif response.status_code != 200:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot access repository: {response.text}"
            )

        repo_data = response.json()

        # Check write permissions
        permissions = repo_data.get("permissions", {})
        if not permissions.get("push", False):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Token does not have write access to this repository"
            )

        save_github_config(config)

        return {
            "status": "configured",
            "user": user_data.get("login"),
            "repo": config.repo_full_name,
            "permissions": permissions,
            "has_pages": repo_data.get("has_pages", False)
        }

    except httpx.RequestError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to connect to GitHub: {str(e)}"
        )


@router.delete("/config")
async def clear_config():
    """Clear GitHub configuration."""
    if GITHUB_CONFIG_FILE.exists():
        GITHUB_CONFIG_FILE.unlink()
    return {"status": "cleared"}


@router.get("/repos", response_model=List[GitHubRepo])
async def list_repos():
    """List repositories accessible with the configured token."""
    config = load_github_config()
    if not config:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="GitHub not configured. Please set up GitHub credentials first."
        )

    try:
        response = await github_request(
            "GET",
            "/user/repos?sort=updated&per_page=100",
            config.token
        )

        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"GitHub API error: {response.text}"
            )

        repos = []
        for repo in response.json():
            # Check if Pages is enabled
            has_pages = repo.get("has_pages", False)
            pages_url = None
            if has_pages:
                pages_url = f"https://{repo['owner']['login']}.github.io/{repo['name']}/"

            repos.append(GitHubRepo(
                full_name=repo["full_name"],
                name=repo["name"],
                owner=repo["owner"]["login"],
                description=repo.get("description"),
                private=repo.get("private", False),
                default_branch=repo.get("default_branch", "main"),
                html_url=repo["html_url"],
                has_pages=has_pages,
                pages_url=pages_url,
                permissions=repo.get("permissions", {})
            ))

        return repos

    except httpx.RequestError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to connect to GitHub: {str(e)}"
        )


@router.get("/repo/{owner}/{repo}/pages")
async def get_pages_status(owner: str, repo: str):
    """Get GitHub Pages status for a repository."""
    config = load_github_config()
    if not config:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="GitHub not configured"
        )

    try:
        response = await github_request(
            "GET",
            f"/repos/{owner}/{repo}/pages",
            config.token
        )

        if response.status_code == 404:
            return {"enabled": False, "message": "GitHub Pages not enabled for this repository"}

        if response.status_code == 200:
            data = response.json()
            return {
                "enabled": True,
                "url": data.get("html_url"),
                "source": data.get("source"),
                "status": data.get("status"),
                "build_type": data.get("build_type")
            }

        return {"enabled": False, "error": response.text}

    except httpx.RequestError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to connect to GitHub: {str(e)}"
        )


@router.post("/repo/{owner}/{repo}/enable-pages")
async def enable_pages(owner: str, repo: str, branch: str = "gh-pages"):
    """Enable GitHub Pages for a repository."""
    config = load_github_config()
    if not config:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="GitHub not configured"
        )

    try:
        # First, ensure the branch exists
        response = await github_request(
            "GET",
            f"/repos/{owner}/{repo}/branches/{branch}",
            config.token
        )

        if response.status_code == 404:
            # Need to create the branch - get default branch first
            response = await github_request("GET", f"/repos/{owner}/{repo}", config.token)
            if response.status_code != 200:
                raise HTTPException(status_code=400, detail="Cannot access repository")

            default_branch = response.json().get("default_branch", "main")

            # Get the SHA of the default branch
            response = await github_request(
                "GET",
                f"/repos/{owner}/{repo}/git/ref/heads/{default_branch}",
                config.token
            )

            if response.status_code == 200:
                sha = response.json()["object"]["sha"]

                # Create the new branch
                response = await github_request(
                    "POST",
                    f"/repos/{owner}/{repo}/git/refs",
                    config.token,
                    {"ref": f"refs/heads/{branch}", "sha": sha}
                )

                if response.status_code not in [200, 201]:
                    logger.warning(f"Failed to create branch: {response.text}")

        # Enable Pages
        response = await github_request(
            "POST",
            f"/repos/{owner}/{repo}/pages",
            config.token,
            {
                "source": {
                    "branch": branch,
                    "path": "/"
                },
                "build_type": "legacy"
            }
        )

        if response.status_code in [200, 201]:
            data = response.json()
            return {
                "status": "enabled",
                "url": data.get("html_url"),
                "branch": branch
            }
        elif response.status_code == 409:
            # Already enabled
            return {"status": "already_enabled", "branch": branch}
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Failed to enable Pages: {response.text}"
            )

    except httpx.RequestError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to connect to GitHub: {str(e)}"
        )


@router.post("/deploy", response_model=DeploymentResult)
async def deploy_reports(request: DeploymentRequest):
    """Deploy selected reports to GitHub Pages."""
    config = load_github_config()
    if not config:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="GitHub not configured. Please set up GitHub credentials first."
        )

    owner, repo = config.repo_full_name.split("/")
    deployed_reports = []
    failed_reports = []
    published_reports = []

    # Load each report and prepare for upload
    for report_id in request.report_ids:
        try:
            # Find the report file and metadata
            meta_file = None
            report_file = None

            for f in REPORTS_DIR.glob(f"report_*_{report_id}.meta.json"):
                meta_file = f
                break

            if not meta_file:
                failed_reports.append({"report_id": report_id, "error": "Report not found"})
                continue

            # Load metadata
            meta_data = json.loads(meta_file.read_text())
            report_path = Path(meta_data.get("file_path", ""))

            if not report_path.exists():
                # Try to find by pattern
                for f in REPORTS_DIR.glob(f"report_*_{report_id}.*"):
                    if not f.name.endswith(".meta.json"):
                        report_path = f
                        break

            if not report_path.exists():
                failed_reports.append({"report_id": report_id, "error": "Report file not found"})
                continue

            # Read report content
            content = report_path.read_bytes()
            content_b64 = base64.b64encode(content).decode()

            # Determine file path in repo
            file_name = report_path.name
            repo_path = f"{config.target_path}/{file_name}"

            # Check if file exists (to get SHA for update)
            existing_sha = None
            response = await github_request(
                "GET",
                f"/repos/{owner}/{repo}/contents/{repo_path}?ref={config.target_branch}",
                config.token
            )
            if response.status_code == 200:
                existing_sha = response.json().get("sha")

            # Create or update file
            commit_data = {
                "message": request.commit_message or f"Deploy HE-300 report {report_id}",
                "content": content_b64,
                "branch": config.target_branch
            }
            if existing_sha:
                commit_data["sha"] = existing_sha

            response = await github_request(
                "PUT",
                f"/repos/{owner}/{repo}/contents/{repo_path}",
                config.token,
                commit_data
            )

            if response.status_code in [200, 201]:
                deployed_reports.append(report_id)

                # Build published report entry
                ext = report_path.suffix.lower()
                pages_base = f"https://{owner}.github.io/{repo}"

                # Extract accuracy from metadata
                accuracy = meta_data.get("accuracy", 0.0)
                batch_id = meta_data.get("batch_id", "unknown")
                model_name = meta_data.get("model_name", "unknown")

                published_reports.append(PublishedReport(
                    report_id=report_id,
                    batch_id=batch_id,
                    model_name=model_name,
                    format=meta_data.get("format", "html"),
                    accuracy=accuracy,
                    published_at=datetime.now(timezone.utc).isoformat(),
                    pages_url=f"{pages_base}/{config.target_path}/{file_name}",
                    raw_url=f"https://raw.githubusercontent.com/{owner}/{repo}/{config.target_branch}/{config.target_path}/{file_name}",
                    file_name=file_name
                ))
            else:
                failed_reports.append({
                    "report_id": report_id,
                    "error": f"GitHub API error: {response.status_code}"
                })

        except Exception as e:
            logger.exception(f"Failed to deploy report {report_id}")
            failed_reports.append({"report_id": report_id, "error": str(e)})

    # Generate and upload index if requested
    index_url = None
    if request.generate_index and published_reports:
        try:
            # Load existing index to merge
            existing_reports = []
            index_path = f"{config.target_path}/index.html"

            response = await github_request(
                "GET",
                f"/repos/{owner}/{repo}/contents/{config.target_path}/reports.json?ref={config.target_branch}",
                config.token
            )
            if response.status_code == 200:
                data = response.json()
                existing_content = base64.b64decode(data["content"]).decode()
                existing_data = json.loads(existing_content)
                existing_reports = [PublishedReport(**r) for r in existing_data.get("reports", [])]

            # Merge reports (update existing, add new)
            reports_by_id = {r.report_id: r for r in existing_reports}
            for r in published_reports:
                reports_by_id[r.report_id] = r
            all_reports = list(reports_by_id.values())

            # Generate index HTML
            index_html = generate_index_html(all_reports, config.repo_full_name)
            index_b64 = base64.b64encode(index_html.encode()).decode()

            # Upload index.html
            existing_sha = None
            response = await github_request(
                "GET",
                f"/repos/{owner}/{repo}/contents/{index_path}?ref={config.target_branch}",
                config.token
            )
            if response.status_code == 200:
                existing_sha = response.json().get("sha")

            commit_data = {
                "message": "Update HE-300 reports index",
                "content": index_b64,
                "branch": config.target_branch
            }
            if existing_sha:
                commit_data["sha"] = existing_sha

            response = await github_request(
                "PUT",
                f"/repos/{owner}/{repo}/contents/{index_path}",
                config.token,
                commit_data
            )

            if response.status_code in [200, 201]:
                index_url = f"https://{owner}.github.io/{repo}/{config.target_path}/"

            # Also save reports.json for future merging
            reports_json = json.dumps({
                "reports": [r.model_dump() for r in all_reports],
                "last_updated": datetime.now(timezone.utc).isoformat()
            }, indent=2)
            reports_json_b64 = base64.b64encode(reports_json.encode()).decode()

            existing_sha = None
            response = await github_request(
                "GET",
                f"/repos/{owner}/{repo}/contents/{config.target_path}/reports.json?ref={config.target_branch}",
                config.token
            )
            if response.status_code == 200:
                existing_sha = response.json().get("sha")

            commit_data = {
                "message": "Update reports manifest",
                "content": reports_json_b64,
                "branch": config.target_branch
            }
            if existing_sha:
                commit_data["sha"] = existing_sha

            await github_request(
                "PUT",
                f"/repos/{owner}/{repo}/contents/{config.target_path}/reports.json",
                config.token,
                commit_data
            )

            # IMPORTANT: Also create a root index.html that redirects to reports
            # This ensures GitHub Pages shows the reports instead of README.md
            root_index_html = generate_root_index_html(config.repo_full_name, config.target_path)
            root_index_b64 = base64.b64encode(root_index_html.encode()).decode()

            existing_sha = None
            response = await github_request(
                "GET",
                f"/repos/{owner}/{repo}/contents/index.html?ref={config.target_branch}",
                config.token
            )
            if response.status_code == 200:
                existing_sha = response.json().get("sha")

            commit_data = {
                "message": "Update root index for GitHub Pages",
                "content": root_index_b64,
                "branch": config.target_branch
            }
            if existing_sha:
                commit_data["sha"] = existing_sha

            await github_request(
                "PUT",
                f"/repos/{owner}/{repo}/contents/index.html",
                config.token,
                commit_data
            )

            # Update the index_url to point to root
            index_url = f"https://{owner}.github.io/{repo}/"

        except Exception as e:
            logger.exception("Failed to generate index")

    pages_base = f"https://{owner}.github.io/{repo}"

    return DeploymentResult(
        status="success" if deployed_reports else "failed",
        deployed_reports=deployed_reports,
        failed_reports=failed_reports,
        pages_url=f"{pages_base}/{config.target_path}/",
        index_url=index_url
    )


@router.get("/published")
async def list_published_reports():
    """List all reports published to GitHub Pages."""
    config = load_github_config()
    if not config:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="GitHub not configured"
        )

    owner, repo = config.repo_full_name.split("/")

    try:
        response = await github_request(
            "GET",
            f"/repos/{owner}/{repo}/contents/{config.target_path}/reports.json?ref={config.target_branch}",
            config.token
        )

        if response.status_code == 404:
            return {"reports": [], "total": 0}

        if response.status_code == 200:
            data = response.json()
            content = base64.b64decode(data["content"]).decode()
            reports_data = json.loads(content)
            return reports_data

        return {"reports": [], "total": 0, "error": response.text}

    except Exception as e:
        logger.exception("Failed to list published reports")
        return {"reports": [], "total": 0, "error": str(e)}


@router.delete("/published/{report_id}")
async def unpublish_report(report_id: str):
    """Remove a report from GitHub Pages."""
    config = load_github_config()
    if not config:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="GitHub not configured"
        )

    owner, repo = config.repo_full_name.split("/")

    try:
        # Find the file in the repo
        response = await github_request(
            "GET",
            f"/repos/{owner}/{repo}/contents/{config.target_path}?ref={config.target_branch}",
            config.token
        )

        if response.status_code != 200:
            raise HTTPException(status_code=404, detail="Reports directory not found")

        files = response.json()
        target_file = None

        for f in files:
            if report_id in f["name"] and not f["name"].endswith(".json"):
                target_file = f
                break

        if not target_file:
            raise HTTPException(status_code=404, detail="Report not found in repository")

        # Delete the file
        response = await github_request(
            "DELETE",
            f"/repos/{owner}/{repo}/contents/{target_file['path']}",
            config.token,
            {
                "message": f"Remove report {report_id}",
                "sha": target_file["sha"],
                "branch": config.target_branch
            }
        )

        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Failed to delete: {response.text}"
            )

        # Update reports.json
        response = await github_request(
            "GET",
            f"/repos/{owner}/{repo}/contents/{config.target_path}/reports.json?ref={config.target_branch}",
            config.token
        )

        if response.status_code == 200:
            data = response.json()
            content = base64.b64decode(data["content"]).decode()
            reports_data = json.loads(content)

            # Remove the report
            reports_data["reports"] = [
                r for r in reports_data["reports"]
                if r.get("report_id") != report_id
            ]
            reports_data["last_updated"] = datetime.now(timezone.utc).isoformat()

            # Update the file
            new_content = base64.b64encode(json.dumps(reports_data, indent=2).encode()).decode()

            await github_request(
                "PUT",
                f"/repos/{owner}/{repo}/contents/{config.target_path}/reports.json",
                config.token,
                {
                    "message": "Update reports manifest after removal",
                    "content": new_content,
                    "sha": data["sha"],
                    "branch": config.target_branch
                }
            )

            # Regenerate index
            all_reports = [PublishedReport(**r) for r in reports_data["reports"]]
            index_html = generate_index_html(all_reports, config.repo_full_name)
            index_b64 = base64.b64encode(index_html.encode()).decode()

            response = await github_request(
                "GET",
                f"/repos/{owner}/{repo}/contents/{config.target_path}/index.html?ref={config.target_branch}",
                config.token
            )
            if response.status_code == 200:
                index_data = response.json()
                await github_request(
                    "PUT",
                    f"/repos/{owner}/{repo}/contents/{config.target_path}/index.html",
                    config.token,
                    {
                        "message": "Regenerate index after removal",
                        "content": index_b64,
                        "sha": index_data["sha"],
                        "branch": config.target_branch
                    }
                )

        return {"status": "removed", "report_id": report_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to unpublish report")
        raise HTTPException(status_code=500, detail=str(e))
