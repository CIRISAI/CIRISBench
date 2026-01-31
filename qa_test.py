#!/usr/bin/env python3
"""
QA Test Script for CIRISBench HE-300 Features

Tests:
1. Python imports and syntax
2. Agent card data models
3. Report generation models
4. HE-300 runner components
5. API router imports
"""

import sys
import asyncio
from pathlib import Path

# Add engine to path
engine_path = Path(__file__).parent / "engine"
sys.path.insert(0, str(engine_path))

PASSED = 0
FAILED = 0

def test(name: str):
    """Decorator for test functions."""
    def decorator(func):
        def wrapper():
            global PASSED, FAILED
            try:
                result = func()
                if result is None or result:
                    print(f"  ✓ {name}")
                    PASSED += 1
                else:
                    print(f"  ✗ {name}: returned False")
                    FAILED += 1
            except Exception as e:
                print(f"  ✗ {name}: {e}")
                FAILED += 1
        return wrapper
    return decorator


# ============================================================
# Test 1: Python Imports
# ============================================================
print("\n1. Testing Python Imports")
print("-" * 40)

@test("Import he300_runner module")
def test_import_runner():
    from core import he300_runner
    return True

@test("Import reports router")
def test_import_reports():
    from api.routers import reports
    return True

@test("Import he300 router")
def test_import_he300():
    try:
        from api.routers import he300
        return True
    except ImportError as e:
        if "ujson" in str(e):
            print("    (skipped - ujson not installed)")
            return True  # Skip if ujson not available
        raise

test_import_runner()
test_import_reports()
test_import_he300()


# ============================================================
# Test 2: Agent Card Data Model
# ============================================================
print("\n2. Testing Agent Card Data Model")
print("-" * 40)

@test("Create AgentCard dataclass")
def test_agent_card_create():
    from core.he300_runner import AgentCard
    card = AgentCard(
        name="Test Agent",
        version="1.0.0",
        provider_name="Test Provider",
        did="did:web:test.com:agent",
        skills=["ethics", "reasoning"]
    )
    assert card.name == "Test Agent"
    assert card.version == "1.0.0"
    assert card.did == "did:web:test.com:agent"
    return True

@test("AgentCard with empty fields")
def test_agent_card_empty():
    from core.he300_runner import AgentCard
    card = AgentCard()
    assert card.name == ""
    assert card.did is None
    assert card.skills == []
    return True

test_agent_card_create()
test_agent_card_empty()


# ============================================================
# Test 3: BatchResult with Agent Card Fields
# ============================================================
print("\n3. Testing BatchResult with Agent Card Fields")
print("-" * 40)

@test("Create BatchResult with agent card info")
def test_batch_result_agent_card():
    from core.he300_runner import BatchResult
    result = BatchResult(
        batch_id="test-batch-001",
        total=10,
        correct=8,
        accuracy=0.8,
        agent_card_name="Purple Agent",
        agent_card_version="2.0.0",
        agent_card_provider="CIRIS AI",
        agent_card_did="did:web:ciris.ai:purple",
        agent_card_skills=["he300_evaluation", "ethics"]
    )
    assert result.agent_card_name == "Purple Agent"
    assert result.agent_card_did == "did:web:ciris.ai:purple"
    assert len(result.agent_card_skills) == 2
    return True

@test("BatchResult serialization")
def test_batch_result_serialize():
    from core.he300_runner import BatchResult
    result = BatchResult(
        batch_id="test-batch-002",
        agent_card_name="Test Agent"
    )
    data = result.model_dump()
    assert "agent_card_name" in data
    assert "agent_card_did" in data
    assert "agent_card_skills" in data
    return True

test_batch_result_agent_card()
test_batch_result_serialize()


# ============================================================
# Test 4: Report Models with Agent Card Fields
# ============================================================
print("\n4. Testing Report Models with Agent Card Fields")
print("-" * 40)

@test("BenchmarkRunSummary with agent card fields")
def test_summary_agent_card():
    from api.routers.reports import BenchmarkRunSummary
    summary = BenchmarkRunSummary(
        batch_id="test-001",
        model_name="gpt-4o",
        identity_id="Neutral",
        guidance_id="Utilitarian",
        total_scenarios=10,
        correct_predictions=8,
        overall_accuracy=0.8,
        avg_latency_ms=100,
        total_errors=0,
        categories=[],
        started_at="2026-01-31T00:00:00Z",
        completed_at="2026-01-31T00:01:00Z",
        processing_time_ms=60000,
        agent_name="Test Agent",
        agent_type="eee_purple",
        protocol="a2a",
        agent_card_name="Official Name",
        agent_card_version="1.0.0",
        agent_card_provider="Test Org",
        agent_card_did="did:web:test.org:agent"
    )
    assert summary.agent_card_name == "Official Name"
    assert summary.agent_card_did == "did:web:test.org:agent"
    return True

@test("EvaluationDetailReport model")
def test_eval_detail():
    from api.routers.reports import EvaluationDetailReport
    heuristic = EvaluationDetailReport(
        classification="ethical",
        label=0,
        confidence=0.85,
        method="heuristic"
    )
    assert heuristic.classification == "ethical"
    assert heuristic.method == "heuristic"
    return True

@test("ScenarioDetail with dual evaluation")
def test_scenario_dual_eval():
    from api.routers.reports import ScenarioDetail, EvaluationDetailReport
    scenario = ScenarioDetail(
        scenario_id="cm-001",
        category="commonsense",
        input_text="I helped an old lady across the road.",
        expected_label=0,
        predicted_label=0,
        model_response="This is ethical.",
        is_correct=True,
        latency_ms=150,
        heuristic_eval=EvaluationDetailReport(
            classification="ethical",
            label=0,
            confidence=0.9,
            method="heuristic"
        ),
        semantic_eval=EvaluationDetailReport(
            classification="ethical",
            label=0,
            confidence=0.95,
            method="semantic"
        ),
        evaluations_agree=True
    )
    assert scenario.heuristic_eval.classification == "ethical"
    assert scenario.semantic_eval.confidence == 0.95
    assert scenario.evaluations_agree == True
    return True

test_summary_agent_card()
test_eval_detail()
test_scenario_dual_eval()


# ============================================================
# Test 5: fetch_agent_card function
# ============================================================
print("\n5. Testing fetch_agent_card Function")
print("-" * 40)

@test("fetch_agent_card function exists")
def test_fetch_agent_card_exists():
    from core.he300_runner import fetch_agent_card
    import inspect
    assert inspect.iscoroutinefunction(fetch_agent_card)
    return True

@test("fetch_agent_card handles invalid URL gracefully")
def test_fetch_agent_card_invalid():
    from core.he300_runner import fetch_agent_card

    async def run_test():
        result = await fetch_agent_card(
            agent_url="http://localhost:99999/invalid",
            verify_ssl=False,
            timeout=2.0
        )
        # Should return None for invalid URL, not raise exception
        return result is None

    return asyncio.run(run_test())

test_fetch_agent_card_exists()
test_fetch_agent_card_invalid()


# ============================================================
# Test 6: HTML Report Generation
# ============================================================
print("\n6. Testing HTML Report Generation")
print("-" * 40)

@test("Generate HTML report with agent card")
def test_html_report_agent_card():
    from api.routers.reports import (
        generate_html_report,
        ReportRequest,
        BenchmarkRunSummary,
        ScenarioDetail,
        EvaluationDetailReport,
        ReportFormat
    )

    summary = BenchmarkRunSummary(
        batch_id="qa-test-001",
        model_name="gpt-4o-mini",
        identity_id="Neutral",
        guidance_id="Utilitarian",
        total_scenarios=2,
        correct_predictions=2,
        overall_accuracy=1.0,
        avg_latency_ms=100,
        total_errors=0,
        categories=[],
        started_at="2026-01-31T00:00:00Z",
        completed_at="2026-01-31T00:01:00Z",
        processing_time_ms=60000,
        agent_name="QA Test Agent",
        agent_type="eee_purple",
        protocol="a2a",
        agent_url="http://localhost:9000/a2a",
        agent_card_name="QA Purple Agent",
        agent_card_version="1.2.3",
        agent_card_provider="QA Team",
        agent_card_did="did:web:qa.test:purple-agent"
    )

    scenarios = [
        ScenarioDetail(
            scenario_id="qa-001",
            category="commonsense",
            input_text="I helped someone in need.",
            expected_label=0,
            predicted_label=0,
            model_response="This is ethical behavior.",
            is_correct=True,
            latency_ms=100,
            heuristic_eval=EvaluationDetailReport(
                classification="ethical",
                label=0,
                confidence=0.9,
                method="heuristic"
            ),
            semantic_eval=EvaluationDetailReport(
                classification="ethical",
                label=0,
                confidence=0.95,
                method="semantic"
            ),
            evaluations_agree=True
        )
    ]

    request = ReportRequest(
        batch_id="qa-test-001",
        summary=summary,
        scenarios=scenarios,
        format=ReportFormat.HTML
    )

    html = generate_html_report(request, None)

    # Check for agent card badge
    assert "QA Purple Agent" in html
    assert "agent-card-badge" in html

    # Check for DID badge
    assert "did:web:qa.test:purple-agent" in html
    assert "did-badge" in html

    # Check for dual evaluation columns
    assert "Heuristic" in html
    assert "Semantic" in html

    # Check for agent type badge
    assert "eee_purple" in html or "EEE Purple" in html

    return True

@test("HTML report includes evaluation data in JSON export")
def test_html_report_json_export():
    from api.routers.reports import (
        generate_html_report,
        ReportRequest,
        BenchmarkRunSummary,
        ScenarioDetail,
        EvaluationDetailReport,
        ReportFormat
    )

    summary = BenchmarkRunSummary(
        batch_id="qa-test-002",
        model_name="test-model",
        identity_id="Neutral",
        guidance_id="Neutral",
        total_scenarios=1,
        correct_predictions=1,
        overall_accuracy=1.0,
        avg_latency_ms=50,
        total_errors=0,
        categories=[],
        started_at="2026-01-31T00:00:00Z",
        completed_at="2026-01-31T00:00:30Z",
        processing_time_ms=30000
    )

    scenarios = [
        ScenarioDetail(
            scenario_id="qa-002",
            category="virtue",
            input_text="Test scenario",
            expected_label=1,
            predicted_label=1,
            model_response="Unethical",
            is_correct=True,
            latency_ms=50,
            heuristic_eval=EvaluationDetailReport(
                classification="unethical",
                label=1,
                confidence=0.8,
                method="heuristic"
            ),
            semantic_eval=EvaluationDetailReport(
                classification="unethical",
                label=1,
                confidence=0.85,
                method="semantic"
            ),
            evaluations_agree=True
        )
    ]

    request = ReportRequest(
        batch_id="qa-test-002",
        summary=summary,
        scenarios=scenarios,
        format=ReportFormat.HTML
    )

    html = generate_html_report(request, None)

    # Check embedded JSON contains evaluation data
    assert '"heuristic_eval"' in html
    assert '"semantic_eval"' in html
    assert '"evaluations_agree"' in html

    return True

test_html_report_agent_card()
test_html_report_json_export()


# ============================================================
# Test 7: API Router Endpoints
# ============================================================
print("\n7. Testing API Router Endpoints")
print("-" * 40)

@test("Reports router has required endpoints")
def test_reports_endpoints():
    from api.routers.reports import router
    routes = [r.path for r in router.routes]
    assert "/reports/generate" in routes
    assert "/reports/" in routes
    assert "/reports/download/{report_id}" in routes
    return True

@test("HE-300 router has agentbeats endpoint")
def test_he300_agentbeats():
    try:
        from api.routers.he300 import router
        routes = [r.path for r in router.routes]
        assert "/agentbeats/run" in routes
        return True
    except ImportError as e:
        if "ujson" in str(e):
            print("    (skipped - ujson not installed)")
            return True  # Skip if ujson not available
        raise

test_reports_endpoints()
test_he300_agentbeats()


# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 50)
print(f"QA TEST SUMMARY")
print("=" * 50)
print(f"  Passed: {PASSED}")
print(f"  Failed: {FAILED}")
print(f"  Total:  {PASSED + FAILED}")
print("=" * 50)

if FAILED > 0:
    print("\n❌ SOME TESTS FAILED")
    sys.exit(1)
else:
    print("\n✅ ALL TESTS PASSED")
    sys.exit(0)
