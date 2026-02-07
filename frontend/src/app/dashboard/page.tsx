"use client";

import { useState, useEffect, useCallback } from "react";
import { getEvaluations, patchEvaluation, type EvaluationSummary } from "@/lib/api";
import { formatAccuracy, formatDate } from "@/lib/utils";
import { Badge } from "@/components/ui/Badge";
import { ScoreBar } from "@/components/ui/ScoreBar";
import Link from "next/link";

export default function DashboardPage() {
  const [token, setToken] = useState("");
  const [authenticated, setAuthenticated] = useState(false);
  const [evaluations, setEvaluations] = useState<EvaluationSummary[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadEvals = useCallback(async (t: string, p: number) => {
    setLoading(true);
    setError(null);
    try {
      const data = await getEvaluations(t, p, 20);
      setEvaluations(data.evaluations);
      setTotal(data.total);
      setAuthenticated(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load evaluations");
    } finally {
      setLoading(false);
    }
  }, []);

  const handleAuth = (e: React.FormEvent) => {
    e.preventDefault();
    if (token.trim()) {
      loadEvals(token.trim(), 1);
    }
  };

  const toggleVisibility = async (evalId: string, currentVis: string) => {
    const newVis = currentVis === "public" ? "private" : "public";
    try {
      await patchEvaluation(token, evalId, { visibility: newVis });
      loadEvals(token, page);
    } catch (e) {
      alert(e instanceof Error ? e.message : "Failed to update visibility");
    }
  };

  if (!authenticated) {
    return (
      <div className="mx-auto max-w-lg px-6 py-20">
        <h1 className="text-2xl font-bold mb-2">Dashboard</h1>
        <p className="text-text-secondary mb-8">
          Enter your API key or JWT token to view your evaluations.
        </p>

        <form onSubmit={handleAuth} className="space-y-4">
          <div>
            <label className="block text-sm font-medium mb-1">API Key / Bearer Token</label>
            <input
              type="password"
              value={token}
              onChange={(e) => setToken(e.target.value)}
              placeholder="sk-... or JWT"
              className="w-full rounded-lg border border-border bg-bg-card px-4 py-2.5 text-sm text-text placeholder:text-text-muted focus:outline-none focus:border-accent"
            />
          </div>
          <button
            type="submit"
            className="w-full py-2.5 rounded-lg bg-accent hover:bg-accent-hover text-white font-medium text-sm transition-colors"
          >
            Connect
          </button>
          {error && (
            <p className="text-red text-sm">{error}</p>
          )}
        </form>

        <div className="mt-8 rounded-xl border border-border-subtle bg-bg-card p-5">
          <h3 className="text-sm font-semibold mb-2">Don&apos;t have an API key?</h3>
          <p className="text-sm text-text-secondary leading-relaxed">
            Get started with a free community account.{" "}
            <Link href="/pricing" className="text-accent hover:text-accent-hover">
              View plans &rarr;
            </Link>
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-7xl px-6 py-12">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold">Your Evaluations</h1>
          <p className="text-sm text-text-secondary mt-1">
            {total} evaluation{total !== 1 ? "s" : ""} found
          </p>
        </div>
        <button
          onClick={() => { setAuthenticated(false); setToken(""); }}
          className="text-sm text-text-muted hover:text-text transition-colors"
        >
          Disconnect
        </button>
      </div>

      {/* Table */}
      <div className="rounded-xl border border-border bg-bg-card overflow-hidden">
        {loading ? (
          <div className="p-12 text-center text-text-muted">Loading...</div>
        ) : evaluations.length === 0 ? (
          <div className="p-12 text-center text-text-muted">
            <p className="text-lg font-medium">No evaluations yet</p>
            <p className="mt-2 text-sm">
              Run your first benchmark via the API to see results here.
            </p>
          </div>
        ) : (
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border text-left text-xs text-text-muted uppercase tracking-wider">
                <th className="py-3 px-4">Model</th>
                <th className="py-3 px-4">Type</th>
                <th className="py-3 px-4 w-44">Score</th>
                <th className="py-3 px-4">Status</th>
                <th className="py-3 px-4">Visibility</th>
                <th className="py-3 px-4">Badges</th>
                <th className="py-3 px-4">Date</th>
              </tr>
            </thead>
            <tbody>
              {evaluations.map((ev) => (
                <tr key={ev.id} className="border-b border-border-subtle score-row">
                  <td className="py-3 px-4">
                    <div>
                      <p className="font-medium">{ev.agent_name || ev.target_model || "Unknown"}</p>
                      {ev.target_provider && (
                        <p className="text-xs text-text-muted">{ev.target_provider}</p>
                      )}
                    </div>
                  </td>
                  <td className="py-3 px-4">
                    <span className={`badge ${ev.eval_type === "frontier" ? "bg-accent-subtle text-accent" : "bg-bg-elevated text-text-secondary"}`}>
                      {ev.eval_type}
                    </span>
                  </td>
                  <td className="py-3 px-4">
                    {ev.accuracy !== null ? (
                      <div className="flex items-center gap-2">
                        <span className="font-mono font-semibold w-14">
                          {formatAccuracy(ev.accuracy)}
                        </span>
                        <ScoreBar value={ev.accuracy} size="sm" showLabel={false} />
                      </div>
                    ) : (
                      <span className="text-text-muted">--</span>
                    )}
                  </td>
                  <td className="py-3 px-4">
                    <span className={`badge ${
                      ev.status === "completed" ? "bg-green-subtle text-green" :
                      ev.status === "running" ? "bg-amber-subtle text-amber" :
                      ev.status === "failed" ? "bg-red-subtle text-red" :
                      "bg-bg-elevated text-text-muted"
                    }`}>
                      {ev.status}
                    </span>
                  </td>
                  <td className="py-3 px-4">
                    <button
                      onClick={() => toggleVisibility(ev.id, ev.visibility)}
                      className={`badge cursor-pointer transition-colors ${
                        ev.visibility === "public"
                          ? "bg-green-subtle text-green hover:bg-green/20"
                          : "bg-bg-elevated text-text-muted hover:bg-bg-card-hover"
                      }`}
                      title={ev.eval_type === "frontier" ? "Frontier evals are always public" : "Click to toggle"}
                      disabled={ev.eval_type === "frontier"}
                    >
                      {ev.visibility}
                    </button>
                  </td>
                  <td className="py-3 px-4">
                    <div className="flex flex-wrap gap-1">
                      {ev.badges?.map((b) => <Badge key={b} name={b} />) ?? null}
                    </div>
                  </td>
                  <td className="py-3 px-4 text-text-muted text-xs">
                    {formatDate(ev.completed_at || ev.created_at)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* Pagination */}
      {total > 20 && (
        <div className="flex justify-center gap-2 mt-6">
          <button
            onClick={() => { setPage(page - 1); loadEvals(token, page - 1); }}
            disabled={page <= 1}
            className="px-4 py-2 rounded-lg border border-border text-sm disabled:opacity-30 hover:bg-bg-card transition-colors"
          >
            Previous
          </button>
          <span className="px-4 py-2 text-sm text-text-muted">
            Page {page} of {Math.ceil(total / 20)}
          </span>
          <button
            onClick={() => { setPage(page + 1); loadEvals(token, page + 1); }}
            disabled={page >= Math.ceil(total / 20)}
            className="px-4 py-2 rounded-lg border border-border text-sm disabled:opacity-30 hover:bg-bg-card transition-colors"
          >
            Next
          </button>
        </div>
      )}
    </div>
  );
}
