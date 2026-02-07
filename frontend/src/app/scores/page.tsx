import { getFrontierScores } from "@/lib/api";
import { FrontierTable } from "@/components/ui/FrontierTable";
import { StatCard } from "@/components/ui/StatCard";
import { formatAccuracy, formatDate } from "@/lib/utils";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Frontier Model Scores",
  description: "Live HE-300 ethical benchmark scores for GPT-4o, Claude, Gemini, Llama, and more.",
};

export const revalidate = 60;

export default async function ScoresPage() {
  let scores = null;
  let error = null;
  try {
    scores = await getFrontierScores();
  } catch (e) {
    error = e instanceof Error ? e.message : "Failed to load scores";
  }

  const list = scores?.scores ?? [];
  const sorted = [...list].sort((a, b) => b.accuracy - a.accuracy);
  const top = sorted[0];
  const avg = list.length
    ? list.reduce((s, m) => s + m.accuracy, 0) / list.length
    : 0;
  const withExcellence = list.filter((m) => m.badges.includes("excellence")).length;

  return (
    <div className="mx-auto max-w-7xl px-6 py-12">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold">Frontier Model Scores</h1>
        <p className="text-text-secondary mt-2">
          Latest HE-300 ethical benchmark results. Evaluated weekly across 300 scenarios.
        </p>
        {scores?.updated_at && (
          <p className="text-xs text-text-muted mt-1">
            Last updated: {formatDate(scores.updated_at)}
          </p>
        )}
      </div>

      {/* Summary stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <StatCard label="Models Evaluated" value={String(list.length)} />
        <StatCard
          label="Highest Score"
          value={top ? formatAccuracy(top.accuracy) : "--"}
          sub={top?.display_name}
          glow="gold"
        />
        <StatCard
          label="Average Score"
          value={avg ? formatAccuracy(avg) : "--"}
        />
        <StatCard
          label="Excellence Badge"
          value={String(withExcellence)}
          sub={withExcellence ? `of ${list.length} models` : "No models yet"}
          glow={withExcellence ? "green" : undefined}
        />
      </div>

      {/* Full table */}
      <div className="rounded-xl border border-border bg-bg-card overflow-hidden">
        {error ? (
          <div className="p-12 text-center text-text-muted">
            <p className="text-lg font-medium">Unable to load scores</p>
            <p className="mt-2 text-sm">{error}</p>
          </div>
        ) : list.length > 0 ? (
          <FrontierTable scores={list} />
        ) : (
          <div className="p-12 text-center text-text-muted">
            <p className="text-lg font-medium">First frontier sweep pending</p>
            <p className="mt-2 text-sm">The weekly evaluation pipeline will populate scores automatically.</p>
          </div>
        )}
      </div>

      {/* Methodology note */}
      <div className="mt-8 rounded-xl border border-border-subtle bg-bg-card p-6">
        <h3 className="font-semibold mb-2">Methodology</h3>
        <p className="text-sm text-text-secondary leading-relaxed">
          Each model is evaluated on the HE-300 benchmark: 300 ethical scenarios across
          virtue ethics (150) and hard commonsense moral reasoning (150). Scenarios are
          sampled deterministically using a fixed seed for reproducibility. Evaluation uses
          dual-method scoring (heuristic classification + semantic analysis) with full
          response capture. Results are cryptographically bound to a unique trace ID
          for auditability. Evaluations run weekly via automated pipeline.
        </p>
      </div>
    </div>
  );
}
