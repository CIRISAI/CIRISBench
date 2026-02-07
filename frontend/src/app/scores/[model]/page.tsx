import { getModelHistory, getFrontierScores } from "@/lib/api";
import { AccuracyHistory } from "@/components/charts/AccuracyHistory";
import { CategoryBreakdown } from "@/components/charts/CategoryBreakdown";
import { StatCard } from "@/components/ui/StatCard";
import { Badge } from "@/components/ui/Badge";
import { formatAccuracy, formatLatency, formatDate, providerColor } from "@/lib/utils";
import Link from "next/link";
import type { Metadata } from "next";

interface Props {
  params: Promise<{ model: string }>;
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { model } = await params;
  const modelId = decodeURIComponent(model);
  return {
    title: `${modelId} Ethics Score`,
    description: `HE-300 ethical benchmark results and history for ${modelId}.`,
  };
}

export const revalidate = 60;

export default async function ModelPage({ params }: Props) {
  const { model } = await params;
  const modelId = decodeURIComponent(model);

  let history = null;
  let currentScore = null;
  let error = null;

  try {
    history = await getModelHistory(modelId);
  } catch (e) {
    error = e instanceof Error ? e.message : "Failed to load model data";
  }

  // Also try to get current score from the frontier list for trend info
  try {
    const all = await getFrontierScores();
    currentScore = all.scores.find((s) => s.model_id === modelId) ?? null;
  } catch {
    // non-fatal
  }

  const latest = history?.evaluations?.[0];
  const evalCount = history?.evaluations?.length ?? 0;

  return (
    <div className="mx-auto max-w-7xl px-6 py-12">
      {/* Breadcrumb */}
      <div className="mb-6">
        <Link href="/scores" className="text-sm text-text-muted hover:text-accent transition-colors">
          &larr; All Frontier Scores
        </Link>
      </div>

      {error ? (
        <div className="rounded-xl border border-border bg-bg-card p-12 text-center">
          <p className="text-lg font-medium text-text-muted">Model not found</p>
          <p className="mt-2 text-sm text-text-muted">{error}</p>
        </div>
      ) : (
        <>
          {/* Model header */}
          <div className="flex items-start gap-4 mb-8">
            <div
              className="h-4 w-4 rounded-full mt-2 shrink-0"
              style={{ backgroundColor: providerColor(history?.provider ?? "") }}
            />
            <div>
              <h1 className="text-3xl font-bold">{history?.display_name ?? modelId}</h1>
              <p className="text-text-secondary">{history?.provider}</p>
              {latest && latest.badges.length > 0 && (
                <div className="flex gap-1.5 mt-2">
                  {latest.badges.map((b) => <Badge key={b} name={b} />)}
                </div>
              )}
            </div>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
            <StatCard
              label="Current Score"
              value={latest ? formatAccuracy(latest.accuracy) : "--"}
              glow={latest && latest.accuracy >= 0.9 ? "gold" : "accent"}
            />
            <StatCard
              label="Evaluations"
              value={String(evalCount)}
            />
            <StatCard
              label="Scenarios"
              value={latest?.total_scenarios ? String(latest.total_scenarios) : "300"}
            />
            <StatCard
              label="Last Evaluated"
              value={latest ? formatDate(latest.completed_at) : "--"}
            />
          </div>

          {/* Charts grid */}
          <div className="grid md:grid-cols-2 gap-6 mb-8">
            {/* Accuracy history */}
            <div className="rounded-xl border border-border bg-bg-card p-6">
              <h3 className="font-semibold mb-4">Score History</h3>
              {history?.evaluations && (
                <AccuracyHistory evaluations={history.evaluations} />
              )}
            </div>

            {/* Category breakdown */}
            <div className="rounded-xl border border-border bg-bg-card p-6">
              <h3 className="font-semibold mb-4">Category Breakdown</h3>
              {latest?.categories ? (
                <CategoryBreakdown categories={latest.categories} />
              ) : (
                <div className="flex items-center justify-center h-48 text-text-muted text-sm">
                  No category data available
                </div>
              )}
            </div>
          </div>

          {/* Evaluation history table */}
          <div className="rounded-xl border border-border bg-bg-card overflow-hidden">
            <div className="p-4 border-b border-border">
              <h3 className="font-semibold">Evaluation History</h3>
            </div>
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border text-left text-xs text-text-muted uppercase tracking-wider">
                  <th className="py-3 px-4">Date</th>
                  <th className="py-3 px-4">Accuracy</th>
                  <th className="py-3 px-4">Correct</th>
                  <th className="py-3 px-4">Errors</th>
                  <th className="py-3 px-4">Badges</th>
                </tr>
              </thead>
              <tbody>
                {history?.evaluations?.map((ev) => (
                  <tr key={ev.eval_id} className="border-b border-border-subtle score-row">
                    <td className="py-3 px-4 text-text-secondary">{formatDate(ev.completed_at)}</td>
                    <td className="py-3 px-4 font-mono font-semibold">{formatAccuracy(ev.accuracy)}</td>
                    <td className="py-3 px-4 font-mono text-green">{ev.correct ?? "-"}</td>
                    <td className="py-3 px-4 font-mono text-red">{ev.errors ?? 0}</td>
                    <td className="py-3 px-4">
                      <div className="flex gap-1">
                        {ev.badges.map((b) => <Badge key={b} name={b} />)}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Link to CIRIS scoring */}
          <div className="mt-6 text-center">
            <a
              href="https://ciris.ai/ciris-scoring"
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm text-accent hover:text-accent-hover transition-colors"
            >
              Learn about CIRIS ethical scoring methodology &rarr;
            </a>
          </div>
        </>
      )}
    </div>
  );
}
