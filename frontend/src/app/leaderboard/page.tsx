import { getLeaderboard } from "@/lib/api";
import { formatAccuracy, formatDate, rankMedal } from "@/lib/utils";
import { Badge } from "@/components/ui/Badge";
import { ScoreBar } from "@/components/ui/ScoreBar";
import Link from "next/link";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Community Leaderboard",
  description: "Public ethical benchmark rankings from the AI community. See how agents perform on the HE-300.",
};

export const revalidate = 30;

export default async function LeaderboardPage() {
  let leaderboard = null;
  let error = null;
  try {
    leaderboard = await getLeaderboard(100);
  } catch (e) {
    error = e instanceof Error ? e.message : "Failed to load leaderboard";
  }

  const entries = leaderboard?.entries ?? [];

  return (
    <div className="mx-auto max-w-7xl px-6 py-12">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold">Community Leaderboard</h1>
        <p className="text-text-secondary mt-2">
          Public HE-300 evaluations from the community. Publish your evaluation from the{" "}
          <Link href="/dashboard" className="text-accent hover:text-accent-hover">Dashboard</Link> to appear here.
        </p>
        {leaderboard?.updated_at && (
          <p className="text-xs text-text-muted mt-1">
            Last updated: {formatDate(leaderboard.updated_at)}
          </p>
        )}
      </div>

      {/* Table */}
      <div className="rounded-xl border border-border bg-bg-card overflow-hidden">
        {error ? (
          <div className="p-12 text-center text-text-muted">
            <p className="text-lg font-medium">Unable to load leaderboard</p>
            <p className="mt-2 text-sm">{error}</p>
          </div>
        ) : entries.length > 0 ? (
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border text-left text-xs text-text-muted uppercase tracking-wider">
                <th className="py-3 px-4 w-12">Rank</th>
                <th className="py-3 px-4">Agent / Model</th>
                <th className="py-3 px-4 w-48">Score</th>
                <th className="py-3 px-4">Badges</th>
                <th className="py-3 px-4">Evaluated</th>
              </tr>
            </thead>
            <tbody>
              {entries.map((entry) => {
                const medal = rankMedal(entry.rank);
                return (
                  <tr key={`${entry.rank}-${entry.target_model}`} className="border-b border-border-subtle score-row">
                    <td className="py-3 px-4 font-mono text-text-muted">
                      {medal || entry.rank}
                    </td>
                    <td className="py-3 px-4">
                      <div>
                        <p className="font-medium">
                          {entry.agent_name || "Anonymous Agent"}
                        </p>
                        {entry.target_model && (
                          <p className="text-xs text-text-muted">{entry.target_model}</p>
                        )}
                      </div>
                    </td>
                    <td className="py-3 px-4">
                      <div className="flex items-center gap-3">
                        <span className="font-mono font-semibold w-14">
                          {formatAccuracy(entry.accuracy)}
                        </span>
                        <ScoreBar value={entry.accuracy} size="sm" showLabel={false} />
                      </div>
                    </td>
                    <td className="py-3 px-4">
                      <div className="flex flex-wrap gap-1">
                        {entry.badges.map((b) => <Badge key={b} name={b} />)}
                      </div>
                    </td>
                    <td className="py-3 px-4 text-text-muted text-xs">
                      {formatDate(entry.completed_at)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        ) : (
          <div className="p-12 text-center text-text-muted">
            <p className="text-lg font-medium">No public evaluations yet</p>
            <p className="mt-2 text-sm">
              Be the first!{" "}
              <Link href="/pricing" className="text-accent hover:text-accent-hover">
                Run a benchmark
              </Link>{" "}
              and publish your results.
            </p>
          </div>
        )}
      </div>

      {/* CTA */}
      <div className="mt-8 text-center">
        <p className="text-sm text-text-secondary mb-3">
          Want your agent on the leaderboard?
        </p>
        <Link
          href="/pricing"
          className="inline-flex items-center px-5 py-2.5 rounded-lg bg-accent hover:bg-accent-hover text-white text-sm font-medium transition-colors"
        >
          Start Benchmarking
        </Link>
      </div>
    </div>
  );
}
