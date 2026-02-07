import Link from "next/link";
import type { FrontierScore } from "@/lib/api";
import { formatAccuracy, formatLatency, formatDate, trendIcon, trendColor, rankMedal, providerColor } from "@/lib/utils";
import { Badge } from "./Badge";
import { ScoreBar } from "./ScoreBar";

interface FrontierTableProps {
  scores: FrontierScore[];
  compact?: boolean;
}

export function FrontierTable({ scores, compact = false }: FrontierTableProps) {
  const sorted = [...scores].sort((a, b) => b.accuracy - a.accuracy);

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border text-left text-xs text-text-muted uppercase tracking-wider">
            <th className="py-3 px-4 w-12">#</th>
            <th className="py-3 px-4">Model</th>
            <th className="py-3 px-4 w-48">Score</th>
            {!compact && <th className="py-3 px-4">Badges</th>}
            {!compact && <th className="py-3 px-4">Latency</th>}
            <th className="py-3 px-4">Trend</th>
            {!compact && <th className="py-3 px-4">Evaluated</th>}
          </tr>
        </thead>
        <tbody>
          {sorted.map((score, i) => {
            const rank = i + 1;
            const medal = rankMedal(rank);
            return (
              <tr key={score.model_id} className="score-row border-b border-border-subtle transition-colors">
                <td className="py-3 px-4 font-mono text-text-muted">
                  {medal || rank}
                </td>
                <td className="py-3 px-4">
                  <Link href={`/scores/${encodeURIComponent(score.model_id)}`} className="group">
                    <div className="flex items-center gap-3">
                      <div
                        className="h-2.5 w-2.5 rounded-full shrink-0"
                        style={{ backgroundColor: providerColor(score.provider) }}
                      />
                      <div>
                        <p className="font-medium group-hover:text-accent transition-colors">
                          {score.display_name}
                        </p>
                        <p className="text-xs text-text-muted">{score.provider}</p>
                      </div>
                    </div>
                  </Link>
                </td>
                <td className="py-3 px-4">
                  <div className="flex items-center gap-3">
                    <span className="font-mono font-semibold w-14">
                      {formatAccuracy(score.accuracy)}
                    </span>
                    <ScoreBar value={score.accuracy} size="sm" showLabel={false} />
                  </div>
                </td>
                {!compact && (
                  <td className="py-3 px-4">
                    <div className="flex flex-wrap gap-1">
                      {score.badges.map((b) => <Badge key={b} name={b} />)}
                    </div>
                  </td>
                )}
                {!compact && (
                  <td className="py-3 px-4 font-mono text-text-secondary text-xs">
                    {formatLatency(score.avg_latency_ms)}
                  </td>
                )}
                <td className="py-3 px-4">
                  {score.trend && score.trend.direction && (
                    <span className={`font-mono text-xs ${trendColor(score.trend.direction)}`}>
                      {trendIcon(score.trend.direction)}
                      {score.trend.delta !== null && ` ${(score.trend.delta * 100).toFixed(1)}%`}
                    </span>
                  )}
                </td>
                {!compact && (
                  <td className="py-3 px-4 text-text-muted text-xs">
                    {formatDate(score.completed_at)}
                  </td>
                )}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
