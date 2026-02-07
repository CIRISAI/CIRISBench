"use client";

import type { CategoryStats } from "@/lib/api";
import { ScoreBar } from "@/components/ui/ScoreBar";

interface Props {
  categories: Record<string, CategoryStats>;
}

const categoryLabels: Record<string, string> = {
  virtue: "Virtue Ethics",
  commonsense_hard: "Commonsense (Hard)",
  commonsense: "Commonsense",
  deontology: "Deontology",
  justice: "Justice",
};

export function CategoryBreakdown({ categories }: Props) {
  const entries = Object.entries(categories).sort(
    ([, a], [, b]) => b.accuracy - a.accuracy,
  );

  return (
    <div className="space-y-3">
      {entries.map(([cat, stats]) => (
        <div key={cat}>
          <div className="flex justify-between mb-1">
            <span className="text-xs font-medium text-text-secondary">
              {categoryLabels[cat] || cat}
            </span>
            <span className="text-xs font-mono text-text-muted">
              {stats.correct}/{stats.total}
            </span>
          </div>
          <ScoreBar value={stats.accuracy} size="md" />
        </div>
      ))}
    </div>
  );
}
