import Link from "next/link";
import { formatDuration, formatPercentage, getStatusColor } from "@/lib/utils";

interface BenchmarkRun {
  id: string;
  model: string;
  quantization: string | null;
  sampleSize: number;
  status: string;
  environment: string;
  gpuType: string | null;
  duration: number | null;
  createdAt: Date;
  results: Array<{
    correct: number;
    total: number;
  }>;
}

interface RunTableProps {
  runs: BenchmarkRun[];
}

export function RunTable({ runs }: RunTableProps) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full">
        <thead>
          <tr className="border-b border-border">
            <th className="text-left py-3 px-4 font-medium">Model</th>
            <th className="text-left py-3 px-4 font-medium">Quantization</th>
            <th className="text-right py-3 px-4 font-medium">Sample Size</th>
            <th className="text-right py-3 px-4 font-medium">Accuracy</th>
            <th className="text-left py-3 px-4 font-medium">Environment</th>
            <th className="text-left py-3 px-4 font-medium">Duration</th>
            <th className="text-left py-3 px-4 font-medium">Status</th>
            <th className="text-left py-3 px-4 font-medium">Date</th>
            <th className="text-left py-3 px-4 font-medium"></th>
          </tr>
        </thead>
        <tbody>
          {runs.map((run) => {
            const totalCorrect = run.results.reduce((acc, r) => acc + r.correct, 0);
            const totalScenarios = run.results.reduce((acc, r) => acc + r.total, 0);
            const accuracy = totalScenarios > 0 ? totalCorrect / totalScenarios : null;

            return (
              <tr
                key={run.id}
                className="border-b border-border/50 hover:bg-muted/50 transition-colors"
              >
                <td className="py-3 px-4">
                  <span className="font-medium">{run.model}</span>
                </td>
                <td className="py-3 px-4 text-muted-foreground">
                  {run.quantization || "—"}
                </td>
                <td className="py-3 px-4 text-right">{run.sampleSize}</td>
                <td className="py-3 px-4 text-right">
                  {accuracy !== null ? (
                    <span
                      className={`font-semibold ${
                        accuracy >= 0.9
                          ? "text-green-500"
                          : accuracy >= 0.7
                          ? "text-yellow-500"
                          : "text-red-500"
                      }`}
                    >
                      {formatPercentage(accuracy)}
                    </span>
                  ) : (
                    "—"
                  )}
                </td>
                <td className="py-3 px-4">
                  <span className="inline-flex items-center gap-1">
                    {run.gpuType && (
                      <span className="text-xs bg-muted px-2 py-0.5 rounded">
                        {run.gpuType}
                      </span>
                    )}
                    <span className="text-muted-foreground">{run.environment}</span>
                  </span>
                </td>
                <td className="py-3 px-4 text-muted-foreground">
                  {run.duration ? formatDuration(run.duration) : "—"}
                </td>
                <td className="py-3 px-4">
                  <span
                    className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(
                      run.status
                    )}`}
                  >
                    {run.status}
                  </span>
                </td>
                <td className="py-3 px-4 text-muted-foreground text-sm">
                  {run.createdAt.toLocaleDateString()}
                </td>
                <td className="py-3 px-4">
                  <Link
                    href={`/runs/${run.id}`}
                    className="text-primary hover:underline text-sm"
                  >
                    View →
                  </Link>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>

      {runs.length === 0 && (
        <div className="text-center py-12 text-muted-foreground">
          <p>No benchmark runs found.</p>
        </div>
      )}
    </div>
  );
}
