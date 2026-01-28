import { prisma } from "@/lib/prisma";
import { notFound } from "next/navigation";
import { formatDuration, formatPercentage, getStatusColor } from "@/lib/utils";
import Link from "next/link";

interface RunPageProps {
  params: { id: string };
}

export default async function RunPage({ params }: RunPageProps) {
  const run = await prisma.benchmarkRun.findUnique({
    where: { id: params.id },
    include: {
      results: {
        orderBy: { category: "asc" },
      },
      artifacts: {
        orderBy: { createdAt: "desc" },
      },
    },
  });

  if (!run) {
    notFound();
  }

  const overallAccuracy =
    run.results.length > 0
      ? run.results.reduce((acc, r) => acc + r.correct, 0) /
        run.results.reduce((acc, r) => acc + r.total, 0)
      : null;

  return (
    <div className="min-h-screen bg-background p-8">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <Link
              href="/"
              className="text-muted-foreground hover:text-foreground text-sm mb-2 inline-block"
            >
              ← Back to Dashboard
            </Link>
            <h1 className="text-3xl font-bold">Benchmark Run Details</h1>
            <p className="text-muted-foreground mt-1 font-mono text-sm">
              {run.id}
            </p>
          </div>
          <span
            className={`px-3 py-1.5 rounded-full text-sm font-medium ${getStatusColor(
              run.status
            )}`}
          >
            {run.status}
          </span>
        </div>

        {/* Run Metadata */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
          <MetadataCard label="Model" value={run.model} />
          <MetadataCard
            label="Quantization"
            value={run.quantization || "None"}
          />
          <MetadataCard label="Sample Size" value={run.sampleSize.toString()} />
          <MetadataCard
            label="Duration"
            value={run.duration ? formatDuration(run.duration) : "—"}
          />
          <MetadataCard label="Environment" value={run.environment} />
          <MetadataCard label="GPU" value={run.gpuType || "Unknown"} />
          <MetadataCard label="Branch" value={run.branch || "main"} />
          <MetadataCard
            label="Commit"
            value={run.commitSha?.substring(0, 7) || "—"}
          />
        </div>

        {/* Overall Score */}
        {overallAccuracy !== null && (
          <div className="bg-card border border-border rounded-lg p-6 mb-8">
            <h2 className="text-lg font-semibold mb-4">Overall Score</h2>
            <div className="flex items-center gap-8">
              <div className="text-5xl font-bold text-primary">
                {formatPercentage(overallAccuracy)}
              </div>
              <div className="flex-1">
                <div className="h-4 bg-muted rounded-full overflow-hidden">
                  <div
                    className="h-full bg-primary transition-all duration-500"
                    style={{ width: `${overallAccuracy * 100}%` }}
                  />
                </div>
                <p className="text-sm text-muted-foreground mt-2">
                  {run.results.reduce((acc, r) => acc + r.correct, 0)} /{" "}
                  {run.results.reduce((acc, r) => acc + r.total, 0)} scenarios
                  correct
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Category Results */}
        <div className="bg-card border border-border rounded-lg p-6 mb-8">
          <h2 className="text-lg font-semibold mb-4">Results by Category</h2>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left py-3 px-4 font-medium">Category</th>
                  <th className="text-right py-3 px-4 font-medium">Correct</th>
                  <th className="text-right py-3 px-4 font-medium">Total</th>
                  <th className="text-right py-3 px-4 font-medium">Accuracy</th>
                  <th className="text-left py-3 px-4 font-medium w-1/3">
                    Progress
                  </th>
                </tr>
              </thead>
              <tbody>
                {run.results.map((result) => (
                  <tr
                    key={result.id}
                    className="border-b border-border/50 hover:bg-muted/50"
                  >
                    <td className="py-3 px-4 font-medium capitalize">
                      {result.category.replace(/_/g, " ")}
                    </td>
                    <td className="py-3 px-4 text-right text-green-500">
                      {result.correct}
                    </td>
                    <td className="py-3 px-4 text-right">{result.total}</td>
                    <td className="py-3 px-4 text-right font-semibold">
                      {formatPercentage(result.accuracy)}
                    </td>
                    <td className="py-3 px-4">
                      <div className="h-2 bg-muted rounded-full overflow-hidden">
                        <div
                          className={`h-full transition-all duration-500 ${
                            result.accuracy >= 0.9
                              ? "bg-green-500"
                              : result.accuracy >= 0.7
                              ? "bg-yellow-500"
                              : "bg-red-500"
                          }`}
                          style={{ width: `${result.accuracy * 100}%` }}
                        />
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Artifacts */}
        {run.artifacts.length > 0 && (
          <div className="bg-card border border-border rounded-lg p-6">
            <h2 className="text-lg font-semibold mb-4">Artifacts</h2>
            <div className="space-y-2">
              {run.artifacts.map((artifact) => (
                <a
                  key={artifact.id}
                  href={artifact.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center justify-between p-3 bg-muted/50 rounded-lg hover:bg-muted transition-colors"
                >
                  <div className="flex items-center gap-3">
                    <span className="text-2xl">
                      {artifact.type === "log"
                        ? "📄"
                        : artifact.type === "report"
                        ? "📊"
                        : artifact.type === "model_output"
                        ? "🤖"
                        : "📁"}
                    </span>
                    <div>
                      <p className="font-medium">{artifact.name}</p>
                      <p className="text-sm text-muted-foreground">
                        {artifact.type} •{" "}
                        {(artifact.size / 1024).toFixed(1)} KB
                      </p>
                    </div>
                  </div>
                  <span className="text-muted-foreground">↗</span>
                </a>
              ))}
            </div>
          </div>
        )}

        {/* Timestamps */}
        <div className="mt-8 text-sm text-muted-foreground">
          <p>Started: {run.createdAt.toLocaleString()}</p>
          {run.completedAt && (
            <p>Completed: {run.completedAt.toLocaleString()}</p>
          )}
        </div>
      </div>
    </div>
  );
}

function MetadataCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-card border border-border rounded-lg p-4">
      <p className="text-sm text-muted-foreground mb-1">{label}</p>
      <p className="font-semibold truncate" title={value}>
        {value}
      </p>
    </div>
  );
}
