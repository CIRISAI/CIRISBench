import Link from "next/link";
import { prisma } from "@/lib/prisma";
import { formatDistanceToNow } from "date-fns";
import { formatPercentage, formatDuration, getStatusColor } from "@/lib/utils";

async function getLatestRuns() {
  return prisma.benchmarkRun.findMany({
    take: 5,
    orderBy: { createdAt: "desc" },
    include: {
      results: true,
    },
  });
}

async function getStats() {
  const [totalRuns, completedRuns, avgAccuracy] = await Promise.all([
    prisma.benchmarkRun.count(),
    prisma.benchmarkRun.count({ where: { status: "COMPLETED" } }),
    prisma.categoryResult.aggregate({
      _avg: { accuracy: true },
    }),
  ]);

  return {
    totalRuns,
    completedRuns,
    avgAccuracy: avgAccuracy._avg.accuracy || 0,
  };
}

export default async function Home() {
  const [latestRuns, stats] = await Promise.all([
    getLatestRuns(),
    getStats(),
  ]);

  return (
    <main className="container mx-auto py-8 px-4">
      <header className="mb-8">
        <h1 className="text-4xl font-bold mb-2">HE-300 Benchmark Dashboard</h1>
        <p className="text-muted-foreground">
          Track and analyze ethical reasoning benchmark results
        </p>
      </header>

      {/* Stats Overview */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <div className="bg-card rounded-lg border p-6">
          <h3 className="text-sm font-medium text-muted-foreground">Total Runs</h3>
          <p className="text-3xl font-bold mt-2">{stats.totalRuns}</p>
        </div>
        <div className="bg-card rounded-lg border p-6">
          <h3 className="text-sm font-medium text-muted-foreground">Completed</h3>
          <p className="text-3xl font-bold mt-2 text-green-600">{stats.completedRuns}</p>
        </div>
        <div className="bg-card rounded-lg border p-6">
          <h3 className="text-sm font-medium text-muted-foreground">Avg Accuracy</h3>
          <p className="text-3xl font-bold mt-2">{formatPercentage(stats.avgAccuracy)}</p>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex gap-4 mb-8">
        <Link
          href="/dashboard"
          className="px-4 py-2 bg-primary text-primary-foreground rounded-md hover:opacity-90"
        >
          View Dashboard
        </Link>
        <Link
          href="/dashboard/runs"
          className="px-4 py-2 bg-secondary text-secondary-foreground rounded-md hover:opacity-90"
        >
          All Runs
        </Link>
        <Link
          href="/dashboard/compare"
          className="px-4 py-2 bg-secondary text-secondary-foreground rounded-md hover:opacity-90"
        >
          Compare Models
        </Link>
      </nav>

      {/* Latest Runs */}
      <section>
        <h2 className="text-2xl font-semibold mb-4">Latest Runs</h2>
        {latestRuns.length === 0 ? (
          <div className="bg-card rounded-lg border p-8 text-center">
            <p className="text-muted-foreground">No benchmark runs yet.</p>
            <p className="text-sm mt-2">
              Run a benchmark to see results here.
            </p>
          </div>
        ) : (
          <div className="grid gap-4">
            {latestRuns.map((run) => {
              const totalCorrect = run.results.reduce((acc, r) => acc + r.correct, 0);
              const totalScenarios = run.results.reduce((acc, r) => acc + r.total, 0);
              const overallAccuracy = totalScenarios > 0 ? totalCorrect / totalScenarios : 0;

              return (
                <Link key={run.id} href={`/dashboard/runs/${run.id}`}>
                  <div className="bg-card rounded-lg border p-6 hover:shadow-md transition-shadow">
                    <div className="flex justify-between items-start">
                      <div>
                        <h3 className="font-semibold text-lg">{run.model}</h3>
                        <p className="text-sm text-muted-foreground">
                          {formatDistanceToNow(run.createdAt, { addSuffix: true })}
                        </p>
                      </div>
                      <span
                        className={`px-2 py-1 rounded text-xs text-white ${getStatusColor(run.status)}`}
                      >
                        {run.status}
                      </span>
                    </div>

                    {run.status === "COMPLETED" && (
                      <div className="mt-4">
                        <div className="grid grid-cols-5 gap-2 text-center text-sm">
                          {run.results.map((result) => (
                            <div key={result.category}>
                              <div className="text-muted-foreground capitalize">
                                {result.category}
                              </div>
                              <div className="font-semibold">
                                {formatPercentage(result.accuracy)}
                              </div>
                            </div>
                          ))}
                        </div>
                        <div className="mt-4 flex justify-between items-center border-t pt-4">
                          <div>
                            <span className="text-2xl font-bold">
                              {formatPercentage(overallAccuracy)}
                            </span>
                            <span className="text-muted-foreground ml-2">overall</span>
                          </div>
                          {run.duration && (
                            <span className="text-sm text-muted-foreground">
                              {formatDuration(run.duration)}
                            </span>
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                </Link>
              );
            })}
          </div>
        )}
      </section>
    </main>
  );
}
