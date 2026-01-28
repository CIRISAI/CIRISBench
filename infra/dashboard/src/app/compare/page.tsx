import { prisma } from "@/lib/prisma";
import { formatPercentage } from "@/lib/utils";
import Link from "next/link";

interface SearchParams {
  models?: string;
}

export default async function ComparePage({
  searchParams,
}: {
  searchParams: SearchParams;
}) {
  // Parse model IDs from query string
  const modelIds = searchParams.models?.split(",").filter(Boolean) || [];

  // Get all available models for selection
  const availableModels = await prisma.benchmarkRun.findMany({
    where: { status: "completed" },
    select: { model: true },
    distinct: ["model"],
    orderBy: { model: "asc" },
  });

  // Get comparison data for selected models
  const comparisonData = modelIds.length > 0 
    ? await Promise.all(
        modelIds.map(async (model) => {
          const latestRun = await prisma.benchmarkRun.findFirst({
            where: { model, status: "completed" },
            include: { results: true },
            orderBy: { createdAt: "desc" },
          });
          return { model, run: latestRun };
        })
      )
    : [];

  // Get all unique categories
  const allCategories = [
    ...new Set(
      comparisonData.flatMap((d) => d.run?.results.map((r) => r.category) || [])
    ),
  ].sort();

  return (
    <div className="min-h-screen bg-background p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <Link
            href="/"
            className="text-muted-foreground hover:text-foreground text-sm mb-2 inline-block"
          >
            ← Back to Dashboard
          </Link>
          <h1 className="text-3xl font-bold">Model Comparison</h1>
          <p className="text-muted-foreground mt-2">
            Compare benchmark results across different models
          </p>
        </div>

        {/* Model Selection */}
        <div className="bg-card border border-border rounded-lg p-6 mb-8">
          <h2 className="text-lg font-semibold mb-4">Select Models to Compare</h2>
          <form className="flex flex-wrap gap-4">
            {availableModels.map(({ model }) => (
              <label
                key={model}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg border cursor-pointer transition-colors ${
                  modelIds.includes(model)
                    ? "border-primary bg-primary/10"
                    : "border-border hover:border-primary/50"
                }`}
              >
                <input
                  type="checkbox"
                  name="models"
                  value={model}
                  defaultChecked={modelIds.includes(model)}
                  className="sr-only"
                  onChange={(e) => {
                    const form = e.target.form;
                    if (form) {
                      const checkboxes = form.querySelectorAll(
                        'input[name="models"]:checked'
                      ) as NodeListOf<HTMLInputElement>;
                      const selectedModels = Array.from(checkboxes).map(
                        (cb) => cb.value
                      );
                      window.location.href = `/compare?models=${selectedModels.join(",")}`;
                    }
                  }}
                />
                <span className="font-medium">{model}</span>
              </label>
            ))}
          </form>
          {availableModels.length === 0 && (
            <p className="text-muted-foreground">
              No completed benchmark runs available for comparison.
            </p>
          )}
        </div>

        {/* Comparison Table */}
        {comparisonData.length > 0 && (
          <div className="bg-card border border-border rounded-lg p-6">
            <h2 className="text-lg font-semibold mb-4">Results Comparison</h2>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-border">
                    <th className="text-left py-3 px-4 font-medium">Category</th>
                    {comparisonData.map((d) => (
                      <th
                        key={d.model}
                        className="text-right py-3 px-4 font-medium"
                      >
                        {d.model}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {allCategories.map((category) => {
                    const accuracies = comparisonData.map((d) => {
                      const result = d.run?.results.find(
                        (r) => r.category === category
                      );
                      return result?.accuracy ?? null;
                    });
                    const maxAccuracy = Math.max(
                      ...accuracies.filter((a): a is number => a !== null)
                    );

                    return (
                      <tr
                        key={category}
                        className="border-b border-border/50 hover:bg-muted/50"
                      >
                        <td className="py-3 px-4 font-medium capitalize">
                          {category.replace(/_/g, " ")}
                        </td>
                        {accuracies.map((accuracy, i) => (
                          <td
                            key={i}
                            className={`py-3 px-4 text-right font-semibold ${
                              accuracy === maxAccuracy && accuracy !== null
                                ? "text-green-500"
                                : ""
                            }`}
                          >
                            {accuracy !== null ? formatPercentage(accuracy) : "—"}
                          </td>
                        ))}
                      </tr>
                    );
                  })}
                  {/* Overall Row */}
                  <tr className="bg-muted/30 font-bold">
                    <td className="py-3 px-4">Overall</td>
                    {comparisonData.map((d) => {
                      const results = d.run?.results || [];
                      const overall =
                        results.length > 0
                          ? results.reduce((acc, r) => acc + r.correct, 0) /
                            results.reduce((acc, r) => acc + r.total, 0)
                          : null;
                      return (
                        <td key={d.model} className="py-3 px-4 text-right">
                          {overall !== null ? formatPercentage(overall) : "—"}
                        </td>
                      );
                    })}
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Radar Chart Placeholder */}
        {comparisonData.length >= 2 && (
          <div className="bg-card border border-border rounded-lg p-6 mt-8">
            <h2 className="text-lg font-semibold mb-4">Visual Comparison</h2>
            <div className="aspect-square max-w-xl mx-auto flex items-center justify-center bg-muted/30 rounded-lg">
              <p className="text-muted-foreground">
                Radar chart visualization will be rendered here
              </p>
            </div>
          </div>
        )}

        {modelIds.length === 0 && availableModels.length > 0 && (
          <div className="text-center py-12 text-muted-foreground">
            <p className="text-lg mb-2">Select models above to compare their results</p>
            <p className="text-sm">
              Choose at least 2 models for a meaningful comparison
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
