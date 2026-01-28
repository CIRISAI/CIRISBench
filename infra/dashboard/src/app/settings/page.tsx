import { prisma } from "@/lib/prisma";
import Link from "next/link";

export default async function SettingsPage() {
  // Get model configurations from database
  const modelConfigs = await prisma.modelConfig.findMany({
    orderBy: { name: "asc" },
  });

  return (
    <div className="min-h-screen bg-background p-8">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <Link
            href="/"
            className="text-muted-foreground hover:text-foreground text-sm mb-2 inline-block"
          >
            ← Back to Dashboard
          </Link>
          <h1 className="text-3xl font-bold">Settings</h1>
          <p className="text-muted-foreground mt-2">
            Configure model settings and benchmark parameters
          </p>
        </div>

        {/* Model Configurations */}
        <div className="bg-card border border-border rounded-lg p-6 mb-8">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold">Model Configurations</h2>
            <button className="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors">
              + Add Model
            </button>
          </div>

          {modelConfigs.length > 0 ? (
            <div className="space-y-4">
              {modelConfigs.map((config) => (
                <div
                  key={config.id}
                  className="flex items-center justify-between p-4 bg-muted/50 rounded-lg"
                >
                  <div>
                    <p className="font-semibold">{config.name}</p>
                    <p className="text-sm text-muted-foreground">
                      {config.provider} • {config.quantization || "Full precision"}
                    </p>
                  </div>
                  <div className="flex items-center gap-2">
                    <span
                      className={`px-2 py-1 rounded text-xs font-medium ${
                        config.isActive
                          ? "bg-green-500/20 text-green-500"
                          : "bg-muted text-muted-foreground"
                      }`}
                    >
                      {config.isActive ? "Active" : "Inactive"}
                    </span>
                    <button className="p-2 hover:bg-muted rounded-lg transition-colors">
                      ⚙️
                    </button>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8 text-muted-foreground">
              <p>No model configurations found.</p>
              <p className="text-sm mt-1">
                Add a model configuration to start running benchmarks.
              </p>
            </div>
          )}
        </div>

        {/* Default Benchmark Settings */}
        <div className="bg-card border border-border rounded-lg p-6 mb-8">
          <h2 className="text-lg font-semibold mb-4">Default Benchmark Settings</h2>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2">
                Default Sample Size
              </label>
              <input
                type="number"
                defaultValue={300}
                min={1}
                max={1000}
                className="w-full px-4 py-2 bg-background border border-border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary"
              />
              <p className="text-xs text-muted-foreground mt-1">
                Number of scenarios to run per benchmark
              </p>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">
                Categories
              </label>
              <div className="flex flex-wrap gap-2">
                {[
                  "commonsense",
                  "deontology",
                  "justice",
                  "utilitarianism",
                  "virtue",
                ].map((category) => (
                  <label
                    key={category}
                    className="flex items-center gap-2 px-3 py-2 bg-muted/50 rounded-lg cursor-pointer hover:bg-muted transition-colors"
                  >
                    <input type="checkbox" defaultChecked className="rounded" />
                    <span className="capitalize">{category}</span>
                  </label>
                ))}
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">
                Timeout per Scenario (seconds)
              </label>
              <input
                type="number"
                defaultValue={30}
                min={5}
                max={300}
                className="w-full px-4 py-2 bg-background border border-border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary"
              />
            </div>
          </div>
        </div>

        {/* Infrastructure Settings */}
        <div className="bg-card border border-border rounded-lg p-6 mb-8">
          <h2 className="text-lg font-semibold mb-4">Infrastructure</h2>
          <div className="space-y-4">
            <div className="flex items-center justify-between p-4 bg-muted/50 rounded-lg">
              <div>
                <p className="font-medium">Vault Status</p>
                <p className="text-sm text-muted-foreground">
                  HashiCorp Vault connection
                </p>
              </div>
              <span className="px-2 py-1 rounded bg-green-500/20 text-green-500 text-sm font-medium">
                Connected
              </span>
            </div>

            <div className="flex items-center justify-between p-4 bg-muted/50 rounded-lg">
              <div>
                <p className="font-medium">GPU Runner</p>
                <p className="text-sm text-muted-foreground">
                  A10 24GB • Lambda Labs
                </p>
              </div>
              <span className="px-2 py-1 rounded bg-green-500/20 text-green-500 text-sm font-medium">
                Available
              </span>
            </div>

            <div className="flex items-center justify-between p-4 bg-muted/50 rounded-lg">
              <div>
                <p className="font-medium">S3 Artifact Storage</p>
                <p className="text-sm text-muted-foreground">
                  he300-artifacts bucket
                </p>
              </div>
              <span className="px-2 py-1 rounded bg-green-500/20 text-green-500 text-sm font-medium">
                Connected
              </span>
            </div>
          </div>
        </div>

        {/* API Configuration */}
        <div className="bg-card border border-border rounded-lg p-6">
          <h2 className="text-lg font-semibold mb-4">API Configuration</h2>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2">
                Webhook Secret
              </label>
              <div className="flex gap-2">
                <input
                  type="password"
                  defaultValue="••••••••••••••••"
                  readOnly
                  className="flex-1 px-4 py-2 bg-background border border-border rounded-lg font-mono"
                />
                <button className="px-4 py-2 bg-muted hover:bg-muted/80 rounded-lg transition-colors">
                  Regenerate
                </button>
              </div>
              <p className="text-xs text-muted-foreground mt-1">
                Used to authenticate webhook requests from CI/CD
              </p>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">
                Webhook Endpoint
              </label>
              <div className="flex gap-2">
                <input
                  type="text"
                  value={`${process.env.NEXT_PUBLIC_APP_URL || "https://he300-dashboard.example.com"}/api/webhook`}
                  readOnly
                  className="flex-1 px-4 py-2 bg-background border border-border rounded-lg font-mono text-sm"
                />
                <button className="px-4 py-2 bg-muted hover:bg-muted/80 rounded-lg transition-colors">
                  Copy
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Save Button */}
        <div className="mt-8 flex justify-end">
          <button className="px-6 py-3 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors font-medium">
            Save Changes
          </button>
        </div>
      </div>
    </div>
  );
}
