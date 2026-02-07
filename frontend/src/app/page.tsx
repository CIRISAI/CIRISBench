import Link from "next/link";
import { getFrontierScores } from "@/lib/api";
import { FrontierTable } from "@/components/ui/FrontierTable";
import { StatCard } from "@/components/ui/StatCard";
import { PricingCard } from "@/components/ui/PricingCard";
import { formatAccuracy } from "@/lib/utils";

export const revalidate = 120; // ISR every 2 min

export default async function Home() {
  let scores = null;
  try {
    scores = await getFrontierScores();
  } catch {
    // API not available — show placeholder
  }

  const topModel = scores?.scores?.[0];
  const modelCount = scores?.scores?.length ?? 0;

  return (
    <div>
      {/* Hero */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-b from-accent/5 via-transparent to-transparent pointer-events-none" />
        <div className="mx-auto max-w-7xl px-6 pt-20 pb-16 text-center relative">
          <h1 className="text-5xl sm:text-6xl font-bold tracking-tight leading-tight">
            How ethical is
            <br />
            <span className="gradient-text">your AI?</span>
          </h1>
          <p className="mt-6 text-lg text-text-secondary max-w-2xl mx-auto leading-relaxed">
            Independent ethical benchmarking for AI systems.
            300 scenarios. Cryptographic audit trails.
            The HE-300 standard.
          </p>
          <div className="mt-8 flex items-center justify-center gap-4">
            <Link
              href="/scores"
              className="px-6 py-3 rounded-lg bg-accent hover:bg-accent-hover text-white font-medium transition-colors"
            >
              View Frontier Scores
            </Link>
            <Link
              href="/pricing"
              className="px-6 py-3 rounded-lg border border-border hover:border-text-muted text-text font-medium transition-colors"
            >
              Benchmark Your Agent
            </Link>
          </div>
        </div>
      </section>

      {/* Stats bar */}
      <section className="mx-auto max-w-7xl px-6 -mt-4">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <StatCard label="Frontier Models" value={String(modelCount || 15)} sub="Evaluated weekly" />
          <StatCard label="Scenarios" value="300" sub="Per evaluation" glow="accent" />
          <StatCard
            label="Top Score"
            value={topModel ? formatAccuracy(topModel.accuracy) : "--"}
            sub={topModel?.display_name || "Run pending"}
            glow="gold"
          />
          <StatCard label="Categories" value="2" sub="Virtue + Commonsense" />
        </div>
      </section>

      {/* Frontier scores preview */}
      <section className="mx-auto max-w-7xl px-6 mt-16">
        <div className="flex items-end justify-between mb-6">
          <div>
            <h2 className="text-2xl font-bold">Frontier Model Scores</h2>
            <p className="text-sm text-text-secondary mt-1">
              Latest HE-300 benchmark results for major AI models
            </p>
          </div>
          <Link href="/scores" className="text-sm text-accent hover:text-accent-hover transition-colors">
            View all &rarr;
          </Link>
        </div>

        <div className="rounded-xl border border-border bg-bg-card overflow-hidden">
          {scores ? (
            <FrontierTable scores={scores.scores.slice(0, 10)} compact />
          ) : (
            <div className="p-12 text-center text-text-muted">
              <p className="text-lg font-medium">First frontier sweep pending</p>
              <p className="mt-2 text-sm">Scores will appear here after the initial evaluation run.</p>
            </div>
          )}
        </div>
      </section>

      {/* How it works */}
      <section className="mx-auto max-w-7xl px-6 mt-20">
        <h2 className="text-2xl font-bold text-center mb-10">How It Works</h2>
        <div className="grid md:grid-cols-3 gap-6">
          {[
            {
              step: "01",
              title: "300 Ethical Scenarios",
              desc: "The HE-300 benchmark tests AI across virtue ethics and hard commonsense moral reasoning with deterministic, reproducible sampling.",
            },
            {
              step: "02",
              title: "Automated Evaluation",
              desc: "Your model is evaluated against the CIRIS Framework with dual-method scoring (heuristic + semantic) and full response capture.",
            },
            {
              step: "03",
              title: "Auditable Results",
              desc: "Every evaluation produces a cryptographically-bound trace ID, per-category breakdowns, and compliance-ready documentation.",
            },
          ].map((item) => (
            <div key={item.step} className="rounded-xl border border-border bg-bg-card p-6">
              <div className="text-accent font-mono text-sm font-bold mb-3">{item.step}</div>
              <h3 className="font-semibold text-lg mb-2">{item.title}</h3>
              <p className="text-sm text-text-secondary leading-relaxed">{item.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Pricing preview */}
      <section className="mx-auto max-w-7xl px-6 mt-20 mb-20">
        <h2 className="text-2xl font-bold text-center">Benchmark Your AI</h2>
        <p className="text-center text-text-secondary mt-2 mb-10">
          From free community access to enterprise compliance
        </p>
        <div className="grid md:grid-cols-3 gap-6 max-w-4xl mx-auto">
          <PricingCard
            name="Community"
            price="Free"
            description="Get started with ethical benchmarking"
            features={[
              "5 evaluations per month",
              "Public leaderboard listing",
              "Basic results report",
              "HE-300 standard benchmark",
            ]}
            cta="Start Free"
            ctaHref="/pricing"
          />
          <PricingCard
            name="Pro"
            price="$399"
            period="mo"
            description="For teams shipping AI agents"
            features={[
              "Unlimited evaluations",
              "Private evaluations",
              "Full audit traces",
              "CI/CD integration",
              "Historical trend tracking",
              "Export compliance reports",
            ]}
            cta="Start Pro Trial"
            ctaHref="/pricing"
            highlighted
            badge="Most Popular"
          />
          <PricingCard
            name="Enterprise"
            price="Custom"
            description="Regulated industries"
            features={[
              "Everything in Pro",
              "Custom scenario libraries",
              "EU AI Act report generation",
              "Dedicated infrastructure",
              "SSO/SAML + audit logs",
              "Quarterly ethics advisory",
            ]}
            cta="Contact Sales"
            ctaHref="mailto:enterprise@ethicsengine.org"
          />
        </div>
      </section>
    </div>
  );
}
