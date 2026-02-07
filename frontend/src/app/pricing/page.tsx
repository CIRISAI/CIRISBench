import { PricingCard } from "@/components/ui/PricingCard";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Pricing",
  description: "Ethical AI benchmarking plans. Free community tier, Pro for teams, Enterprise for regulated industries.",
};

const faqs = [
  {
    q: "What is the HE-300 benchmark?",
    a: "HE-300 is a 300-scenario ethical evaluation covering virtue ethics and commonsense moral reasoning. Each evaluation uses deterministic sampling for full reproducibility and produces a cryptographically-bound audit trace.",
  },
  {
    q: "How are models evaluated?",
    a: "Models receive ethical scenarios and their responses are scored using dual-method evaluation: heuristic classification plus semantic analysis. Results include per-category breakdowns, confidence scores, and full response capture.",
  },
  {
    q: "Can I keep my evaluations private?",
    a: "Yes. Pro and Enterprise evaluations are private by default. You can choose to publish results to the community leaderboard when you're ready, or keep them internal for compliance documentation.",
  },
  {
    q: "What about EU AI Act compliance?",
    a: "Enterprise plans include EU AI Act report generation that maps HE-300 results to the high-risk AI transparency and governance requirements under Articles 9-15. Reports are structured for regulatory submission.",
  },
  {
    q: "Do you offer custom scenario libraries?",
    a: "Enterprise customers can request industry-specific scenario sets (healthcare, finance, legal, HR) that augment the standard HE-300 with domain-relevant ethical dilemmas.",
  },
  {
    q: "Is this open source?",
    a: "CIRISBench is AGPL-3.0 licensed. The benchmark, evaluation engine, and scoring methodology are fully auditable. EthicsEngine.org is the managed platform that adds infrastructure, automation, and compliance tooling on top.",
  },
];

export default function PricingPage() {
  return (
    <div className="mx-auto max-w-7xl px-6 py-12">
      {/* Header */}
      <div className="text-center mb-12">
        <h1 className="text-3xl font-bold">Simple, transparent pricing</h1>
        <p className="text-text-secondary mt-2 max-w-xl mx-auto">
          From free community benchmarking to enterprise compliance automation.
          No hidden fees. Cancel anytime.
        </p>
      </div>

      {/* Pricing cards */}
      <div className="grid md:grid-cols-3 gap-6 max-w-5xl mx-auto mb-16">
        <PricingCard
          name="Community"
          price="Free"
          description="Explore ethical benchmarking"
          features={[
            "5 evaluations per month",
            "Public leaderboard listing",
            "Basic results report",
            "HE-300 standard (300 scenarios)",
            "API access (rate limited)",
          ]}
          cta="Start Free"
          ctaHref="/dashboard"
        />
        <PricingCard
          name="Pro"
          price="$399"
          period="mo"
          description="For teams shipping AI agents"
          features={[
            "Unlimited evaluations",
            "Private evaluations",
            "Full audit traces with crypto binding",
            "CI/CD integration (GitHub Actions)",
            "Historical trend tracking",
            "Export compliance reports (PDF)",
            "Email support",
          ]}
          cta="Start 14-Day Trial"
          ctaHref="/dashboard"
          highlighted
          badge="Most Popular"
        />
        <PricingCard
          name="Enterprise"
          price="Custom"
          description="Regulated industries & at-scale"
          features={[
            "Everything in Pro",
            "Custom scenario libraries",
            "EU AI Act report generation",
            "NIST AI RMF alignment mapping",
            "Dedicated eval infrastructure",
            "SSO/SAML + role-based access",
            "White-label reports",
            "Quarterly ethics advisory calls",
          ]}
          cta="Contact Sales"
          ctaHref="mailto:enterprise@ethicsengine.org"
        />
      </div>

      {/* Certification add-on */}
      <div className="max-w-3xl mx-auto mb-16">
        <div className="rounded-2xl border border-accent/30 bg-accent-subtle p-8 text-center">
          <div className="text-3xl mb-3">&#x1F396;&#xFE0F;</div>
          <h3 className="text-xl font-bold mb-2">Ethics Certification Badge</h3>
          <p className="text-text-secondary text-sm mb-4 max-w-lg mx-auto">
            Verified evaluation with published results. Cryptographically signed certificate.
            Listed on verified leaderboard. Re-certification required annually.
          </p>
          <div className="flex items-center justify-center gap-2 mb-4">
            <span className="text-2xl font-bold">$2,000</span>
            <span className="text-text-muted text-sm">per certification</span>
          </div>
          <p className="text-xs text-text-muted">
            Available as an add-on to any paid plan
          </p>
        </div>
      </div>

      {/* FAQ */}
      <div className="max-w-3xl mx-auto">
        <h2 className="text-2xl font-bold text-center mb-8">Frequently Asked Questions</h2>
        <div className="space-y-4">
          {faqs.map((faq) => (
            <div key={faq.q} className="rounded-xl border border-border bg-bg-card p-5">
              <h3 className="font-semibold mb-2">{faq.q}</h3>
              <p className="text-sm text-text-secondary leading-relaxed">{faq.a}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Bottom CTA */}
      <div className="text-center mt-16 mb-8">
        <p className="text-text-secondary mb-4">
          Questions? Need a custom plan for your organization?
        </p>
        <a
          href="mailto:hello@ethicsengine.org"
          className="inline-flex items-center px-6 py-3 rounded-lg border border-border hover:border-accent text-text font-medium transition-colors"
        >
          Talk to Us
        </a>
      </div>
    </div>
  );
}
