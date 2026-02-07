import { cn } from "@/lib/utils";
import Link from "next/link";

interface PricingCardProps {
  name: string;
  price: string;
  period?: string;
  description: string;
  features: string[];
  cta: string;
  ctaHref: string;
  highlighted?: boolean;
  badge?: string;
}

export function PricingCard({
  name, price, period, description, features, cta, ctaHref, highlighted, badge,
}: PricingCardProps) {
  return (
    <div className={cn(
      "relative rounded-2xl border p-6 flex flex-col transition-all",
      highlighted
        ? "border-accent bg-accent-subtle glow-accent"
        : "border-border bg-bg-card hover:border-border/60"
    )}>
      {badge && (
        <div className="absolute -top-3 left-1/2 -translate-x-1/2">
          <span className="bg-accent text-white text-xs font-semibold px-3 py-1 rounded-full">
            {badge}
          </span>
        </div>
      )}

      <div className="mb-5">
        <h3 className="text-lg font-semibold">{name}</h3>
        <p className="text-sm text-text-secondary mt-1">{description}</p>
      </div>

      <div className="mb-6">
        <span className="text-3xl font-bold">{price}</span>
        {period && <span className="text-text-muted text-sm">/{period}</span>}
      </div>

      <ul className="space-y-2.5 mb-8 flex-1">
        {features.map((f) => (
          <li key={f} className="flex items-start gap-2 text-sm">
            <svg className="w-4 h-4 text-green mt-0.5 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
            </svg>
            <span className="text-text-secondary">{f}</span>
          </li>
        ))}
      </ul>

      <Link
        href={ctaHref}
        className={cn(
          "block w-full text-center py-2.5 rounded-lg text-sm font-medium transition-colors",
          highlighted
            ? "bg-accent hover:bg-accent-hover text-white"
            : "bg-bg-elevated hover:bg-bg-card-hover text-text border border-border"
        )}
      >
        {cta}
      </Link>
    </div>
  );
}
