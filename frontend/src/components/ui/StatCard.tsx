import { cn } from "@/lib/utils";

interface StatCardProps {
  label: string;
  value: string;
  sub?: string;
  glow?: "green" | "accent" | "gold";
}

export function StatCard({ label, value, sub, glow }: StatCardProps) {
  return (
    <div className={cn(
      "rounded-xl border border-border bg-bg-card p-5 transition-all hover:border-border/60",
      glow === "green" && "glow-green",
      glow === "accent" && "glow-accent",
      glow === "gold" && "glow-gold",
    )}>
      <p className="text-xs font-medium text-text-muted uppercase tracking-wider">{label}</p>
      <p className="mt-1 text-2xl font-bold tracking-tight">{value}</p>
      {sub && <p className="mt-1 text-xs text-text-secondary">{sub}</p>}
    </div>
  );
}
