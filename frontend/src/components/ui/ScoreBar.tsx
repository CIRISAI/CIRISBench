import { cn } from "@/lib/utils";

interface ScoreBarProps {
  value: number; // 0-1
  size?: "sm" | "md" | "lg";
  showLabel?: boolean;
}

function barColor(v: number): string {
  if (v >= 0.9) return "bg-green";
  if (v >= 0.7) return "bg-accent";
  if (v >= 0.5) return "bg-amber";
  return "bg-red";
}

export function ScoreBar({ value, size = "md", showLabel = true }: ScoreBarProps) {
  const h = size === "sm" ? "h-1.5" : size === "lg" ? "h-3" : "h-2";
  return (
    <div className="flex items-center gap-2 w-full">
      <div className={cn("flex-1 rounded-full bg-bg-elevated overflow-hidden", h)}>
        <div
          className={cn("h-full rounded-full transition-all duration-500", barColor(value))}
          style={{ width: `${Math.min(value * 100, 100)}%` }}
        />
      </div>
      {showLabel && (
        <span className="text-xs font-mono text-text-secondary w-12 text-right">
          {(value * 100).toFixed(1)}%
        </span>
      )}
    </div>
  );
}
