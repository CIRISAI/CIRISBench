import { cn, formatPercentage } from "@/lib/utils";

interface ScoreCardProps {
  title: string;
  value: number | string;
  subtitle?: string;
  trend?: {
    value: number;
    isPositive: boolean;
  };
  icon?: React.ReactNode;
  className?: string;
}

export function ScoreCard({
  title,
  value,
  subtitle,
  trend,
  icon,
  className,
}: ScoreCardProps) {
  return (
    <div
      className={cn(
        "bg-card border border-border rounded-lg p-6",
        className
      )}
    >
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm text-muted-foreground">{title}</p>
          <p className="text-3xl font-bold mt-2">
            {typeof value === "number" ? formatPercentage(value) : value}
          </p>
          {subtitle && (
            <p className="text-sm text-muted-foreground mt-1">{subtitle}</p>
          )}
        </div>
        {icon && (
          <div className="text-2xl opacity-70">{icon}</div>
        )}
      </div>
      {trend && (
        <div className="mt-4 flex items-center gap-1">
          <span
            className={cn(
              "text-sm font-medium",
              trend.isPositive ? "text-green-500" : "text-red-500"
            )}
          >
            {trend.isPositive ? "↑" : "↓"} {Math.abs(trend.value).toFixed(1)}%
          </span>
          <span className="text-sm text-muted-foreground">vs last run</span>
        </div>
      )}
    </div>
  );
}

interface ScoreCardGridProps {
  children: React.ReactNode;
  columns?: 2 | 3 | 4;
  className?: string;
}

export function ScoreCardGrid({
  children,
  columns = 4,
  className,
}: ScoreCardGridProps) {
  return (
    <div
      className={cn(
        "grid gap-4",
        {
          "grid-cols-1 md:grid-cols-2": columns === 2,
          "grid-cols-1 md:grid-cols-3": columns === 3,
          "grid-cols-1 md:grid-cols-2 lg:grid-cols-4": columns === 4,
        },
        className
      )}
    >
      {children}
    </div>
  );
}
