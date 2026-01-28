"use client";

import {
  BarChart as RechartsBarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";

interface CategoryResult {
  category: string;
  accuracy: number;
  correct: number;
  total: number;
}

interface BarChartProps {
  data: CategoryResult[];
}

function getBarColor(accuracy: number): string {
  if (accuracy >= 0.9) return "#10b981"; // green
  if (accuracy >= 0.7) return "#f59e0b"; // amber
  return "#ef4444"; // red
}

export function BarChart({ data }: BarChartProps) {
  const chartData = data.map((item) => ({
    ...item,
    accuracyPercent: item.accuracy * 100,
    label: item.category.replace(/_/g, " "),
  }));

  return (
    <ResponsiveContainer width="100%" height="100%">
      <RechartsBarChart
        data={chartData}
        layout="vertical"
        margin={{ top: 5, right: 30, left: 100, bottom: 5 }}
      >
        <CartesianGrid
          strokeDasharray="3 3"
          stroke="hsl(var(--border))"
          horizontal={true}
          vertical={false}
        />
        <XAxis
          type="number"
          domain={[0, 100]}
          tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 12 }}
          tickFormatter={(value) => `${value}%`}
        />
        <YAxis
          type="category"
          dataKey="label"
          tick={{ fill: "hsl(var(--foreground))", fontSize: 12 }}
          tickLine={false}
          axisLine={false}
          width={95}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: "hsl(var(--card))",
            border: "1px solid hsl(var(--border))",
            borderRadius: "8px",
          }}
          labelStyle={{ color: "hsl(var(--foreground))" }}
          formatter={(value: number, name: string, props: any) => [
            `${value.toFixed(1)}% (${props.payload.correct}/${props.payload.total})`,
            "Accuracy",
          ]}
        />
        <Bar dataKey="accuracyPercent" radius={[0, 4, 4, 0]}>
          {chartData.map((entry, index) => (
            <Cell key={`cell-${index}`} fill={getBarColor(entry.accuracy)} />
          ))}
        </Bar>
      </RechartsBarChart>
    </ResponsiveContainer>
  );
}
