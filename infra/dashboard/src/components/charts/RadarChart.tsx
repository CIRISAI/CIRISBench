"use client";

import {
  Radar,
  RadarChart as RechartsRadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  Legend,
  Tooltip,
} from "recharts";

interface ModelData {
  name: string;
  color: string;
  results: Record<string, number>;
}

interface RadarChartProps {
  models: ModelData[];
  categories: string[];
}

const COLORS = [
  "#3b82f6", // blue
  "#10b981", // green
  "#f59e0b", // amber
  "#ef4444", // red
  "#8b5cf6", // purple
  "#ec4899", // pink
];

export function RadarChart({ models, categories }: RadarChartProps) {
  // Transform data for Recharts
  const data = categories.map((category) => {
    const point: Record<string, string | number> = {
      category: category.replace(/_/g, " "),
    };
    models.forEach((model) => {
      point[model.name] = (model.results[category] || 0) * 100;
    });
    return point;
  });

  return (
    <ResponsiveContainer width="100%" height="100%">
      <RechartsRadarChart data={data}>
        <PolarGrid stroke="hsl(var(--border))" />
        <PolarAngleAxis
          dataKey="category"
          tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 12 }}
          tickLine={{ stroke: "hsl(var(--border))" }}
        />
        <PolarRadiusAxis
          angle={90}
          domain={[0, 100]}
          tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 10 }}
          tickFormatter={(value) => `${value}%`}
        />
        {models.map((model, index) => (
          <Radar
            key={model.name}
            name={model.name}
            dataKey={model.name}
            stroke={model.color || COLORS[index % COLORS.length]}
            fill={model.color || COLORS[index % COLORS.length]}
            fillOpacity={0.2}
            strokeWidth={2}
          />
        ))}
        <Legend
          wrapperStyle={{
            paddingTop: "20px",
          }}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: "hsl(var(--card))",
            border: "1px solid hsl(var(--border))",
            borderRadius: "8px",
          }}
          labelStyle={{ color: "hsl(var(--foreground))" }}
          formatter={(value: number) => [`${value.toFixed(1)}%`, ""]}
        />
      </RechartsRadarChart>
    </ResponsiveContainer>
  );
}
