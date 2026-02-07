"use client";

import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from "recharts";
import type { ModelEvaluation } from "@/lib/api";

interface Props {
  evaluations: ModelEvaluation[];
}

export function AccuracyHistory({ evaluations }: Props) {
  const data = [...evaluations]
    .reverse()
    .map((ev, i) => ({
      idx: i + 1,
      accuracy: +(ev.accuracy * 100).toFixed(1),
      date: ev.completed_at
        ? new Date(ev.completed_at).toLocaleDateString("en-US", { month: "short", day: "numeric" })
        : `#${i + 1}`,
    }));

  if (data.length < 2) {
    return (
      <div className="flex items-center justify-center h-48 text-text-muted text-sm">
        Needs 2+ evaluations to show trend
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={260}>
      <LineChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#2a2a3a" />
        <XAxis
          dataKey="date"
          tick={{ fill: "#8888a0", fontSize: 11 }}
          tickLine={false}
          axisLine={{ stroke: "#2a2a3a" }}
        />
        <YAxis
          domain={[0, 100]}
          tick={{ fill: "#8888a0", fontSize: 11 }}
          tickLine={false}
          axisLine={{ stroke: "#2a2a3a" }}
          tickFormatter={(v: number) => `${v}%`}
        />
        <Tooltip
          contentStyle={{
            background: "#12121a",
            border: "1px solid #2a2a3a",
            borderRadius: 8,
            fontSize: 12,
          }}
          formatter={(v: number) => [`${v}%`, "Accuracy"]}
        />
        <Line
          type="monotone"
          dataKey="accuracy"
          stroke="#6c5ce7"
          strokeWidth={2}
          dot={{ fill: "#6c5ce7", r: 4 }}
          activeDot={{ r: 6, stroke: "#fff", strokeWidth: 2 }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
