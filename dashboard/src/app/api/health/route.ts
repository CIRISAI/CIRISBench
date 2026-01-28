import { NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";

export async function GET() {
  const startTime = Date.now();
  
  const checks = {
    database: { status: "unknown" as "ok" | "error" | "unknown", latency: 0 },
    version: process.env.npm_package_version || "1.0.0",
    uptime: process.uptime(),
  };

  // Check database connectivity
  try {
    const dbStart = Date.now();
    await prisma.$queryRaw`SELECT 1`;
    checks.database = {
      status: "ok",
      latency: Date.now() - dbStart,
    };
  } catch (error) {
    checks.database = {
      status: "error",
      latency: 0,
    };
  }

  const isHealthy = checks.database.status === "ok";
  const totalLatency = Date.now() - startTime;

  return NextResponse.json(
    {
      status: isHealthy ? "healthy" : "unhealthy",
      timestamp: new Date().toISOString(),
      latency: totalLatency,
      checks,
    },
    { status: isHealthy ? 200 : 503 }
  );
}
