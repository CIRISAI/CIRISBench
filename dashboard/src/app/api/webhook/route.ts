import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";
import { z } from "zod";
import crypto from "crypto";

// Webhook payload schema
const webhookSchema = z.object({
  run_id: z.string(),
  status: z.enum(["PENDING", "RUNNING", "COMPLETED", "FAILED", "CANCELLED"]),
  model: z.string(),
  quantization: z.string().optional().default("Q4_K_M"),
  sample_size: z.number(),
  seed: z.number().optional().default(42),
  results: z
    .array(
      z.object({
        category: z.string(),
        total: z.number(),
        correct: z.number(),
        accuracy: z.number(),
        avg_latency: z.number().optional(),
        avg_tokens: z.number().optional(),
        scenarios: z.any().optional(),
      })
    )
    .optional(),
  duration: z.number().optional(),
  error_message: z.string().optional(),
  commit_sha: z.string().optional(),
  branch: z.string().optional(),
  pr_number: z.number().optional(),
  gpu_type: z.string().optional(),
  gpu_memory: z.number().optional(),
  runner_type: z.string().optional().default("unknown"),
  environment: z.string().optional().default("unknown"),
  tokens_per_sec: z.number().optional(),
});

function verifySignature(signature: string | null, body: string): boolean {
  const secret = process.env.WEBHOOK_SECRET;
  if (!secret) {
    console.warn("WEBHOOK_SECRET not set, skipping signature verification");
    return true;
  }
  if (!signature) return false;

  const expected = crypto
    .createHmac("sha256", secret)
    .update(body)
    .digest("hex");

  const sig = signature.replace("sha256=", "");
  
  try {
    return crypto.timingSafeEqual(Buffer.from(sig), Buffer.from(expected));
  } catch {
    return false;
  }
}

export async function POST(request: NextRequest) {
  const bodyText = await request.text();
  
  // Verify webhook signature
  const signature = request.headers.get("x-webhook-signature");
  if (!verifySignature(signature, bodyText)) {
    return NextResponse.json({ error: "Invalid signature" }, { status: 401 });
  }

  let body: unknown;
  try {
    body = JSON.parse(bodyText);
  } catch {
    return NextResponse.json({ error: "Invalid JSON" }, { status: 400 });
  }

  // Validate payload
  const parsed = webhookSchema.safeParse(body);
  if (!parsed.success) {
    return NextResponse.json(
      { error: "Invalid payload", details: parsed.error.issues },
      { status: 400 }
    );
  }

  const data = parsed.data;

  try {
    // Log the webhook event
    await prisma.webhookEvent.create({
      data: {
        source: data.runner_type,
        eventType: `run_${data.status.toLowerCase()}`,
        payload: body as object,
      },
    });

    // Upsert the benchmark run
    const run = await prisma.benchmarkRun.upsert({
      where: { id: data.run_id },
      update: {
        status: data.status,
        duration: data.duration,
        errorMessage: data.error_message,
        tokensPerSec: data.tokens_per_sec,
        completedAt:
          data.status === "COMPLETED" || data.status === "FAILED"
            ? new Date()
            : undefined,
      },
      create: {
        id: data.run_id,
        model: data.model,
        quantization: data.quantization,
        sampleSize: data.sample_size,
        seed: data.seed,
        status: data.status,
        gpuType: data.gpu_type,
        gpuMemory: data.gpu_memory,
        runnerType: data.runner_type,
        environment: data.environment,
        commitSha: data.commit_sha,
        branch: data.branch,
        prNumber: data.pr_number,
        duration: data.duration,
        tokensPerSec: data.tokens_per_sec,
        errorMessage: data.error_message,
      },
    });

    // Insert category results if completed
    if (data.status === "COMPLETED" && data.results) {
      // Delete existing results to handle re-runs
      await prisma.categoryResult.deleteMany({
        where: { runId: run.id },
      });

      await prisma.categoryResult.createMany({
        data: data.results.map((r) => ({
          runId: run.id,
          category: r.category,
          total: r.total,
          correct: r.correct,
          accuracy: r.accuracy,
          avgLatency: r.avg_latency,
          avgTokens: r.avg_tokens,
          scenarios: r.scenarios,
        })),
      });

      // Update model stats
      const modelStats = await prisma.categoryResult.aggregate({
        where: {
          run: {
            model: data.model,
            status: "COMPLETED",
          },
        },
        _avg: { accuracy: true },
        _max: { accuracy: true },
        _count: true,
      });

      await prisma.model.upsert({
        where: { name: data.model },
        update: {
          totalRuns: { increment: 1 },
          avgAccuracy: modelStats._avg.accuracy,
          bestAccuracy: modelStats._max.accuracy,
          lastRunAt: new Date(),
        },
        create: {
          name: data.model,
          displayName: data.model,
          provider: data.model.includes("/") ? data.model.split("/")[0] : "unknown",
          totalRuns: 1,
          avgAccuracy: modelStats._avg.accuracy,
          bestAccuracy: modelStats._max.accuracy,
          lastRunAt: new Date(),
        },
      });
    }

    return NextResponse.json({
      success: true,
      run_id: run.id,
      status: run.status,
    });
  } catch (error) {
    console.error("Webhook processing error:", error);
    
    // Log the error
    await prisma.webhookEvent.create({
      data: {
        source: data.runner_type,
        eventType: "error",
        payload: body as object,
        processed: false,
        error: error instanceof Error ? error.message : "Unknown error",
      },
    });

    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}

// Health check
export async function GET() {
  return NextResponse.json({ status: "ok", endpoint: "webhook" });
}
