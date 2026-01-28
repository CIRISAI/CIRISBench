# HE-300 Dashboard

Web interface for viewing and comparing Hendrycks Ethics 300 benchmark results.

## Features

- **Real-time Results Dashboard**: View benchmark results as they come in
- **Model Comparison**: Compare accuracy across different models and configurations
- **Category Breakdown**: Detailed analysis by ethical category (commonsense, deontology, justice, utilitarianism, virtue)
- **Artifact Storage**: Access logs, reports, and model outputs
- **CI/CD Integration**: Webhook endpoint for automated result ingestion

## Tech Stack

- **Framework**: Next.js 14 (App Router)
- **Database**: PostgreSQL with Prisma ORM
- **Styling**: Tailwind CSS with CSS variables for theming
- **Charts**: Recharts for data visualization
- **Authentication**: JWT-based webhook authentication

## Getting Started

### Prerequisites

- Node.js 20+
- PostgreSQL 15+
- Docker (optional)

### Local Development

1. Install dependencies:
   ```bash
   npm install
   ```

2. Set up environment variables:
   ```bash
   cp .env.example .env.local
   # Edit .env.local with your database URL and secrets
   ```

3. Initialize the database:
   ```bash
   npx prisma migrate dev
   npx prisma generate
   ```

4. Start the development server:
   ```bash
   npm run dev
   ```

5. Open [http://localhost:3000](http://localhost:3000)

### Docker Deployment

```bash
# From the staging directory
docker compose -f docker/docker-compose.dashboard.yml up -d
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `DATABASE_URL` | PostgreSQL connection string | Yes |
| `WEBHOOK_SECRET` | Secret for webhook authentication | Yes |
| `S3_BUCKET` | S3 bucket for artifact storage | No |
| `S3_REGION` | AWS region for S3 | No |
| `NEXT_PUBLIC_APP_URL` | Public URL of the dashboard | No |

## API Endpoints

### `GET /api/health`
Health check endpoint for load balancers and monitoring.

### `POST /api/webhook`
Webhook endpoint for CI/CD to push benchmark results.

**Headers:**
- `Authorization: Bearer <WEBHOOK_SECRET>`

**Body:**
```json
{
  "run_id": "uuid",
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "sample_size": 300,
  "results": [
    {"category": "commonsense", "total": 60, "correct": 54, "accuracy": 0.9}
  ],
  "artifacts": [
    {"name": "results.json", "url": "s3://...", "type": "report", "size": 1024}
  ]
}
```

### `GET /api/results`
Query benchmark results with filtering and pagination.

**Query Parameters:**
- `model`: Filter by model name
- `status`: Filter by status (pending, running, completed, failed)
- `environment`: Filter by environment (dev, staging, prod)
- `limit`: Number of results (default: 20)
- `offset`: Pagination offset (default: 0)

## Pages

- `/` - Dashboard home with recent runs
- `/runs/[id]` - Detailed view of a benchmark run
- `/compare` - Side-by-side model comparison
- `/settings` - Configuration and API settings

## Database Schema

See `prisma/schema.prisma` for the complete schema. Key models:

- **BenchmarkRun**: Individual benchmark execution
- **ScenarioResult**: Per-category results
- **ModelConfig**: Model configuration presets
- **Artifact**: Stored files (logs, reports, outputs)

## Development

### Type Checking
```bash
npm run type-check
```

### Linting
```bash
npm run lint
```

### Format
```bash
npm run format
```

### Database Migrations
```bash
# Create a new migration
npx prisma migrate dev --name <migration_name>

# Apply migrations in production
npx prisma migrate deploy
```

## License

See LICENSE in the root directory.
