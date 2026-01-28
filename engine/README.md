# EthicsEngine 1.0

EthicsEngine 1.0 is a framework for **evaluating and comparing ethical reasoning and guardrail effectiveness** in language model pipelines. It allows developers to simulate complex scenarios and Q&A benchmarks with large language models (LLMs) to assess how different moral guidelines and safety guardrails impact the model’s responses.

Built on the AG2 v0.8.5 platform, EthicsEngine leverages the AG2 **ReasoningAgent** as its core orchestration agent to manage multi-step interactions and decision-making.

## Core Concepts

The system uses a modular architecture based on several key schemas:

*   **Pipelines:** Define self-contained sequences of interactions and evaluations simulating ethical scenarios or benchmarks. Configured via JSON files in `data/pipelines/`.
*   **Stages:** Represent discrete steps within a pipeline (e.g., LLM interaction, evaluation). Pipelines are composed of ordered stages.
*   **Identities:** Define user profiles or contexts (e.g., personas, cultural backgrounds) to test how LLM responses adapt. Defined in `data/identities/`.
*   **Ethical Guidance:** Specifies the moral reasoning framework (e.g., Utilitarian, Deontological) the LLM should follow. Defined in `data/guidances/`.
*   **Guardrails:** Safety constraints and content rules (e.g., no hate speech, no self-harm) applied during pipeline execution. Defined in `data/guardrails/`.
*   **Results:** Structured output capturing the configuration, interactions, violations, and metrics from a pipeline run. Saved in `results/`.

## Project Structure

*   `config/`: Global settings and configuration loading.
*   `core/`: Core engine logic, including the EthicsAgent, stage handlers, and guardrail engine.
*   `data/`: Configuration files for identities, guidances, guardrails, and pipeline definitions.
*   `datasets/`: Source datasets used for ingestion.
*   `logs/`: Log files generated during runs.
*   `results/`: Output JSON files for each pipeline run.
*   `schemas/`: Pydantic schemas defining the structure for pipelines, stages, results, etc.
*   `scripts/`: Utility scripts for tasks like data ingestion.
*   `tests/`: Unit and integration tests.
*   `utils/`: Shared utility functions (e.g., logging config, placeholder resolution).

## Setup

1.  Clone the repository.
2.  Create and activate a Python virtual environment (recommended).
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  Configure any necessary environment variables (e.g., API keys for LLMs) as needed by the `config/settings.py` or `.env` file (if used).

## License

EthicsEngine is provided as a closed-source Software-as-a-Service (SaaS) product. Your use of the service is subject to the terms and conditions outlined in the End-User License Agreement (EULA). Please review the `EULA.md` file for full details.

## Usage

### Running Pipelines

Pipelines can be executed using the `run_pipeline.py` script. Provide the path to the pipeline definition file (or just the ID if it's in the default `data/pipelines/` directory).

```bash
# Example: Run a specific pipeline from the ethics dataset
python3 run_pipeline.py data/pipelines/ethics/commonsense/ethics_commonsense_AITA..._some_id.json

# Example: Run a benchmark pipeline by ID
python3 run_pipeline.py bench_q1
```

Results will be saved as JSON files in the `results/` directory.

### Running the API and UI

The project includes a FastAPI backend and a Streamlit frontend for interacting with pipelines.

1.  **Start the FastAPI Backend:**
    Ensure all dependencies are installed (`pip install -r requirements.txt`). Run the Uvicorn server from the project root:
    ```bash
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
    ```
    The `--reload` flag automatically restarts the server when code changes are detected (useful for development). The API documentation will be available at `http://localhost:8000/docs`.

2.  **Start the Streamlit UI:**
    In a **separate terminal**, navigate to the project root and run:
    ```bash
    streamlit run ui/app.py
    ```
    This will open the Streamlit application in your web browser, allowing you to interact with the API to manage and monitor pipeline runs.

### Data Ingestion

Use scripts in the `scripts/` directory to ingest data from source datasets (like those listed in the FSD Appendix) into the pipeline format.

```bash
# Example: Ingest the ethics dataset
python3 scripts/ingest_ethics_dataset.py
```

## Ethical Principles

The framework is designed to evaluate AI behavior based on core ethical principles, including:

*   Autonomy
*   Fairness (Justice)
*   Harm Reduction (Non-Maleficence)
*   Beneficence
*   Honesty (Truthfulness)
*   Accountability
*   Privacy
*   Transparency

These principles inform scenario design, ethical guidance configurations, and evaluation metrics.
