from fastapi import APIRouter, HTTPException, status, Body, BackgroundTasks, Path as FastApiPath, Depends, Request
from pathlib import Path
import logging
import uuid
import ujson
from enum import Enum
from typing import List

# Assuming schemas and config are importable relative to the project root
# Adjust imports based on actual project structure if needed
# Need to adjust sys.path or use relative imports if api/ is not in PYTHONPATH
import sys
import os
# Add project root to path to allow absolute imports like 'schemas.pipeline'
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from schemas.pipeline import Pipeline
# Import specific loaders instead of the non-existent load_component
from config.loader import (
    load_all_identities,
    load_all_guidances,
    load_all_guardrails
)
from config.settings import settings # To get base data paths
from core.engine import EthicsEngine # Import the engine class

logger = logging.getLogger(__name__)
# Configure basic logging if not already configured elsewhere
# logging.basicConfig(level=logging.INFO) # Consider moving config to main.py

router = APIRouter(
    prefix="/pipelines",
    tags=["pipelines"],
    responses={404: {"description": "Not found"}},
)

# Removed placeholder GET /pipelines/ endpoint as it's redundant with /definitions
# Removed the old placeholder /run endpoint. The new implementation is below.


@router.post("/validate", status_code=status.HTTP_200_OK)
async def validate_pipeline_definition(pipeline: Pipeline = Body(...)):
    """
    Checks if a provided pipeline definition JSON is structurally valid
    (handled by Pydantic/FastAPI) and if its referenced components
    (identity, guidance, guardrails) exist.
    """
    logger.info(f"Validating pipeline definition: {pipeline.id}") # Corrected attribute
    errors = []

    # --- Component Validation Logic ---
    # Note: Loading all components on each request is inefficient.
    # Consider caching these in app.state during startup for better performance.
    try:
        # 1. Validate Identity
        all_identities = load_all_identities()
        if pipeline.identity_id not in all_identities:
            errors.append(f"Identity with id '{pipeline.identity_id}' not found.")
            logger.warning(f"Validation failed: Identity '{pipeline.identity_id}' not found.")
        else:
            logger.debug(f"Identity '{pipeline.identity_id}' found.")

        # 2. Validate Ethical Guidance
        all_guidances = load_all_guidances()
        if pipeline.ethical_guidance_id not in all_guidances:
            errors.append(f"Ethical guidance with id '{pipeline.ethical_guidance_id}' not found.")
            logger.warning(f"Validation failed: Guidance '{pipeline.ethical_guidance_id}' not found.")
        else:
             logger.debug(f"Guidance '{pipeline.ethical_guidance_id}' found.")

        # 3. Validate Guardrails
        if pipeline.guardrail_ids:
            all_guardrails = load_all_guardrails()
            for guardrail_id in pipeline.guardrail_ids:
                if guardrail_id not in all_guardrails:
                    errors.append(f"Guardrail with id '{guardrail_id}' not found.")
                    logger.warning(f"Validation failed: Guardrail '{guardrail_id}' not found.")
                else:
                    logger.debug(f"Guardrail '{guardrail_id}' found.")
        else:
            logger.debug("No guardrails specified in the pipeline.")

    except Exception as e:
        # Catch potential errors during the loading process itself (e.g., file system issues)
        logger.error(f"Unexpected error during component loading for validation: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error during component validation: {e}"
        )
    # --- End Component Validation ---
    else:
        logger.debug("No guardrails specified in the pipeline.")


    if errors:
        # Use 400 for client errors (e.g., referencing non-existent components)
        # FastAPI's 422 handles structural validation errors automatically
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Pipeline validation failed: {'; '.join(errors)}"
        )

    logger.info(f"Pipeline definition '{pipeline.pipeline_id}' is valid.")
    return {"message": "Pipeline definition is valid."}


# --- Helper for validation (Refactored) ---
async def _validate_pipeline_components(pipeline: Pipeline):
    """Helper function to validate component existence."""
    errors = []
    # Use pipeline.id for logging/error messages if needed inside helper, though currently not used.
    try:
        # 1. Validate Identity
        all_identities = load_all_identities()
        if pipeline.identity_id not in all_identities:
            errors.append(f"Identity with id '{pipeline.identity_id}' not found.")

        # 2. Validate Ethical Guidance
        all_guidances = load_all_guidances()
        if pipeline.ethical_guidance_id not in all_guidances:
            errors.append(f"Ethical guidance with id '{pipeline.ethical_guidance_id}' not found.")

        # 3. Validate Guardrails
        if pipeline.guardrail_ids:
            all_guardrails = load_all_guardrails()
            for guardrail_id in pipeline.guardrail_ids:
                if guardrail_id not in all_guardrails:
                    errors.append(f"Guardrail with id '{guardrail_id}' not found.")

    except Exception as e:
        logger.error(f"Unexpected error during component loading for validation helper: {e}", exc_info=True)
        # Re-raise or handle as appropriate, maybe return a generic error message
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error during component validation: {e}"
        )

    if errors:
         raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Pipeline validation failed: {'; '.join(errors)}"
        )
# --- End Helper ---

# --- Dependency to get shared EthicsEngine instance ---
async def get_ethics_engine(request: Request) -> EthicsEngine:
    """Dependency to retrieve the EthicsEngine instance from app state."""
    if hasattr(request.app.state, 'ethics_engine'):
        return request.app.state.ethics_engine
    else:
        logger.error("EthicsEngine not found in application state!")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server configuration error: Ethics engine not initialized."
        )
# --- End Dependency ---


@router.post("/create", status_code=status.HTTP_201_CREATED)
async def create_pipeline_definition_file(pipeline: Pipeline = Body(...)):
    """
    Validates the provided pipeline definition's components and saves it
    as a new JSON file in the data/pipelines directory.
    Uses the pipeline.id as the filename base.
    Returns 409 Conflict if a file with that ID already exists.
    """
    logger.info(f"Received request to create pipeline definition file for ID: {pipeline.id}") # Corrected attribute

    # 1. Validate referenced components (Identity, Guidance, Guardrails)
    try:
        await _validate_pipeline_components(pipeline) # Use the existing helper
        logger.info(f"Pipeline components validated successfully for creation: {pipeline.id}") # Corrected attribute
    except HTTPException as http_exc:
        # If validation helper raised an HTTPException (e.g., 400 or 500), re-raise it
        logger.warning(f"Pipeline component validation failed for creation '{pipeline.id}': {http_exc.detail}") # Corrected attribute
        raise http_exc
    except Exception as e:
        # Catch any other unexpected errors from the helper
        logger.error(f"Unexpected error during component validation for creation '{pipeline.id}': {e}", exc_info=True) # Corrected attribute
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error during pipeline component validation: {e}"
        )

    # 2. Define target file path and check for conflicts
    pipelines_dir = project_root / settings.data_dir / "pipelines"
    # Basic sanitization - replace potential path separators in ID, though ideally IDs shouldn't contain them.
    # A more robust approach would involve stricter ID validation or mapping.
    safe_filename_base = pipeline.id.replace("/", "_").replace("\\", "_") # Corrected attribute
    definition_file = pipelines_dir / f"{safe_filename_base}.json"

    if definition_file.exists():
        logger.warning(f"Pipeline definition file already exists for ID '{pipeline.id}' at {definition_file}") # Corrected attribute
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Pipeline definition with ID '{pipeline.id}' already exists." # Corrected attribute
        )

    # 3. Save the pipeline definition file
    try:
        # Ensure parent directory exists
        definition_file.parent.mkdir(parents=True, exist_ok=True)
        with open(definition_file, 'w', encoding='utf-8') as f:
            f.write(pipeline.model_dump_json(indent=2))
        logger.info(f"Pipeline definition saved successfully to {definition_file}")
    except Exception as e:
        logger.error(f"Failed to save pipeline definition file for ID '{pipeline.id}' to {definition_file}: {e}", exc_info=True) # Corrected attribute
        # Clean up potentially partially created file? (Maybe not necessary)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save pipeline definition file: {e}"
        )

    # 4. Return success response
    return {"message": "Pipeline definition saved successfully.", "pipeline_id": pipeline.id} # Corrected attribute


@router.post("/{pipeline_name:path}/run", status_code=status.HTTP_202_ACCEPTED)
async def run_existing_pipeline(
    pipeline_name: str = FastApiPath(..., title="The ID/name of the pipeline definition file (without .json, path allowed)"),
    num_runs: int = 1, # Optional query parameter, default 1
    background_tasks: BackgroundTasks = BackgroundTasks(),
    engine: EthicsEngine = Depends(get_ethics_engine) # Inject engine instance
):
    """
    Loads an existing pipeline definition by its name/ID, validates its components,
    and triggers its asynchronous execution 'num_runs' times.
    """
    logger.info(f"Received request to run pipeline '{pipeline_name}' {num_runs} time(s).")

    # 1. Load the pipeline definition file
    pipelines_dir = project_root / settings.data_dir / "pipelines"
    # Use the :path converter name directly
    definition_file = pipelines_dir / f"{pipeline_name}.json"

    if not definition_file.exists() or not definition_file.is_file():
        logger.warning(f"Pipeline definition file not found for name '{pipeline_name}' at path {definition_file}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Pipeline definition '{pipeline_name}' not found.")

    try:
        with open(definition_file, 'r') as f:
            pipeline_json = f.read()
        # Validate the loaded data against the Pydantic model
        pipeline = Pipeline.model_validate_json(pipeline_json)
        logger.info(f"Successfully loaded pipeline definition: {pipeline.id}") # Corrected attribute
    except ujson.JSONDecodeError as e:
        logger.error(f"Error decoding JSON for pipeline definition '{pipeline_name}' at {definition_file}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"Invalid JSON format in pipeline definition '{pipeline_name}'.")
    except Exception as e: # Catches Pydantic validation errors and others
        logger.error(f"Error loading or validating pipeline definition '{pipeline_name}' from {definition_file}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"Error processing pipeline definition '{pipeline_name}': {e}")

    # 2. Validate referenced components (Identity, Guidance, Guardrails)
    try:
        await _validate_pipeline_components(pipeline) # Use the existing helper
        logger.info(f"Pipeline components validated successfully for run: {pipeline.id}") # Corrected attribute
    except HTTPException as http_exc:
        logger.warning(f"Pipeline component validation failed for run '{pipeline.id}': {http_exc.detail}") # Corrected attribute
        raise http_exc # Re-raise validation errors (400 or 500)
    except Exception as e:
        logger.error(f"Unexpected error during component validation for run '{pipeline.id}': {e}", exc_info=True) # Corrected attribute
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error during pipeline component validation: {e}"
        )

    # 3. Trigger runs in background
    run_ids = []
    if num_runs < 1:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="num_runs must be at least 1.")

    for i in range(num_runs):
        run_id = f"run_{uuid.uuid4().hex}"
        run_ids.append(run_id)
        # Pass a copy of the pipeline object if modifications are possible during run?
        # For now, assume engine.run_pipeline doesn't modify the input object.
        background_tasks.add_task(engine.run_pipeline, pipeline=pipeline, run_id=run_id)
        logger.info(f"Added pipeline run {i+1}/{num_runs} ({run_id}) for '{pipeline_name}' to background tasks.")

    # 4. Return Accepted response with run IDs
    return {"message": f"Pipeline '{pipeline_name}' submitted for execution {num_runs} time(s).", "run_ids": run_ids}


class PipelineStatus(str, Enum):
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"
    NOT_FOUND = "NOT_FOUND"


@router.get("/status/{run_id}", response_model=dict)
async def get_pipeline_run_status(run_id: str = FastApiPath(..., title="The ID of the pipeline run")):
    """
    Retrieves the current status of a specific pipeline run.
    Checks results file first, then log file.
    Note: The run_id path parameter *includes* the 'run_' prefix.
    """
    # Use the run_id directly from the path as it includes the prefix
    logger.info(f"Checking status for run_id: {run_id}")
    results_dir = project_root / settings.results_dir

    # --- Dynamically find the log file path ---
    log_file_path_str = None
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            log_file_path_str = handler.baseFilename
            break # Assume the first file handler is the main one

    if not log_file_path_str:
        logger.error("Could not find configured log file path from logging handlers.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Server configuration error: Log file path not found.")

    log_file_path = Path(log_file_path_str)
    # --- End dynamic log path finding ---

    # Construct filename using the run_id directly
    results_file = results_dir / f"{run_id}.json"

    # 1. Check for results file (indicates completion or error)
    if results_file.exists():
        try:
            with open(results_file, 'r') as f:
                results_data = ujson.load(f)
            # Check the 'outcome' field from the Results schema
            outcome = results_data.get("outcome")
            if outcome == "error":
                 logger.info(f"Status for {run_id}: ERROR (outcome='error' in results file)")
                 return {"run_id": run_id, "status": PipelineStatus.ERROR}
            # Treat success, failure, or guardrail violation as COMPLETED for status purposes
            elif outcome in ["success", "failure", "guardrail_violation"]:
                logger.info(f"Status for {run_id}: COMPLETED (outcome='{outcome}' in results file)")
                return {"run_id": run_id, "status": PipelineStatus.COMPLETED}
            else: # Handle unexpected or missing outcome field (e.g., "pending" shouldn't be in a saved file)
                 logger.warning(f"Unexpected or missing 'outcome' field ('{outcome}') in results file for {run_id}. Treating as ERROR.")
                 return {"run_id": run_id, "status": PipelineStatus.ERROR}
        except ujson.JSONDecodeError as e:
             logger.error(f"Error decoding results file {results_file} for {run_id}: {e}", exc_info=True)
             return {"run_id": run_id, "status": PipelineStatus.ERROR} # Treat JSON error as run error
        except Exception as e:
            logger.error(f"Error reading results file {results_file} for {run_id}: {e}", exc_info=True)
            return {"run_id": run_id, "status": PipelineStatus.ERROR} # Treat other read errors as run error

    # 2. Check log file for signs of running
    if log_file_path.exists():
        try:
            with open(log_file_path, 'r') as f:
                # This can be inefficient for large logs. Consider optimization if needed.
                log_content = f.read()
                # Check for start marker but not finish marker
                # Assumes core.engine logs messages like:
                # "Starting pipeline run: run_..."
                # "Finished pipeline run: run_..." or "Error in pipeline run: run_..."
                # Use run_id directly for checking markers
                start_marker = f"Starting pipeline run: {run_id}"
                finish_marker = f"Finished pipeline run: {run_id}"
                error_marker = f"Error in pipeline run: {run_id}" # Or similar

                if start_marker in log_content and finish_marker not in log_content and error_marker not in log_content:
                    logger.info(f"Status for {run_id}: RUNNING (start marker found in logs, no end/error marker)")
                    return {"run_id": run_id, "status": PipelineStatus.RUNNING}
                elif start_marker in log_content and (finish_marker in log_content or error_marker in log_content):
                    # Log indicates finished/error, but results file wasn't found/readable? Log this inconsistency.
                    logger.warning(f"Log inconsistency for {run_id}: Log shows completion/error, but results file issue detected earlier.")
                    # Fall through to NOT_FOUND based on results file check priority
                elif start_marker not in log_content:
                     logger.info(f"Status for {run_id}: NOT_FOUND (start marker not found in logs)")
                     # Fall through to NOT_FOUND
                # else: fall through

        except Exception as e:
            logger.error(f"Error reading log file {log_file_path} for {run_id}: {e}", exc_info=True)
            # If logs can't be read, we can't determine RUNNING status reliably. Fall back.

    # 3. If neither results nor logs indicate the run, assume not found
    logger.info(f"Status for {run_id}: NOT_FOUND (no results file or running state in logs)")
    # Raise 404 Not Found
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Pipeline run '{run_id}' not found or status could not be determined.")


@router.get("/logs/{run_id}", response_model=dict)
async def get_pipeline_run_logs(run_id: str = FastApiPath(..., title="The ID of the pipeline run")):
    """
    Retrieves detailed log entries associated with a specific pipeline run
    by filtering the main log file.
    Note: The run_id path parameter *includes* the 'run_' prefix.
    """
    # Use the run_id directly from the path as it includes the prefix
    logger.info(f"Fetching logs for run_id: {run_id}")

    # --- Dynamically find the log file path ---
    log_file_path_str = None
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            log_file_path_str = handler.baseFilename
            break # Assume the first file handler is the main one

    if not log_file_path_str:
        logger.error("Could not find configured log file path from logging handlers.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Server configuration error: Log file path not found.")

    log_file_path = Path(log_file_path_str)
    # --- End dynamic log path finding ---


    if not log_file_path.exists():
        logger.warning(f"Log file not found at determined path: {log_file_path} while fetching logs for {run_id}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Log file not found.")

    relevant_logs: List[str] = []
    try:
        with open(log_file_path, 'r') as f:
            for line in f:
                # Use run_id directly for checking log lines
                if run_id in line:
                    relevant_logs.append(line.strip())
    except Exception as e:
        logger.error(f"Error reading log file {log_file_path} for {run_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error reading log file.")

    if not relevant_logs:
        # It's debatable whether this should be 404 or 200 with empty list.
        # Let's go with 404 if *no* logs for this ID are found, implying the ID might be wrong
        # or the run never logged anything (which might be an issue itself).
        logger.info(f"No logs found containing run_id: {run_id}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No logs found for run_id '{run_id}'.")

    logger.info(f"Found {len(relevant_logs)} log entries for run_id: {run_id}")
    return {"run_id": run_id, "logs": relevant_logs}


# --- Pipeline Definition Endpoints ---

@router.get("/definitions", response_model=dict)
async def list_pipeline_definitions():
    """
    Returns a list of available pipeline definition IDs (filenames relative to data/pipelines).
    """
    logger.info("Listing available pipeline definitions.")
    pipelines_dir = project_root / settings.data_dir / "pipelines"
    definition_ids: List[str] = []
    try:
        # Use rglob to find all .json files recursively
        for file_path in pipelines_dir.rglob("*.json"):
            # Calculate relative path from the pipelines_dir to use as ID
            relative_path = file_path.relative_to(pipelines_dir)
            # Convert to string and remove .json suffix
            definition_id = str(relative_path.with_suffix(''))
            definition_ids.append(definition_id)
        logger.info(f"Found {len(definition_ids)} pipeline definitions.")
        return {"pipelines": sorted(definition_ids)}
    except Exception as e:
        logger.error(f"Error scanning pipeline definitions directory {pipelines_dir}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error listing pipeline definitions.")


@router.get("/definitions/{pipeline_id:path}", response_model=Pipeline)
async def get_pipeline_definition(pipeline_id: str = FastApiPath(..., title="The ID of the pipeline definition (e.g., 'example_pipeline' or 'subdir/my_pipeline')")):
    """
    Retrieves the full JSON definition of a specific pipeline by its ID.
    The pipeline_id corresponds to the relative path within data/pipelines without the .json extension.
    """
    logger.info(f"Fetching pipeline definition for ID: {pipeline_id}")
    pipelines_dir = project_root / settings.data_dir / "pipelines"
    # Reconstruct the file path from the ID
    # The :path converter in FastAPI allows slashes in the parameter
    definition_file = pipelines_dir / f"{pipeline_id}.json"

    if not definition_file.exists() or not definition_file.is_file():
        logger.warning(f"Pipeline definition file not found for ID '{pipeline_id}' at path {definition_file}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Pipeline definition '{pipeline_id}' not found.")

    try:
        with open(definition_file, 'r') as f:
            pipeline_data = ujson.load(f)
        # Validate the loaded data against the Pydantic model
        # FastAPI does this automatically if response_model=Pipeline is used,
        # but explicit validation here catches errors before returning.
        pipeline_model = Pipeline(**pipeline_data)
        logger.info(f"Successfully loaded and validated pipeline definition for ID: {pipeline_id}")
        return pipeline_model # Return the Pydantic model instance
    except ujson.JSONDecodeError as e:
        logger.error(f"Error decoding JSON for pipeline definition '{pipeline_id}' at {definition_file}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Invalid JSON format in pipeline definition '{pipeline_id}'.")
    except Exception as e: # Catches Pydantic validation errors and others
        logger.error(f"Error loading or validating pipeline definition '{pipeline_id}' from {definition_file}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error processing pipeline definition '{pipeline_id}': {e}")
