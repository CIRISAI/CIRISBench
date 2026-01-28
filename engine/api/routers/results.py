from fastapi import APIRouter, HTTPException, status, Path as FastApiPath
from pathlib import Path
import logging
import ujson
from typing import List

# Add project root to path if not already added
import sys
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from schemas.results import Results # Import the Results schema
from config.settings import settings # To get base data paths

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/results",
    tags=["results"],
    responses={404: {"description": "Not found"}},
)

@router.get("/", response_model=dict)
async def list_pipeline_results():
    """
    Returns a list of run IDs for which results files exist.
    """
    logger.info("Listing available pipeline results.")
    results_dir = project_root / settings.results_dir
    result_ids: List[str] = []

    if not results_dir.exists() or not results_dir.is_dir():
        logger.warning(f"Results directory not found or is not a directory: {results_dir}")
        # Return empty list if dir doesn't exist, as no results are available
        return {"results": []}

    try:
        # Find files matching run_*.json
        for file_path in results_dir.glob("run_*.json"):
            if file_path.is_file():
                # Extract run_id (filename without prefix 'run_' and suffix '.json')
                run_id = file_path.stem[4:] # Remove 'run_' prefix
                result_ids.append(run_id)
        logger.info(f"Found {len(result_ids)} pipeline result files.")
        return {"results": sorted(result_ids)}
    except Exception as e:
        logger.error(f"Error scanning results directory {results_dir}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error listing pipeline results.")


@router.get("/{run_id}", response_model=Results)
async def get_pipeline_result(run_id: str = FastApiPath(..., title="The ID of the pipeline run (excluding 'run_' prefix)")):
    """
    Retrieves the full results JSON for a completed pipeline run,
    validated against the Results schema.
    """
    logger.info(f"Fetching results for run_id: {run_id}")
    results_dir = project_root / settings.results_dir
    # Construct filename using the 'run_' prefix
    results_file = results_dir / f"run_{run_id}.json"

    if not results_file.exists() or not results_file.is_file():
        logger.warning(f"Results file not found for run_id '{run_id}' at path {results_file}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Results for run_id '{run_id}' not found.")

    try:
        with open(results_file, 'r') as f:
            results_data = ujson.load(f)
        # Validate the loaded data against the Pydantic model.
        # FastAPI handles this automatically via response_model=Results.
        # If validation fails here, FastAPI returns a 500 error.
        # For more specific error handling (e.g., returning 422 on validation fail),
        # you could explicitly validate:
        # try:
        #     results_model = Results(**results_data)
        # except ValidationError as val_err:
        #     logger.error(f"Validation error for results '{run_id}': {val_err}")
        #     raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"Invalid results format: {val_err}")
        results_model = Results(**results_data) # Explicit validation example
        logger.info(f"Successfully loaded and validated results for run_id: {run_id}")
        return results_model # Return the Pydantic model instance
    except ujson.JSONDecodeError as e:
        logger.error(f"Error decoding JSON for results '{run_id}' at {results_file}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Invalid JSON format in results file for run_id '{run_id}'.")
    except Exception as e: # Catches Pydantic validation errors and others
        logger.error(f"Error loading or validating results '{run_id}' from {results_file}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error processing results for run_id '{run_id}': {e}")
