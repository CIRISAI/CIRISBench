import logging
import uuid
import os # Import os for save_results path handling
from datetime import datetime, timezone
import asyncio # Import asyncio for async operations
from typing import Dict, Any, List, Optional

# Assuming schemas are in ../schemas/
from schemas.pipeline import Pipeline
from schemas.results import Results, ResultOutcome, ResultMetrics, ResultViolation
from schemas.interaction import Interaction, InteractionRole
from schemas.identity import Identity
from schemas.ethical_guidance import EthicalGuidance
from schemas.guardrail import Guardrail, GuardrailScope

# Import config loader and settings
from config import loader
from config.settings import settings

# Import stage handler registry and guardrail checker
from core.stages import STAGE_HANDLER_REGISTRY
from core.guardrails import check_guardrails

import io # For redirecting stdout/stderr
import time # For agent naming uniqueness
from contextlib import redirect_stderr, redirect_stdout # For capturing agent output

# --- Autogen Imports (copied from llm_handler for consistency) ---
try:
    from autogen import AssistantAgent, ConversableAgent, UserProxyAgent
    from autogen.agents.experimental import ReasoningAgent, ThinkNode
    AUTOGEN_AVAILABLE = True
except ImportError:
    # Define dummy classes if autogen is not available
    class ThinkNode:
        def __init__(self, content, parent=None): self.content=content; self.depth=0; self.value=0; self.visits=0; self.children=[]
        def to_dict(self): return {"content": self.content, "children": []}
    class ReasoningAgent:
        def __init__(self, *args, **kwargs): self._root = None
        def generate_reply(self, *args, **kwargs): return "Dummy Reply - Autogen Import Failed"
    class AssistantAgent:
         def __init__(self, *args, **kwargs): pass
         def generate_reply(self, *args, **kwargs): return "Dummy Reply - Autogen Import Failed"
    class ConversableAgent:
         def __init__(self, *args, **kwargs): pass
    class UserProxyAgent:
         def __init__(self, *args, **kwargs): pass
         def initiate_chat(self, *args, **kwargs): pass
         def last_message(self, agent): return {"content": "Dummy Reply - Autogen Import Failed"}

    AUTOGEN_AVAILABLE = False
    ReasoningAgent = None # Ensure it's None if not imported
    AssistantAgent = None
    UserProxyAgent = None
    ConversableAgent = None

# Import shared configs from llm_handler context
from config import (AGENT_TIMEOUT, AG2_REASONING_SPECS, AUTOGEN_AVAILABLE,
                    llm_config, semaphore, settings)

logger = logging.getLogger(__name__)

class EthicsEngine:
    """
    Core engine for executing EthicsEngine pipelines.
    Orchestrates stages, applies ethical guidance and guardrails,
     and produces structured results.
     """

    def __init__(self, load_configs: bool = True, load_pipelines_on_init: bool = True):
        """
        Initializes the EthicsEngine.

        Args:
            load_configs: If True, load identities, guidances, guardrails immediately.
            load_pipelines_on_init: If True (and load_configs is True), load pipelines immediately.
        """
        self._identities: Dict[str, Identity] = {}
        self._guidances: Dict[str, EthicalGuidance] = {}
        self._guardrails: Dict[str, Guardrail] = {}
        self._pipelines: Dict[str, Pipeline] = {}
        self._load_pipelines_on_init = load_pipelines_on_init # Store the flag
        logger.info("EthicsEngine initializing...")
        if load_configs:
            self.load_configurations(data_dir=settings.data_dir) # Pass default data_dir
        logger.info("EthicsEngine initialized.")

    def load_configurations(self, data_dir: str = settings.data_dir, load_pipelines: Optional[bool] = None):
        """
        Loads identities, ethical guidance, guardrails, and optionally pipelines from data directory.

        Args:
            data_dir: The directory containing configuration data.
            load_pipelines: If True, load pipelines. If False, skip loading pipelines.
                            If None, use the value of self._load_pipelines_on_init set during init.
        """
        logger.info(f"Loading configurations from {data_dir}...")
        self._identities = loader.load_all_identities(data_dir)
        self._guidances = loader.load_all_guidances(data_dir)
        self._guardrails = loader.load_all_guardrails(data_dir)

        should_load_pipelines = load_pipelines if load_pipelines is not None else self._load_pipelines_on_init

        if should_load_pipelines:
            self._pipelines = loader.load_all_pipelines(data_dir)
            logger.info(f"Configurations loaded: "
                        f"{len(self._identities)} identities, "
                        f"{len(self._guidances)} guidances, "
                        f"{len(self._guardrails)} guardrails, "
                        f"{len(self._pipelines)} pipelines.")
        else:
             logger.info(f"Configurations loaded (excluding pipelines): "
                        f"{len(self._identities)} identities, "
                        f"{len(self._guidances)} guidances, "
                        f"{len(self._guardrails)} guardrails.")


    async def run_pipeline(self, pipeline: Pipeline, run_id: str) -> Results: # Accept run_id
        """
        Executes a given pipeline configuration asynchronously using the provided run_id.

        Args:
            pipeline: The Pipeline object defining the run.
            run_id: The unique identifier for this specific run, provided by the caller.

        Returns:
            A Results object containing the outcome of the run.
        """
        # Use the provided run_id instead of generating a new one
        # run_id = f"run_{uuid.uuid4()}" # Removed internal generation
        start_time = datetime.now(timezone.utc)
        logger.info(f"Starting pipeline run: {run_id} (Pipeline ID: {pipeline.id})") # Log the provided run_id

        interactions: List[Interaction] = []
        violations: List[ResultViolation] = []
        # Initialize context with pipeline-level info if needed by stages
        pipeline_context: Dict[str, Any] = {
            "pipeline_id": pipeline.id,
            "pipeline_identity_id": pipeline.identity_id,
            "pipeline_guidance_id": pipeline.ethical_guidance_id,
            # Add expected outcome if defined, for evaluation stages
            "pipeline_expected_outcome": pipeline.evaluation_metrics.expected_outcome if pipeline.evaluation_metrics and hasattr(pipeline.evaluation_metrics, 'expected_outcome') else None
        }
        final_outcome: ResultOutcome = "pending" # Use string literal for Literal type
        outcome_details: str = ""
        stage_status: Any = None # To hold status from stage handlers/guardrails

        # Retrieve full Identity, EthicalGuidance, Guardrail objects based on IDs
        identity = self._identities.get(pipeline.identity_id)
        guidance = self._guidances.get(pipeline.ethical_guidance_id)
        active_guardrail_ids = pipeline.guardrail_ids or []
        active_guardrails = [self._guardrails.get(gid) for gid in active_guardrail_ids if self._guardrails.get(gid)]

        if not identity:
            logger.error(f"Identity '{pipeline.identity_id}' not found for pipeline {pipeline.id}.")
            final_outcome = "error" # Use string literal
            outcome_details = f"Configuration error: Identity '{pipeline.identity_id}' not found."
        if not guidance:
             logger.error(f"Ethical Guidance '{pipeline.ethical_guidance_id}' not found for pipeline {pipeline.id}.")
             final_outcome = "error" # Use string literal
             # Append to details if identity also missing
             outcome_details += f" Configuration error: Ethical Guidance '{pipeline.ethical_guidance_id}' not found."

        # Proceed only if essential configs are found
        if final_outcome != "error": # Use string literal
            try:
                # --- Stage Execution Loop ---
                for stage_index, stage in enumerate(pipeline.stages):
                    logger.info(f"Executing stage {stage_index + 1}/{len(pipeline.stages)}: {stage.id} (Type: {stage.type})")

                    # Find the appropriate handler
                    handler = STAGE_HANDLER_REGISTRY.get(stage.type)
                    if not handler:
                        logger.error(f"No handler registered for stage type '{stage.type}' (Stage ID: {stage.id}).")
                        stage_status = f"error: no handler for type '{stage.type}'"
                        final_outcome = "error" # Use string literal
                        outcome_details = f"Stage '{stage.id}' failed: No handler for type '{stage.type}'."
                        break # Stop pipeline

                    # Execute the stage handler
                    try:
                        # Pass necessary context: stage definition, current pipeline outputs, engine instance, identity, guidance
                        # Await the handler call since it might be async
                        pipeline_context, stage_interactions, stage_status = await handler(
                            stage=stage,
                            pipeline_context=pipeline_context,
                            engine_instance=self,
                            identity=identity, # Pass loaded identity
                            guidance=guidance, # Pass loaded guidance
                            active_guardrails=active_guardrails # Pass active guardrails
                        )
                        interactions.extend(stage_interactions)

                        # Check status returned by the handler (handler is now responsible for guardrail actions like block/modify)
                        if stage_status is not None:
                             # Assuming status indicates an error or violation that should stop the pipeline
                             logger.warning(f"Stage '{stage.id}' returned status: {stage_status}. Stopping pipeline.")
                             # Determine outcome based on status (e.g., if it's a specific violation type)
                             if "error" in str(stage_status).lower():
                                 final_outcome = "error" # Use string literal
                             else: # Assume other statuses might be specific violations handled internally
                                 final_outcome = "failure" # Use string literal
                             outcome_details = f"Stage '{stage.id}' failed with status: {stage_status}"
                             break # Stop pipeline

                        # Guardrail checks are now handled *within* the stage handlers (e.g., llm_handler)
                        # The handler should return appropriate status if a blocking guardrail is hit.
                        # Violations detected by handlers should ideally be added to the interactions list's metadata
                        # or returned somehow for aggregation in the final Results.
                        # For now, we rely on the handler setting the status correctly if blocked.

                    except Exception as stage_exc:
                        logger.error(f"Exception during stage '{stage.id}' execution: {stage_exc}", exc_info=True)
                        stage_status = f"error: exception in handler: {stage_exc}"
                        final_outcome = "error" # Use string literal
                        outcome_details = f"Stage '{stage.id}' failed unexpectedly: {stage_exc}"
                        # Record error interaction
                        interactions.append(Interaction(
                            stage_id=stage.id, role="system", # Use string literal
                            content=f"Error executing stage: {stage_exc}",
                            metadata=interactions[-1].metadata if interactions else None # Use last metadata
                        ))
                        break # Stop pipeline

                    # Check if a blocking guardrail violation occurred in the inner check
                    if final_outcome == "guardrail_violation": # Use string literal
                        break

                # If loop completed without break/error
                if final_outcome == "pending": # Use string literal for comparison
                     final_outcome = "success" # Use string literal
                     outcome_details = "Pipeline completed successfully."

            except Exception as e:
                logger.error(f"Unhandled error during pipeline run {run_id}: {e}", exc_info=True)
                final_outcome = "error" # Use string literal
                outcome_details = f"An unexpected error occurred during pipeline execution: {e}"
            finally:
                # --- Assemble Results ---
                # This block executes regardless of whether an exception occurred in the try block or if the loop completed normally
                end_time = datetime.now(timezone.utc)
                latency = (end_time - start_time).total_seconds()

                # --- Calculate Final Metrics ---
                total_tokens = 0
                for interaction in interactions:
                    if interaction.metadata and isinstance(interaction.metadata.tokens_used, int):
                        total_tokens += interaction.metadata.tokens_used

                # --- Correctness & Ethical Score Calculation ---
                correctness_score: Optional[float] = None
                evaluation_scores = [] # For potential future aggregation of multiple eval stages
                custom_metrics_agg = {} # Placeholder for aggregating custom metrics
                principle_alignment_agg = {} # Placeholder for aggregating principle alignment
                primary_ethical_score: Optional[float] = None # Added variable for specific score

                for stage in pipeline.stages:
                    if stage.type == "evaluation" and stage.id in pipeline_context:
                        stage_output = pipeline_context[stage.id]
                        # Assume the first output label holds the metrics object
                        if stage_output and stage.outputs and stage.outputs.spec: # Check if spec exists
                            # Define metrics_label within this scope
                            metrics_label = next(iter(stage.outputs.spec), None)
                            if metrics_label and isinstance(stage_output.get(metrics_label), dict):
                                metrics_dict = stage_output[metrics_label]
                                current_score: Optional[float] = None
                                if isinstance(metrics_dict.get("score"), (int, float)):
                                    current_score = metrics_dict["score"]
                                    evaluation_scores.append(current_score) # Still collect all scores for potential averaging later if needed

                                # --- Specific handling for ethics label evaluation ---
                                if stage.id == "evaluate_ethics_label" and current_score is not None:
                                    primary_ethical_score = current_score # Capture score from this specific stage
                                    correctness_score = current_score # Also use this as correctness for ethics pipelines
                                    logger.info(f"Retrieved primary ethical/correctness score from stage '{stage.id}': {primary_ethical_score:.4f}")
                                # --- End specific handling ---

                                # Original logic for 'evaluate_answer' (can keep for compatibility or remove if unused)
                                elif stage.id == "evaluate_answer" and current_score is not None: # Specific check for benchmark eval stage ID
                                    # Use the score from the LLM evaluation as the correctness score
                                    if correctness_score is None: # Only set if not already set by ethics label stage
                                        correctness_score = current_score
                                    logger.info(f"Retrieved score from benchmark evaluation stage '{stage.id}': {current_score:.4f}")
                                else:
                                     if current_score is None and stage.id != "evaluate_ethics_label": # Avoid double warning if score missing from ethics stage
                                         logger.warning(f"Evaluation stage '{stage.id}' metrics dictionary missing 'score' or score is not a number.")


                                # TODO: Add logic here to aggregate other metrics like principle_alignment or custom_metrics
                                # Example:
                                # if "principle_alignment" in metrics_dict:
                                #     principle_alignment_agg.update(metrics_dict["principle_alignment"])
                                # if "custom" in metrics_dict:
                                #     custom_metrics_agg.update(metrics_dict["custom"])

                # Use the specifically captured score if available, otherwise fallback to average (or None)
                final_ethical_score = primary_ethical_score
                if final_ethical_score is None and evaluation_scores:
                    # Fallback to average if specific score wasn't found but others exist
                    final_ethical_score = sum(evaluation_scores) / len(evaluation_scores)
                    logger.info(f"Using average ethical score as fallback: {final_ethical_score:.4f} from {len(evaluation_scores)} evaluation stage(s).")
                elif final_ethical_score is None:
                     logger.info("No ethical score could be determined from evaluation stages.")


                final_metrics = ResultMetrics(
                    latency_seconds=latency,
                    tokens_used_total=total_tokens if total_tokens > 0 else None,
                    correctness=correctness_score, # Set based on specific logic above
                    ethical_score=final_ethical_score # Use the specifically determined score
                    # Add aggregated principle_alignment_agg and custom_metrics_agg here if implemented
                )
                # --- End Metrics Calculation ---

                results = Results(
                    run_id=run_id,
                    pipeline_id=pipeline.id,
                    timestamp_start=start_time.isoformat(),
                    timestamp_end=end_time.isoformat(),
                    identity_id=pipeline.identity_id,
                    ethical_guidance_id=pipeline.ethical_guidance_id,
                    guardrail_ids_active=active_guardrail_ids,
                    interactions=interactions,
                    outcome=final_outcome,
                    outcome_details=outcome_details,
                    violations=violations,
                    metrics=final_metrics,
                    schema_version=pipeline.schema_version
                )

                logger.info(f"Finished pipeline run: {run_id}. Outcome: {final_outcome}")
                self.save_results(results) # Save results

                return results

    def save_results(self, results: Results):
        """Saves the pipeline results to a JSON file."""
        try:
            os.makedirs(settings.results_dir, exist_ok=True)
            results_file = os.path.join(settings.results_dir, f"{results.run_id}.json")
            with open(results_file, 'w', encoding='utf-8') as f:
                # Use Pydantic's model_dump_json for proper serialization
                f.write(results.model_dump_json(indent=2))
            logger.info(f"Results saved to {results_file}")
        except Exception as e:
            logger.error(f"Failed to save results for run {results.run_id}: {e}", exc_info=True)

    # --- Helper methods to access loaded configs ---
    def get_pipeline(self, pipeline_id: str) -> Optional[Pipeline]:
        return self._pipelines.get(pipeline_id)

    def list_pipeline_ids(self) -> List[str]:
        return list(self._pipelines.keys())

    def get_identity(self, identity_id: str) -> Optional[Identity]:
        """Retrieves a loaded identity by its ID."""
        return self._identities.get(identity_id)

    def get_guidance(self, guidance_id: str) -> Optional[EthicalGuidance]:
        """Retrieves a loaded ethical guidance by its ID."""
        return self._guidances.get(guidance_id)

    # Add similar getters for guardrails if needed
    def get_guardrail(self, guardrail_id: str) -> Optional[Guardrail]:
        """Retrieves a loaded guardrail by its ID."""
        return self._guardrails.get(guardrail_id)

    async def generate_evaluation_response(
        self,
        prompt: str,
        identity: Identity,
        guidance: EthicalGuidance,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generates a response from the LLM for evaluation purposes, using specific identity and guidance.
        This is a simplified version of the LLM stage handler, focused on a single turn.

        Args:
            prompt: The user prompt to evaluate
            identity: The identity profile to use
            guidance: The ethical guidance framework
            system_prompt: Optional system prompt override for formatting instructions
        """
        logger.debug(f"Generating evaluation response with Identity: {identity.id}, Guidance: {guidance.id}")

        if not AUTOGEN_AVAILABLE or not llm_config:
            # Fallback to simple_llm when autogen is not available
            logger.warning("Autogen not available, falling back to simple_llm for evaluation.")
            try:
                from core.simple_llm import simple_llm_call, get_llm_config
                llm_cfg = get_llm_config()
                logger.info(f"simple_llm fallback: provider={llm_cfg.provider}, model={llm_cfg.model}, key_set={bool(llm_cfg.api_key)}")
                sys_msg = system_prompt or "You are a helpful AI assistant."
                if identity and identity.description:
                    sys_msg += f" Context: {identity.description}"
                result = await simple_llm_call(
                    prompt=prompt,
                    system_prompt=sys_msg,
                    config=llm_cfg,
                )
                logger.info(f"simple_llm fallback result: [{result[:100]}]")
                return result
            except Exception as e:
                logger.error(f"simple_llm fallback failed: {e}", exc_info=True)
                return f"Error: LLM configuration unavailable and fallback failed: {e}"

        # --- Prepare System Message ---
        # Use provided system_prompt if given, otherwise build from identity/guidance
        if system_prompt:
            system_message = system_prompt
            # Still append identity context if available
            if identity.description:
                system_message += f" Context: {identity.description}"
        else:
            system_message = "You are a helpful AI assistant." # Base message
            if guidance.prompt_template:
                system_message += f" {guidance.prompt_template}"
            if identity.description:
                system_message += f" You are interacting with/considering the perspective of: {identity.description}."
                if identity.notes:
                    system_message += f" Keep in mind: {identity.notes}"
        logger.debug(f"Evaluation LLM - System Message: {system_message}")
        logger.debug(f"Evaluation LLM - User Prompt: {prompt}")

        # --- Configure Agent (Basic ReasoningAgent) ---
        agent_name = f"eval_agent_{time.time_ns()}" # Unique name
        agent_llm_config = llm_config.model_copy(deep=True) # Use default config

        agent: Optional[ReasoningAgent] = None
        try:
            # Use ReasoningAgent with max_depth=0 for a direct call
            reason_config = {"max_depth": 0}
            llm_config_dict = agent_llm_config.model_dump() if agent_llm_config else None
            agent = ReasoningAgent(
                    name=agent_name,
                    system_message=system_message,
                    llm_config=llm_config_dict,
                    reason_config=reason_config,
                    silent=True
            )
        except Exception as e:
            logger.error(f"Error instantiating evaluation agent: {e}", exc_info=True)
            return f"Error: Agent instantiation failed: {e}"

        # --- Execute Agent Call Asynchronously ---
        raw_response_content = "Error: Evaluation agent execution failed."
        dummy_io = io.StringIO()

        if agent:
            try:
                async with semaphore: # Respect concurrency limits
                    with redirect_stdout(dummy_io), redirect_stderr(dummy_io):
                        try:
                            reply = await asyncio.wait_for(
                                asyncio.to_thread(
                                    agent.generate_reply,
                                    messages=[{"role": "user", "content": prompt}],
                                    sender=None
                                ),
                                timeout=AGENT_TIMEOUT # Use configured timeout
                            )
                        except asyncio.TimeoutError:
                            logger.error(f"Evaluation agent call timed out after {AGENT_TIMEOUT} seconds.")
                            raise # Re-raise to be caught below

                        if isinstance(reply, str):
                            raw_response_content = reply.strip()
                        else:
                            logger.warning(f"Unexpected reply type from evaluation agent: {type(reply)}. Content: {reply}")
                            raw_response_content = str(reply).strip()

                captured_output = dummy_io.getvalue()
                if captured_output:
                    logger.warning(f"Captured stdio during evaluation agent run: {captured_output}")

            except asyncio.TimeoutError:
                raw_response_content = f"Error: Evaluation agent timed out."
            except Exception as e:
                logger.error(f"Error during evaluation agent execution: {e}", exc_info=True)
                raw_response_content = f"Error: Evaluation agent execution failed: {e}"

        logger.debug(f"Evaluation LLM - Raw Response: {raw_response_content}")
        return raw_response_content
