import streamlit as st
import requests
import ujson
import os
import sys
from datetime import datetime
import pandas as pd # For better table display

# Add project root to sys.path to allow importing sibling modules like 'open_llm'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from open_llm.config_llm import LLMSetter
# --- Configuration ---
DEFAULT_API_BASE_URL = "http://127.0.0.1:8080" # Assuming the FastAPI server runs locally on port 8000

# --- Helper Functions for API Calls ---

# Initialize session state for API URL
if 'api_base_url' not in st.session_state:
    st.session_state.api_base_url = DEFAULT_API_BASE_URL

# Use this function to get the current API URL
def get_api_base_url():
    """Get the current API base URL from session state."""
    return st.session_state.api_base_url

def check_api_connection(base_url=None):
    """
    Check if the API server is running and responsive.
    
    Args:
        base_url: The base URL of the API server
        
    Returns:
        A tuple (is_connected, message)
    """
    if base_url is None:
        base_url = get_api_base_url()
        
    try:
        # Try a basic health endpoint first
        health_url = f"{base_url}/health"
        health_response = requests.get(health_url, timeout=5)
        
        if health_response.status_code == 200:
            return True, f"API server is running at {base_url}"
        
        # If health endpoint fails, try the root endpoint
        root_response = requests.get(base_url, timeout=5)
        if root_response.status_code == 200:
            return True, f"API server root endpoint is responding at {base_url}"
            
        return False, f"API server responded with status code {health_response.status_code}"
        
    except requests.exceptions.ConnectionError:
        return False, f"Could not connect to API server at {base_url}. The server might not be running."
    except requests.exceptions.Timeout:
        return False, f"Connection to API server at {base_url} timed out."
    except Exception as e:
        return False, f"Error checking API connection: {str(e)}"

    
def handle_api_error(response, context="API call"):
    """Handles common API errors and displays messages."""
    try:
        response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
        return True  # Indicate success
    except requests.exceptions.HTTPError as http_err:
        try:
            detail = response.json().get("detail", str(http_err))
        except ujson.JSONDecodeError:
            detail = str(http_err)
        st.error(f"{context} failed ({response.status_code}): {detail}")
    except requests.exceptions.ConnectionError as conn_err:
        st.error(f"{context} failed: Could not connect to API at {DEFAULT_API_BASE_URL}. Is the backend running?")
    except requests.exceptions.RequestException as req_err:
        st.error(f"{context} failed: {req_err}")
    return False  # Indicate failure

@st.cache_data(ttl=60)  # Cache for 1 minute
def get_definitions(max_retries=1):
    """Fetches the list of pipeline definition IDs with retry support."""
    url = f"{get_api_base_url()}/pipelines/definitions"
    
    for attempt in range(max_retries + 1):
        try:
            response = requests.get(url, timeout=10)
            if handle_api_error(response, "Fetching pipeline definitions"):
                return response.json().get("pipelines", [])
            elif attempt < max_retries:
                st.warning(f"Retry {attempt+1}/{max_retries} for fetching definitions...")
        except Exception as e:
            st.error(f"Error fetching definitions: {e}")
            if attempt < max_retries:
                st.warning(f"Retry {attempt+1}/{max_retries} for fetching definitions...")
    
    return []

@st.cache_data(ttl=60)
def get_definition_details(pipeline_id: str):
    """Fetches the details of a specific pipeline definition."""
    # Need to handle potential '/' in pipeline_id for the URL
    # Requests library handles URL encoding automatically if needed, but FastAPI path converter expects raw path
    url = f"{get_api_base_url()}/pipelines/definitions/{pipeline_id}"
    try:
        response = requests.get(url, timeout=10)
        if handle_api_error(response, f"Fetching details for '{pipeline_id}'"):
            return response.json()
    except Exception as e:
        st.error(f"Error fetching definition details for '{pipeline_id}': {e}")
    return None

def run_pipeline_api(pipeline_name: str, num_runs: int):
    """Calls the API to run a pipeline."""
    url = f"{get_api_base_url()}/pipelines/{pipeline_name}/run?num_runs={num_runs}"
    try:
        response = requests.post(url, timeout=15) # Longer timeout for triggering runs
        if handle_api_error(response, f"Running pipeline '{pipeline_name}'"):
            return response.json() # Should contain {"message": "...", "run_ids": [...]}
    except Exception as e:
        st.error(f"Error running pipeline '{pipeline_name}': {e}")
    return None

@st.cache_data(ttl=10) # Short cache for potentially changing results
def get_results_list():
    """Fetches the list of available result run IDs."""
    url = f"{get_api_base_url()}/results"
    try:
        response = requests.get(url, timeout=10)
        if handle_api_error(response, "Fetching results list"):
            # API returns IDs *without* prefix, prepend 'run_' for consistency in UI
            ids_without_prefix = response.json().get("results", [])
            return [f"run_{id_}" for id_ in ids_without_prefix]
    except Exception as e:
        st.error(f"Error fetching results list: {e}")
    return []

@st.cache_data(ttl=5) # Very short cache for status
def get_run_status(run_id: str):
    """Fetches the status of a specific run."""
    url = f"{get_api_base_url()}/pipelines/status/{run_id}"
    try:
        response = requests.get(url, timeout=10)
        # Status might be 404 if not found/running yet, handle this gracefully
        if response.status_code == 404:
            return {"run_id": run_id, "status": "NOT_FOUND"}
        if handle_api_error(response, f"Fetching status for run '{run_id}'"):
            return response.json()
    except Exception as e:
        # Don't show error if it's just not found yet
        if "404" not in str(e):
             st.error(f"Error fetching status for run '{run_id}': {e}")
    return None

@st.cache_data(ttl=30)
def get_run_logs(run_id: str):
    """Fetches the logs for a specific run."""
    url = f"{get_api_base_url()}/pipelines/logs/{run_id}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 404:
            return [] # No logs found is not necessarily an error here
        if handle_api_error(response, f"Fetching logs for run '{run_id}'"):
            return response.json().get("logs", [])
    except Exception as e:
        st.error(f"Error fetching logs for run '{run_id}': {e}")
    return None

@st.cache_data(ttl=30)
def get_run_results(run_id: str):
    """Fetches the full results for a specific run."""
    # API expects run_id without 'run_' prefix for this endpoint
    run_id_short = run_id.replace("run_", "")
    url = f"{get_api_base_url()}/results/{run_id_short}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 404:
            return None # Result file doesn't exist (yet)
        if handle_api_error(response, f"Fetching results for run '{run_id}'"):
            return response.json()
    except Exception as e:
        st.error(f"Error fetching results for run '{run_id}': {e}")
    return None

def validate_pipeline_api(pipeline_json: str):
    """Calls the API to validate a pipeline definition."""
    url = f"{get_api_base_url()}/pipelines/validate"
    try:
        pipeline_data = ujson.loads(pipeline_json)
        response = requests.post(url, json=pipeline_data, timeout=10)
        # Validate endpoint returns 200 on success, 400/422 on failure
        if response.status_code == 200:
            st.success("Validation successful: " + response.json().get("message", ""))
            return True
        else:
            # Use handle_api_error for consistent error display
            handle_api_error(response, "Pipeline validation")
            return False
    except ujson.JSONDecodeError:
        st.error("Invalid JSON format.")
        return False
    except Exception as e:
        st.error(f"Error during validation request: {e}")
        return False

def create_pipeline_api(pipeline_json: str):
    """Calls the API to create a new pipeline definition file."""
    url = f"{get_api_base_url()}/pipelines/create"
    try:
        pipeline_data = ujson.loads(pipeline_json)
        response = requests.post(url, json=pipeline_data, timeout=10)
        if handle_api_error(response, "Creating pipeline definition"):
            st.success("Pipeline definition created successfully!")
            return response.json() # Contains {"message": "...", "pipeline_id": "..."}
    except Exception as e:
        st.error(f"Error creating pipeline definition: {e}")
    return None

@st.cache_data(ttl=5)
def get_concurrency_status():
    """Fetches the server concurrency status."""
    url = f"{get_api_base_url()}/server/concurrency"
    try:
        response = requests.get(url, timeout=5)
        if handle_api_error(response, "Fetching concurrency status"):
            return response.json()
    except Exception as e:
        st.error(f"Error fetching concurrency status: {e}")
    return None

# --- UI Sections ---

def display_server_status():
    """Display server connection status in the sidebar."""
    st.sidebar.header("🖥️ Server Status")

    is_connected, message = check_api_connection()
    
    if is_connected:
        st.sidebar.success(message)
    else:
        st.sidebar.error(message)
        st.sidebar.warning("⚠️ The application may not function correctly without a connection to the API server.")
    
    # Add server configuration options
    with st.sidebar.expander("Server Configuration"):
        new_api_url = st.text_input("API Base URL", value=get_api_base_url())
        if st.button("Update API URL"):
            # Update the API_BASE_URL through session state instead of global
            st.session_state.api_base_url = new_api_url
            st.rerun()  # Rerun the app with the new URL
            
def display_dashboard():
    st.header("🚀 Pipeline Dashboard")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.write("List of available pipeline definitions.")
    with col2:
        if st.button("Refresh List & Cache"):
            st.cache_data.clear() # Clear ALL streamlit cache data
            st.rerun()

    definitions = get_definitions()

    if not definitions:
        st.warning("No pipeline definitions found or failed to fetch from API.")
        return

    # Use session state to track expanded details
    # Use a dropdown (selectbox) to choose the pipeline
    selected_pipeline_id = st.selectbox(
        "Select Pipeline Definition:",
        definitions,
        index=None, # Default to no selection
        placeholder="Choose a pipeline..."
    )

    if selected_pipeline_id:
        st.markdown("---")
        st.subheader(f"Actions for: `{selected_pipeline_id}`")

        # Actions for the selected pipeline
        col1, col2, col3 = st.columns([1, 1, 3])
        with col1:
            if st.button("View Details", key=f"details_{selected_pipeline_id}"):
                # Use an expander directly instead of session state for simplicity here
                with st.expander("Details", expanded=True):
                    details = get_definition_details(selected_pipeline_id)
                    if details:
                        st.json(details)
                    else:
                        st.error("Could not load details.")

        with col2:
            num_runs = st.number_input("Runs", min_value=1, max_value=10, value=1, key=f"num_{selected_pipeline_id}")

        with col3:
            if st.button("Run Pipeline", key=f"run_{selected_pipeline_id}"):
                 with st.spinner(f"Triggering {num_runs} run(s) for '{selected_pipeline_id}'..."):
                    run_response = run_pipeline_api(selected_pipeline_id, num_runs)
                    if run_response and "run_ids" in run_response:
                        # Display only the run IDs clearly
                        st.success("Pipeline run(s) submitted successfully.")
                        run_ids_str = ", ".join(run_response['run_ids'])
                        st.code(run_ids_str, language=None) # Use st.code for easy copying
                        st.info("You can monitor these runs on the 'Run Monitoring' page.")
                        # Store run IDs for monitoring page
                        if 'submitted_run_ids' not in st.session_state:
                            st.session_state.submitted_run_ids = set() # Use set for uniqueness
                        st.session_state.submitted_run_ids.update(run_response['run_ids'])
                    # Error handling is done within run_pipeline_api

def display_creation():
    st.header("📝 Pipeline Creation & Validation")

    # Initialize session state for text area content
    if 'pipeline_json_input' not in st.session_state:
        st.session_state.pipeline_json_input = """{
    "id": "my_new_pipeline",
    "description": "A description of the new pipeline.",
    "identity_id": "default_identity",
    "ethical_guidance_id": "default_guidance",
    "guardrail_ids": ["default_guardrail_example"],
    "stages": [
        {
            "id": "initial_prompt",
            "type": "interaction",
            "role": "user",
            "content": "Initial user prompt for the scenario.",
            "outputs": {"label": "user_input"}
        },
        {
            "id": "llm_response",
            "type": "llm",
            "inputs": {"prompt": "{user_input}"},
            "outputs": {"label": "llm_output"}
        }
    ],
    "evaluation_metrics": {
        "expected_outcome": "Description of what should happen."
    }
}"""

    uploaded_file = st.file_uploader("Upload Pipeline JSON", type=["json"])
    if uploaded_file is not None:
        try:
            # Read content and update text area
            st.session_state.pipeline_json_input = uploaded_file.getvalue().decode("utf-8")
            st.info("File content loaded into editor.")
        except Exception as e:
            st.error(f"Error reading uploaded file: {e}")

    st.write("Edit Pipeline JSON:")
    pipeline_json = st.text_area("Pipeline Definition (JSON)", value=st.session_state.pipeline_json_input, height=400, key="pipeline_editor")

    # Update session state if editor changes
    st.session_state.pipeline_json_input = pipeline_json

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Validate Definition"):
            with st.spinner("Validating..."):
                validate_pipeline_api(pipeline_json)
    with col2:
        if st.button("Create Definition File"):
            with st.spinner("Creating..."):
                create_response = create_pipeline_api(pipeline_json)
                if create_response:
                    # Optionally refresh dashboard list
                    st.cache_data.clear() # Clear definition list cache


def display_monitoring():
    st.header("📊 Run Monitoring")

    # Combine results list with potentially newly submitted runs
    # Use a set to avoid duplicates if a run finishes quickly
    known_run_ids = set(get_results_list())
    if 'submitted_run_ids' in st.session_state:
        known_run_ids.update(st.session_state.submitted_run_ids)

    if not known_run_ids:
        st.info("No pipeline runs found or submitted yet.")
        return

    # Sort for display (newest first?) - Requires timestamps, which we don't have easily here. Simple alpha sort.
    sorted_run_ids = sorted(list(known_run_ids), reverse=True)

    selected_run_id = st.selectbox("Select Run ID to View Details:", sorted_run_ids, index=None, placeholder="Choose a run...")

    if selected_run_id:
        st.subheader(f"Details for Run: `{selected_run_id}`")

        # Fetch and display status
        status_info = get_run_status(selected_run_id)
        if status_info:
            status = status_info.get("status", "Unknown")
            color = "blue"
            if status == "COMPLETED": color = "green"
            elif status == "ERROR": color = "red"
            elif status == "RUNNING": color = "orange"
            st.metric("Status", status, delta=None, label_visibility="visible") # Use metric for better visibility
            # st.markdown(f"**Status:** :{color}[{status}]")
        else:
            st.warning("Could not fetch status.")

        # Fetch and display logs
        with st.expander("Logs"):
            logs = get_run_logs(selected_run_id)
            if logs:
                st.text("\n".join(logs))
            else:
                st.caption("No logs found for this run ID (yet).")

        # Fetch and display results
        with st.expander("Results", expanded=True):
            results_data = get_run_results(selected_run_id)
            if results_data:
                st.markdown(f"**Outcome:** {results_data.get('outcome', 'N/A')}")
                st.caption(f"Details: {results_data.get('outcome_details', 'N/A')}")

                st.markdown("---")
                st.subheader("Metrics")
                metrics = results_data.get('metrics', {})
                if metrics:
                    m_col1, m_col2, m_col3 = st.columns(3)
                    m_col1.metric("Latency (s)", f"{metrics.get('latency_seconds', 0):.2f}")
                    m_col2.metric("Total Tokens", metrics.get('tokens_used_total', 'N/A'))
                    m_col3.metric("Correctness", f"{metrics.get('correctness', 'N/A')}")
                    st.metric("Ethical Score", f"{metrics.get('ethical_score', 'N/A')}")
                    # TODO: Display other metrics like principle alignment if available
                else:
                    st.caption("No metrics available.")

                st.markdown("---")
                st.subheader("Violations")
                violations = results_data.get('violations', [])
                if violations:
                    for i, v in enumerate(violations):
                        st.warning(f"**Violation {i+1}:** Guardrail '{v.get('guardrail_id')}' triggered at stage '{v.get('stage_id')}'. Reason: {v.get('reason')}")
                else:
                    st.caption("No violations recorded.")

                st.markdown("---")
                st.subheader("Interactions")
                interactions = results_data.get('interactions', [])
                if interactions:
                    for interaction in interactions:
                        role = interaction.get("role", "unknown")
                        content = interaction.get("content", "")
                        metadata = interaction.get("metadata", {})
                        reasoning = metadata.get("reasoning_tree") # Check for reasoning tree

                        with st.chat_message(role):
                            st.markdown(content)
                            if reasoning:
                                # Display reasoning directly, not inside a nested expander
                                st.caption("Reasoning Tree:") # Add a caption for context
                                st.json(reasoning, expanded=False) # Display JSON, default collapsed
                            # Optionally display other metadata
                            # st.caption(f"Stage: {interaction.get('stage_id', 'N/A')}, Tokens: {metadata.get('tokens_used', 'N/A')}")
                else:
                    st.caption("No interactions recorded.")

            elif status_info and status_info.get("status") != "NOT_FOUND":
                 st.info("Results file not found (run may be in progress or failed before saving).")


def display_concurrency():
    st.sidebar.header("🚦 Concurrency Status")
    status = get_concurrency_status()
    if status:
        st.sidebar.metric("LLM Limit", status.get('limit', 'N/A'))
        st.sidebar.metric("Active Calls", status.get('active', 'N/A'))
        st.sidebar.metric("Waiting Calls", status.get('waiting', 'N/A'))
    else:
        st.sidebar.warning("Could not fetch status.")
    if st.sidebar.button("Refresh Status"):
        st.cache_data.clear() # Clear all cache - might be too broad
        st.rerun()

def set_llm():
    """Configure LLM settings from the UI."""
    setter = LLMSetter()

    llm_config = st.selectbox("LLM Configuration", options=["OpenAI", "Ollama"])
    
    if llm_config == "OpenAI":
        st.session_state.llm = "openai"
        
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            api_key = st.text_input("OpenAI API Key", type="password", key="openai_key")
        
        model = st.selectbox(
            "OpenAI Model", 
            options=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"], 
            index=0,
            key="openai_model"
        )
        
        openai_dict = {
            "api_type": "openai",
            "model": model,
            "api_key": api_key,
            "base_url": "https://api.openai.com/v1"
        }
        
        if st.button("Set OpenAI Configuration"):
            if not api_key:
                st.error("API Key is required for OpenAI")
            else:
                config = setter.config_llm(openai_dict)
                if config:
                    st.success(f"Successfully configured {model}")
                    st.session_state.current_config = openai_dict
        
        return openai_dict
    
    elif llm_config == "Ollama":
        st.session_state.llm = "ollama"
        
        base_url = st.text_input(
            "Ollama Base URL", 
            value="http://127.0.0.1:11434/v1",
            key="ollama_url"
        )
        
        # Fetch available models from the Ollama API
        available_models = []
        try:
            models_url = f"{base_url}/models"
            response = requests.get(models_url, timeout=5)
            if response.status_code == 200:
                st.info(f"Successfully connected to Ollama server at {base_url}.")
                raw_response = response.json()
                available_models = [model["id"] for model in raw_response.get("data", [])]
                with st.expander("Extra Info", expanded=False):
                    st.caption("Raw response from server:")
                    st.json(raw_response)
                    st.caption("Parsed models:")
                    st.write(available_models)
            else:
                st.warning(f"Failed to fetch available models from {base_url}. Response code: {response.status_code}")
                st.info(f"Response content: {response.text}")
        except Exception as e:
            st.error(f"Error fetching available models: {e}")

        # Display available models and allow selection
        model = None  # Ensure model is always defined
        if available_models:
            st.write("Available Ollama Models:")
            selected_models = st.multiselect(
                "Select one or more models to use:",
                options=available_models,
                default=available_models[0] if available_models else None,
                key="ollama_models"
            )
            model = selected_models[0] if selected_models else None
        else:
            st.warning("No models found on the Ollama system. Please check the base URL or add models manually.")
            model = st.text_input(
                "Ollama Model (Enter manually if not listed)", 
                value="gemma3:4b-it-q8_0",
                key="ollama_model"
            )
        
        ollama_dict = {
            "api_type": "ollama",
            "base_url": base_url,
            "model": model,
            "api_key": "None"  # Ollama doesn't need an API key
        }
        
        if st.button("Set Ollama Configuration"):
            if not base_url or not model:
                st.error("Base URL and Model are required for Ollama")
            else:
                # Verify the availability of the specified Ollama model
                try:
                    verify_url = f"{base_url}/models/{model}"
                    response = requests.get(verify_url, timeout=5)
                    if response.status_code == 200:
                        config = setter.config_llm(ollama_dict)
                        if config:
                            st.success(f"Successfully configured Ollama with {model}")
                            st.session_state.current_config = ollama_dict
                    else:
                        st.error(f"Failed to verify Ollama model '{model}' at {base_url}. Response code: {response.status_code}")
                except Exception as e:
                    st.error(f"Error verifying Ollama model '{model}': {e}")
        
        return ollama_dict
    
# --- Main App Layout ---
st.set_page_config(layout="wide", page_title="Ethics Engine UI")

st.title("⚖️ Ethics Engine Enterprise UI")

# Display server status in sidebar
display_server_status()

# Initialize session state for navigation
if 'page' not in st.session_state:
    st.session_state.page = "Dashboard"

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Create/Validate", "Run Monitoring"], key="nav_radio", index=["Dashboard", "Create/Validate", "Run Monitoring"].index(st.session_state.page))
with st.sidebar:
    llm_configuration = set_llm()
st.session_state.page = page  # Update session state based on radio button

# Display Concurrency Status in Sidebar
display_concurrency()

# Display selected page
if page == "Dashboard":
    display_dashboard()
elif page == "Create/Validate":
    display_creation()
elif page == "Run Monitoring":
    display_monitoring()
