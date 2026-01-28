# Developer Guide

This repository contains the EthicsEngine API and evaluation suite.

## Setup
1. Use **Python 3.10+**.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the API
- Launch the main server:
  ```bash
  python server_start.py
  ```
  The API listens on `http://127.0.0.1:8080` by default.
- Launch the batch API:
  ```bash
  uvicorn batch_api.main_api:app --reload
  ```

## Running Tests
After modifying code, run the test suite from the repository root:
```bash
python -m pytest -q
```
All tests should pass before committing changes.

