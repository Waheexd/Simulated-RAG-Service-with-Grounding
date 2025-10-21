#!/usr/bin/env bash
# run_eval_and_tests.sh
echo "Running evals (server must be running at http://127.0.0.1:8000)"
python eval_run.py

echo "Running pytest tests"
pytest -q
