#!/usr/bin/env bash
set -ex

# cd to the directory of this script
cd "$(dirname "${BASH_SOURCE[0]}")"

# https://fastapi.tiangolo.com/tutorial/first-steps/
uvicorn app:app --reload --port 8000 --host 0.0.0.0
