#!/usr/bin/env bash
set -ex

# cd to the directory of this script
cd "$(dirname "${BASH_SOURCE[0]}")"

# cd to the svelte app
cd ../ui

# run in dev mode with hot reload for convenience
npm install
npm run dev
