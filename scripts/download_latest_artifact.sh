#!/usr/bin/env bash
set -euo pipefail

# Downloads the latest 'generated-output' artifact for the 'generate.yml' workflow
# Requires: gh CLI authenticated (gh auth login)

WORKFLOW=generate.yml
ARTIFACT_NAME=generated-output
OUT_DIR=./artifacts

echo "Finding latest run for workflow $WORKFLOW..."
RUN_ID=$(gh run list --workflow "$WORKFLOW" --limit 5 --json databaseId,headBranch,status --jq '.[0].databaseId')
if [ -z "$RUN_ID" ] || [ "$RUN_ID" = "null" ]; then
  echo "No recent workflow runs found for $WORKFLOW"
  exit 1
fi

echo "Downloading artifact '$ARTIFACT_NAME' from run $RUN_ID..."
mkdir -p "$OUT_DIR"
gh run download "$RUN_ID" --name "$ARTIFACT_NAME" --dir "$OUT_DIR"
ZIP=$(ls -1 "$OUT_DIR"/*.zip 2>/dev/null | head -n1 || true)
if [ -z "$ZIP" ]; then
  echo "No artifact zip found in $OUT_DIR"
  exit 1
fi

echo "Unzipping $ZIP into $OUT_DIR..."
unzip -o "$ZIP" -d "$OUT_DIR"
echo "Result saved to $OUT_DIR/result.txt"
cat "$OUT_DIR/result.txt"