name: Fetch Sports Odds Data

on:
  schedule:
    - cron: "0 6 * * *"
  workflow_dispatch:
    inputs:
      mode:
        description: "Fetch mode"
        required: false
        default: "both"
        type: choice
        options:
          - both
          - current-only
          - historical-only
      reset_checkpoint:
        description: "Reset checkpoint and refetch from scratch"
        required: false
        default: false
        type: boolean

jobs:
  fetch:
    runs-on: ubuntu-latest
    timeout-minutes: 340
    permissions:
      contents: write
    env:
      ODDS_API_KEY: ${{ secrets.ODDS_API_KEY }}
      PYTHONIOENCODING: utf-8

    steps:
      - name: Checkout
        uses: actions/checkout@v4
        with:
          token: ${{ secrets.GITHUB_TOKEN }}

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip

      - name: Install dependencies
        run: pip install aiohttp aiofiles pandas numpy

      - name: Git config
        run: |
          git config user.name  "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"

      - name: Reset checkpoint
        if: ${{ github.event.inputs.reset_checkpoint == 'true' }}
        run: rm -f odds_data/checkpoint.json

      - name: Fetch current odds
        if: ${{ github.event.inputs.mode != 'historical-only' }}
        run: python odds.py --current-only
        continue-on-error: true

      - name: Commit current odds
        if: ${{ github.event.inputs.mode != 'historical-only' }}
        continue-on-error: true
        run: |
          git add odds_data/
          git diff --cached --quiet && exit 0
          git commit -m "data: current odds $(date -u '+%Y-%m-%d %H:%M UTC')"
          git push

      - name: Fetch historical odds
        if: ${{ github.event.inputs.mode != 'current-only' }}
        run: python odds.py --historical-only
        continue-on-error: true

      - name: Commit historical + ML
        continue-on-error: true
        run: |
          git pull --rebase origin main
          git add odds_data/
          git diff --cached --quiet && echo "No new data." && exit 0
          TASKS=$(python -c "
          import json
          from pathlib import Path
          cp = Path('odds_data/checkpoint.json')
          print(len(json.loads(cp.read_text()).get('done', [])) if cp.exists() else 0)
          ")
          git commit -m "data: historical $(date -u '+%Y-%m-%d %H:%M UTC') | ${TASKS} tasks done"
          git push

      - name: Build ML tables
        run: python odds.py --ml-only
        continue-on-error: true

      - name: Commit ML tables
        continue-on-error: true
        run: |
          git pull --rebase origin main
          git add odds_data/
          git diff --cached --quiet && echo "No new ML data." && exit 0
          git commit -m "ml: tables rebuilt $(date -u '+%Y-%m-%d %H:%M UTC')"
          git push
