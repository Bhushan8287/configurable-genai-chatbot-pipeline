"""
Main script for executing the pipeline.

This script initializes and executes the full GenAI pipeline workflow defined
in the `src.pipeline` module.

Run this script directly to launch the end-to-end pipeline.
"""

from src.pipeline import run_pipeline

if __name__ == '__main__':
    # Execute the main pipeline when this script is run directly
    run_pipeline()