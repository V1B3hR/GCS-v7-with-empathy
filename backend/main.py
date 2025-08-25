import argparse
import logging
import os
import time
import numpy as np

# --- GCS Core Modules ---
from gcs.config_loader import load_config
from gcs.training import Trainer
from gcs.closed_loop_agent import ClosedLoopAgent

def configure_logging():
    """Sets up a clean, standardized logging format."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def run_live_simulation(config: dict):
    """
    Initiates and runs the main operational loop for the Closed-Loop Agent.
    This function simulates a live data stream from hardware.
    """
    try:
        # --- 1. Initialize the Master Agent ---
        # The agent handles loading all necessary models and controllers.
        agent = ClosedLoopAgent(config)

        # --- 2. Load Static Assets ---
        # The anatomical graph is loaded once, as it does not change.
        graph_data = np.load(config["graph_scaffold_path"])
        adj_matrix = np.expand_dims(graph_data['adjacency_matrix'], 0)

        logging.info("--- GCS v7.0 Active Interface: ENGAGED ---")
        logging.info("Starting closed-loop SENSE -> DECIDE -> ACT -> LEARN cycle.")
        logging.info("Simulating live data stream. Press Ctrl+C to terminate.")

        # --- 3. The Main Real-Time Loop ---
        while True:
            # --- SIMULATING LIVE DATA FROM HARDWARE ---
            # In a real system, this dictionary would be populated by the LiveConnector
            # after receiving and preprocessing data from an OpenBCI headset, a Polar
            # heart monitor, and a microphone.
            live_data = {
                "source_eeg": np.random.randn(1, config["cortical_nodes"], config["timesteps"]),
                "adj_matrix": adj_matrix,
                "physio": np.random.randn(1, config["physio_features"]),
                "voice": np.random.randn(1, 128) # Placeholder for voice features
            }
            
            # --- 4. Run a Single Cycle of the Agent ---
            # The agent encapsulates the entire Sense->Decide->Act->Learn logic.
            agent.run_cycle(live_data)
            
            # Control the loop speed to simulate a real-time system.
            time.sleep(2)

    except KeyboardInterrupt:
        logging.info("\n--- Closed-loop session terminated by user. ---")
    except Exception as e:
        logging.error(f"A critical error occurred in the main loop: {e}", exc_info=True)

def main():
    """
    The main entry point for the Grand Council Sentient backend.
    Parses command-line arguments to determine the operational mode.
    """
    configure_logging()

    parser = argparse.ArgumentParser(
        description="Grand Council Sentient - The central orchestrator for the BCI system.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'mode',
        choices=['train-foundational', 'train-affective', 'run-closed-loop'],
        help="""The operational mode:
  - train-foundational: Run the Leave-One-Subject-Out cross-validation to train the core GNN models.
  - train-affective:    Train the multi-modal classifier for empathetic understanding.
  - run-closed-loop:    Run the live, interactive agent with simulated data."""
    )
    args = parser.parse_args()

    # --- Load the master configuration from the project's root directory ---
    try:
        # This path navigates up one level from /backend to the project root
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
        config = load_config(config_path)
    except Exception:
        logging.error("Failed to load configuration. Exiting.")
        return

    # --- Execute the selected mode ---
    if args.mode == 'train-foundational':
        logging.info("MODE: Training Foundational Model Initiated")
        trainer = Trainer(config)
        trainer.run_loso_cross_validation()
    
    elif args.mode == 'train-affective':
        logging.info("MODE: Training Affective Model Initiated")
        trainer = Trainer(config)
        trainer.train_affective_model()

    elif args.mode == 'run-closed-loop':
        logging.info("MODE: Live Closed-Loop Simulation Initiated")
        run_live_simulation(config)

if __name__ == "__main__":
    main()
