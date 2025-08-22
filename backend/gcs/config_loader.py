import yaml
import logging

def load_config(path: str) -> dict:
    """Loads the YAML configuration file with robust error handling."""
    try:
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
            logging.info("Configuration file loaded successfully.")
            return config
    except FileNotFoundError:
        logging.error(f"FATAL: Configuration file not found at {path}. Please ensure it exists.")
        raise
    except yaml.YAMLError as e:
        logging.error(f"FATAL: Error parsing YAML configuration file: {e}")
        raise
