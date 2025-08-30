import os
import logging
from pathlib import Path
from cryptography.fernet import Fernet, InvalidToken

# Use a dedicated logger for this module for better traceability
logger = logging.getLogger(__name__)

class SecurityManager:
    """
    Handles the generation, loading, and use of encryption keys for securing
    sensitive project assets. This version is hardened with robust error handling,
    secure file permissions, and non-destructive operations.
    """

    @staticmethod
    def generate_key(key_path: str) -> None:
        """
        Generates a new Fernet key and saves it to a file with secure permissions (600).
        Ensures the parent directory exists before writing.
        """
        try:
            key_file = Path(key_path)
            # Ensure the directory exists to prevent errors
            key_file.parent.mkdir(parents=True, exist_ok=True)

            key = Fernet.generate_key()
            # Atomic write for resilience
            tmp_path = key_file.with_suffix(key_file.suffix + '.tmp')
            with open(tmp_path, "wb") as f:
                f.write(key)
            os.replace(tmp_path, key_file)
            os.chmod(key_file, 0o600)

            logger.info(f"New encryption key generated and saved securely to {key_path}")
        except IOError as e:
            logger.error(f"Failed to write key file to '{key_path}': {e}", exc_info=True)
            raise

    @staticmethod
    def load_key(key_path: str) -> Fernet:
        """
        Loads the encryption key and initializes a Fernet instance for use.
        Verifies length and format for robustness.
        """
        try:
            with open(key_path, "rb") as key_file:
                key = key_file.read()
                # Validate Fernet key length
                if len(key) != 44:
                    raise ValueError("Fernet key must be 44 bytes base64.")
                return Fernet(key)
        except FileNotFoundError:
            logger.error(f"FATAL: Encryption key not found at '{key_path}'. Cannot proceed.")
            raise
        except (ValueError, TypeError) as e:
            logger.error(f"FATAL: Key file at '{key_path}' is corrupt or not a valid Fernet key: {e}", exc_info=True)
            raise

    @staticmethod
    def encrypt_file(file_path: str, key_path: str) -> bool:
        """
        Encrypts a file in place using the provided key.
        Returns True on success, False on failure.
        """
        logger.info(f"Attempting to encrypt file: '{file_path}'")
        try:
            fernet = SecurityManager.load_key(key_path)

            with open(file_path, "rb") as file:
                file_data = file.read()

            encrypted_data = fernet.encrypt(file_data)

            # Atomic write to prevent partial file corruption
            tmp_path = Path(file_path).with_suffix(Path(file_path).suffix + '.enc_tmp')
            with open(tmp_path, "wb") as file:
                file.write(encrypted_data)
            os.replace(tmp_path, file_path)

            logger.info(f"File '{file_path}' has been successfully encrypted.")
            return True
        except Exception as e:
            logger.error(f"An unexpected error occurred during encryption of '{file_path}': {e}", exc_info=True)
            return False

    @staticmethod
    def decrypt_file_safely(file_path: str, key_path: str) -> bool:
        """
        Decrypts a file non-destructively. The original file is only replaced upon
        successful decryption. This prevents data loss on failure.

        Returns:
            True if decryption was successful, False otherwise.
        """
        logger.info(f"Attempting to safely decrypt file: '{file_path}'")
        original_file = Path(file_path)
        decrypted_temp_file = original_file.with_suffix(original_file.suffix + '.decrypted_tmp')

        try:
            fernet = SecurityManager.load_key(key_path)

            with open(original_file, "rb") as file:
                encrypted_data = file.read()

            # Specifically catch the most common crypto error
            decrypted_data = fernet.decrypt(encrypted_data)

            # Non-destructive write to a temporary file first
            with open(decrypted_temp_file, "wb") as file:
                file.write(decrypted_data)

            # Only replace the original if the entire process was successful
            os.replace(decrypted_temp_file, original_file)

            logger.info(f"File '{file_path}' has been successfully decrypted.")
            return True

        except FileNotFoundError:
            logger.error(f"Cannot decrypt: File not found at '{file_path}'.")
            return False
        except InvalidToken:
            logger.error(f"DECRYPTION FAILED for '{file_path}': The key is incorrect or the data is corrupt.")
            if decrypted_temp_file.exists():
                os.remove(decrypted_temp_file)
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred during decryption of '{file_path}': {e}", exc_info=True)
            if decrypted_temp_file.exists():
                os.remove(decrypted_temp_file)
            return False
