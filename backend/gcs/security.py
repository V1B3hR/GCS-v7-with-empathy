import logging
from cryptography.fernet import Fernet

def generate_key(key_path: str):
    """Generates and saves a new encryption key."""
    key = Fernet.generate_key()
    with open(key_path, "wb") as key_file:
        key_file.write(key)
    logging.info(f"New encryption key generated and saved to {key_path}")

def load_key(key_path: str) -> bytes:
    """Loads the encryption key from a file."""
    try:
        with open(key_path, "rb") as key_file:
            return key_file.read()
    except FileNotFoundError:
        logging.error(f"Encryption key not found at {key_path}. Please generate one.")
        raise

def encrypt_file(file_path: str, key_path: str):
    """Encrypts a file (e.g., a trained model) using the provided key."""
    key = load_key(key_path)
    f = Fernet(key)
    with open(file_path, "rb") as file:
        file_data = file.read()
    encrypted_data = f.encrypt(file_data)
    with open(file_path, "wb") as file:
        file.write(encrypted_data)
    logging.info(f"File '{file_path}' has been encrypted.")

def decrypt_file(file_path: str, key_path: str):
    """Decrypts a file using the provided key."""
    key = load_key(key_path)
    f = Fernet(key)
    with open(file_path, "rb") as file:
        encrypted_data = file.read()
    decrypted_data = f.decrypt(encrypted_data)
    with open(file_path, "wb") as file:
        file.write(decrypted_data)
    logging.info(f"File '{file_path}' has been decrypted.")
