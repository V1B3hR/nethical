import os
import logging
import stat
import fcntl
import pwd
import grp
import errno
from pathlib import Path
from typing import Optional, List, Any
from cryptography.fernet import Fernet, InvalidToken, MultiFernet

# Use a dedicated logger for this module for better traceability
logger = logging.getLogger(__name__)

MAGIC_HEADER = b'FEN1'
CHUNK_SIZE = 1024 * 1024  # 1 MB per chunk

class SecurityError(Exception):
    pass

class AlreadyEncryptedError(SecurityError):
    pass

class NotEncryptedError(SecurityError):
    pass

class KeyPermissionWarning(Warning):
    pass

def _acquire_file_lock(fileobj):
    try:
        fcntl.flock(fileobj, fcntl.LOCK_EX)
    except Exception as e:
        logger.warning(f"Could not acquire file lock: {e}")

def _release_file_lock(fileobj):
    try:
        fcntl.flock(fileobj, fcntl.LOCK_UN)
    except Exception as e:
        logger.warning(f"Could not release file lock: {e}")

def _preserve_metadata(src_path: Path, dst_path: Path):
    st = src_path.stat()
    os.chmod(dst_path, st.st_mode)
    try:
        os.chown(dst_path, st.st_uid, st.st_gid)
    except Exception:
        pass  # Running as non-root, ignore
    os.utime(dst_path, (st.st_atime, st.st_mtime))

def _warn_if_permissive(path: Path):
    st = path.stat()
    if (st.st_mode & 0o077):
        logger.warning(f"Key file {path} has overly permissive permissions: {oct(st.st_mode)}")

class SecurityManager:
    """
    Improved SecurityManager with idempotency, metadata preservation,
    chunked encryption, key rotation, and concurrency safety.
    """

    @staticmethod
    def generate_key(key_path: str) -> None:
        key_file = Path(key_path)
        key_file.parent.mkdir(parents=True, exist_ok=True)
        key = Fernet.generate_key()
        tmp_path = key_file.with_suffix(key_file.suffix + '.tmp')
        with open(tmp_path, "wb") as f:
            f.write(key)
        os.replace(tmp_path, key_file)
        os.chmod(key_file, 0o600)
        logger.info(f"New encryption key generated and saved securely to {key_path}")

    @staticmethod
    def load_key(key_path: str, extra_keys: Optional[List[bytes]] = None) -> MultiFernet:
        with open(key_path, "rb") as key_file:
            key = key_file.read().strip()
            if len(key) != 44:
                raise ValueError("Fernet key must be 44 bytes base64.")
            _warn_if_permissive(Path(key_path))
            keys = [Fernet(key)]
            if extra_keys:
                for ek in extra_keys:
                    ek = ek.strip()
                    if len(ek) == 44:
                        keys.append(Fernet(ek))
            return MultiFernet(keys)

    @staticmethod
    def is_encrypted(file_path: str) -> bool:
        with open(file_path, "rb") as f:
            header = f.read(len(MAGIC_HEADER))
            return header == MAGIC_HEADER

    @staticmethod
    def encrypt_file(file_path: str, key_path: str, output_path: Optional[str] = None, extra_keys: Optional[List[bytes]] = None) -> None:
        src_path = Path(file_path)
        dst_path = Path(output_path) if output_path else src_path

        if SecurityManager.is_encrypted(str(src_path)):
            logger.warning(f"File '{src_path}' is already encrypted.")
            raise AlreadyEncryptedError(f"File '{src_path}' is already encrypted.")

        fernet = SecurityManager.load_key(key_path, extra_keys=extra_keys)
        tmp_path = dst_path.with_suffix(dst_path.suffix + '.enc_tmp')

        with open(src_path, "rb") as infile, open(tmp_path, "wb") as outfile:
            _acquire_file_lock(outfile)
            outfile.write(MAGIC_HEADER)
            while True:
                chunk = infile.read(CHUNK_SIZE)
                if not chunk:
                    break
                ciphertext = fernet.encrypt(chunk)
                chunk_len = len(ciphertext).to_bytes(4, 'big')
                outfile.write(chunk_len)
                outfile.write(ciphertext)
            _release_file_lock(outfile)

        _preserve_metadata(src_path, tmp_path)
        os.replace(tmp_path, dst_path)
        logger.info(f"File '{file_path}' has been successfully encrypted to '{dst_path}'.")

    @staticmethod
    def decrypt_file_safely(file_path: str, key_path: str, output_path: Optional[str] = None, extra_keys: Optional[List[bytes]] = None) -> None:
        src_path = Path(file_path)
        dst_path = Path(output_path) if output_path else src_path
        tmp_path = dst_path.with_suffix(dst_path.suffix + '.decrypted_tmp')

        with open(src_path, "rb") as infile:
            header = infile.read(len(MAGIC_HEADER))
            if header != MAGIC_HEADER:
                logger.error(f"File '{file_path}' is not encrypted with expected header.")
                raise NotEncryptedError(f"File '{file_path}' is not encrypted with expected header.")

            fernet = SecurityManager.load_key(key_path, extra_keys=extra_keys)
            with open(tmp_path, "wb") as outfile:
                _acquire_file_lock(outfile)
                while True:
                    chunk_len_bytes = infile.read(4)
                    if not chunk_len_bytes:
                        break
                    chunk_len = int.from_bytes(chunk_len_bytes, 'big')
                    chunk_ciphertext = infile.read(chunk_len)
                    try:
                        plaintext = fernet.decrypt(chunk_ciphertext)
                    except InvalidToken:
                        logger.error("DECRYPTION FAILED: The key is incorrect or the data is corrupt.")
                        os.remove(tmp_path)
                        raise
                    outfile.write(plaintext)
                _release_file_lock(outfile)

        _preserve_metadata(src_path, tmp_path)
        os.replace(tmp_path, dst_path)
        logger.info(f"File '{file_path}' has been successfully decrypted to '{dst_path}'.")

    @staticmethod
    def encrypt_file_to(file_path: str, key_path: str, output_path: str, extra_keys: Optional[List[bytes]] = None) -> None:
        SecurityManager.encrypt_file(file_path, key_path, output_path=output_path, extra_keys=extra_keys)

    @staticmethod
    def decrypt_file_to(file_path: str, key_path: str, output_path: str, extra_keys: Optional[List[bytes]] = None) -> None:
        SecurityManager.decrypt_file_safely(file_path, key_path, output_path=output_path, extra_keys=extra_keys)
