import hashlib
import uuid


class GeneratorUtility:

    @staticmethod
    def generate_uuid_v4(seed: str = "") -> uuid.UUID:
        if not seed:
            return uuid.uuid4()
        hash_object: hashlib._Hash = hashlib.sha256(seed.encode('utf-8'))
        hash_bytes: bytes = hash_object.digest()[:16]
        return uuid.UUID(bytes=hash_bytes, version=4)