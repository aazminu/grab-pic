import json
import math
import os
import signal
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # pragma: no cover
    pass

MAX_IMAGE_PIXELS = 25_000_000
if Image is not None:
    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

DEEPFACE_TIMEOUT_SECS = 300


class DeepFaceTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise DeepFaceTimeout("DeepFace processing timed out")


def _set_alarm_if_supported(seconds: int) -> None:
    if hasattr(signal, "SIGALRM") and hasattr(signal, "alarm"):
        if seconds > 0:
            signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(seconds)


def _validate_image_dimensions(path: str) -> bool:
    if Image is None:
        return os.path.exists(path)
    try:
        with Image.open(path) as img:
            w, h = img.size
            pixels = w * h
            if pixels > MAX_IMAGE_PIXELS:
                print(f"    -> REJECTED: {w}x{h} = {pixels:,} pixels (limit {MAX_IMAGE_PIXELS:,})")
                return False
            return True
    except Exception as e:
        print(f"    -> Image validation failed: {e}")
        return False


def cosine_distance(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Compute pgvector-compatible cosine distance: 1 - cos(vec_a, vec_b)."""
    if len(vec_a) != len(vec_b) or len(vec_a) == 0:
        raise ValueError("Embedding vectors must be non-empty and of equal dimension")
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(vec_a, vec_b):
        fa = float(a)
        fb = float(b)
        dot += fa * fb
        norm_a += fa * fa
        norm_b += fb * fb
    denom = math.sqrt(norm_a) * math.sqrt(norm_b)
    if denom <= 1e-12:
        return 1.0
    cos_sim = max(-1.0, min(1.0, dot / denom))
    return 1.0 - cos_sim


def match_selfie_embeddings(
    query_embeddings: Sequence[Sequence[float]],
    stored_photo_embeddings: Iterable[Dict[str, Any]],
    threshold: float = 0.70,
    max_results: int = 200,
) -> List[Dict[str, Any]]:
    """
    Match one or more query face embeddings against stored album face embeddings
    using cosine distance <= threshold. Deduplicates by photo_id using the best
    (minimum) distance across all faces in the photo and query.
    """
    if not query_embeddings or max_results <= 0:
        return []
    threshold_clamped = max(0.0, min(2.0, float(threshold)))
    best_by_photo: Dict[str, float] = {}

    for row in stored_photo_embeddings:
        photo_id = str(row["photo_id"])
        emb = row["embedding"]
        for q_emb in query_embeddings:
            dist = cosine_distance(q_emb, emb)
            if dist <= threshold_clamped:
                prev = best_by_photo.get(photo_id)
                if prev is None or dist < prev:
                    best_by_photo[photo_id] = dist

    ranked: List[Tuple[str, float]] = sorted(best_by_photo.items(), key=lambda kv: (kv[1], kv[0]))
    return [
        {"photo_id": pid, "distance": round(dist, 6)}
        for pid, dist in ranked[:max_results]
    ]


QUEUE_URL = os.getenv("SQS_QUEUE_URL")
BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")

_sqs_client = None
_s3_client = None
_db_pool = None


def get_sqs_client():
    global _sqs_client
    if _sqs_client is None:
        import boto3
        _sqs_client = boto3.client("sqs", region_name=os.getenv("AWS_REGION", "us-east-2"))
    return _sqs_client


def get_s3_client():
    global _s3_client
    if _s3_client is None:
        import boto3
        _s3_client = boto3.client("s3", region_name=os.getenv("AWS_REGION", "us-east-2"))
    return _s3_client


def get_db_pool():
    global _db_pool
    if _db_pool is None:
        from psycopg2 import pool
        _db_pool = pool.SimpleConnectionPool(
            minconn=1,
            maxconn=2,
            host=os.getenv("DB_HOST"),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            port=os.getenv("DB_PORT"),
        )
        print("[+] Database connection pool initialized (1-2 connections)")
    return _db_pool


def _default_represent(local_path: str) -> List[Dict[str, Any]]:
    from deepface import DeepFace
    return DeepFace.represent(
        img_path=local_path,
        model_name="GhostFaceNet",
        detector_backend="retinaface",
        enforce_detection=False,
    )


def process_message(
    message: Dict[str, Any],
    db_pool_override: Optional[Any] = None,
    s3_override: Optional[Any] = None,
    represent_fn: Optional[Callable[[str], List[Dict[str, Any]]]] = None,
    validate_fn: Optional[Callable[[str], bool]] = None,
) -> None:
    body = json.loads(message["Body"])
    photo_id = body["photoId"]
    storage_url = body["storageUrl"]

    active_pool = db_pool_override or get_db_pool()
    active_s3 = s3_override or get_s3_client()
    active_represent = represent_fn or _default_represent
    active_validate = validate_fn or _validate_image_dimensions

    conn = active_pool.getconn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT id FROM photos WHERE id = %s", (photo_id,))
        if cur.fetchone() is None:
            cur.close()
            return
        cur.close()
    finally:
        active_pool.putconn(conn)

    local_path = f"temp_{photo_id}.jpg"
    try:
        active_s3.download_file(BUCKET_NAME, storage_url, local_path)

        if not active_validate(local_path):
            conn = active_pool.getconn()
            try:
                cur = conn.cursor()
                cur.execute("UPDATE photos SET processed = True WHERE id = %s", (photo_id,))
                conn.commit()
                cur.close()
            finally:
                active_pool.putconn(conn)
            return

        try:
            _set_alarm_if_supported(DEEPFACE_TIMEOUT_SECS)
            faces = active_represent(local_path)
            _set_alarm_if_supported(0)
        except DeepFaceTimeout:
            faces = []
        except Exception:
            faces = []
        finally:
            _set_alarm_if_supported(0)

        valid_faces = [f for f in faces if f.get("facial_area", {}).get("w", 0) > 0]

        conn = active_pool.getconn()
        cur = conn.cursor()
        try:
            for face in valid_faces:
                embedding_str = f"[{','.join(map(str, face['embedding']))}]"
                box_area_json = json.dumps(face["facial_area"])
                cur.execute(
                    "INSERT INTO photo_embeddings (photo_id, embedding, box_area) VALUES (%s, %s, %s)",
                    (photo_id, embedding_str, box_area_json),
                )

            cur.execute("UPDATE photos SET processed = True WHERE id = %s", (photo_id,))
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()
            active_pool.putconn(conn)
    finally:
        if os.path.exists(local_path):
            os.remove(local_path)


def main():
    sqs = get_sqs_client()
    while True:
        response = sqs.receive_message(
            QueueUrl=QUEUE_URL,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=20,
        )
        if "Messages" in response:
            for msg in response["Messages"]:
                try:
                    process_message(msg)
                    sqs.delete_message(
                        QueueUrl=QUEUE_URL,
                        ReceiptHandle=msg["ReceiptHandle"],
                    )
                except Exception as e:
                    print(f"Failed to process message entirely: {e}")


if __name__ == "__main__":
    main()
