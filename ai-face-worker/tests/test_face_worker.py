import json
import pytest
from main import (
    cosine_distance,
    match_selfie_embeddings,
    process_message,
)


class FakeCursor:
    def __init__(self, state):
        self.state = state
        self._last_row = None

    def execute(self, sql, params=None):
        self.state["queries"].append((sql, params))
        if sql.startswith("SELECT id FROM photos"):
            pid = params[0]
            self._last_row = (pid,) if pid in self.state["existing_photos"] else None
        elif sql.startswith("INSERT INTO photo_embeddings"):
            self.state["embeddings"].append(params)
        elif sql.startswith("UPDATE photos SET processed = True"):
            self.state["processed"].add(params[0])

    def fetchone(self):
        return self._last_row

    def close(self):
        pass


class FakeConn:
    def __init__(self, state):
        self.state = state

    def cursor(self):
        return FakeCursor(self.state)

    def commit(self):
        self.state["commits"] += 1

    def rollback(self):
        self.state["rollbacks"] += 1


class FakePool:
    def __init__(self, existing_photos):
        self.state = {
            "existing_photos": set(existing_photos),
            "embeddings": [],
            "processed": set(),
            "queries": [],
            "commits": 0,
            "rollbacks": 0,
        }

    def getconn(self):
        return FakeConn(self.state)

    def putconn(self, conn):
        pass


class FakeS3:
    def download_file(self, bucket, key, local_path):
        with open(local_path, "wb") as f:
            f.write(b"fake-jpeg-bytes")


def test_cosine_distance_and_selfie_matching():
    assert cosine_distance([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]) == pytest.approx(0.0, abs=1e-6)
    assert cosine_distance([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]) == pytest.approx(1.0, abs=1e-6)
    assert cosine_distance([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]) == 1.0

    stored = [
        {"photo_id": "photo-1", "embedding": [0.99, 0.05, 0.0]},
        {"photo_id": "photo-1", "embedding": [0.10, 0.95, 0.0]},  # 2nd face in same photo
        {"photo_id": "photo-2", "embedding": [0.0, 1.0, 0.0]},
        {"photo_id": "photo-3", "embedding": [0.85, 0.50, 0.0]},
    ]
    matches = match_selfie_embeddings([[1.0, 0.0, 0.0]], stored, threshold=0.30)
    assert [m["photo_id"] for m in matches] == ["photo-1", "photo-3"]
    assert matches[0]["distance"] < matches[1]["distance"]


def test_process_message_persists_valid_faces_and_cleans_temp_file():
    pool = FakePool(existing_photos={"p-100"})
    msg = {"Body": json.dumps({"photoId": "p-100", "storageUrl": "albums/a1/p-100.jpg"})}

    def fake_represent(_path):
        return [
            {"embedding": [0.1, 0.2, 0.3], "facial_area": {"x": 10, "y": 20, "w": 50, "h": 60}},
            {"embedding": [0.0, 0.0, 0.0], "facial_area": {"x": 0, "y": 0, "w": 0, "h": 0}},  # invalid 0-width box
        ]

    process_message(
        msg,
        db_pool_override=pool,
        s3_override=FakeS3(),
        represent_fn=fake_represent,
        validate_fn=lambda _p: True,
    )

    assert "p-100" in pool.state["processed"]
    assert len(pool.state["embeddings"]) == 1
    assert pool.state["embeddings"][0][0] == "p-100"
    assert pool.state["embeddings"][0][1] == "[0.1,0.2,0.3]"


def test_process_message_skips_deleted_photo():
    pool = FakePool(existing_photos=set())
    msg = {"Body": json.dumps({"photoId": "deleted-id", "storageUrl": "albums/a1/deleted.jpg"})}
    process_message(
        msg,
        db_pool_override=pool,
        s3_override=FakeS3(),
        represent_fn=lambda _p: [],
        validate_fn=lambda _p: True,
    )
    assert len(pool.state["processed"]) == 0
    assert len(pool.state["embeddings"]) == 0
