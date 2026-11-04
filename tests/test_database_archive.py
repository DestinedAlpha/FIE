import gzip
import json
import sqlite3
from contextlib import closing

import pytest

from fpl_intelligence.database_archive import pack_database, unpack_database


@pytest.mark.parametrize("legacy", [False, True])
def test_database_archive_roundtrip_preserves_account_and_never_overwrites(tmp_path, legacy):
    original = tmp_path / "original.sqlite"
    with closing(sqlite3.connect(original)) as database:
        database.execute("CREATE TABLE account (bank INT)")
        database.execute("INSERT INTO account VALUES (16)")
        database.commit()
    folder = tmp_path / "package"
    folder.mkdir()
    if legacy:
        (folder / "state.sqlite.gz").write_bytes(gzip.compress(original.read_bytes()))
    else:
        paths = pack_database(original, folder, part_bytes=1024)
        assert len(paths) > 2
    restored = tmp_path / "existing" / "state.sqlite"
    unpack_database(folder, restored)
    assert restored.read_bytes() == original.read_bytes()
    with closing(sqlite3.connect(restored)) as database:
        assert database.execute("SELECT bank FROM account").fetchone()[0] == 16
    with pytest.raises(ValueError, match="existing account"):
        unpack_database(folder, restored)


@pytest.mark.parametrize("fault", ["truncated", "tampered", "unsafe", "raw_hash"])
def test_damaged_archive_never_leaves_a_partial_database(tmp_path, fault):
    original = tmp_path / "original.sqlite"
    with closing(sqlite3.connect(original)) as database:
        database.execute("CREATE TABLE account (bank INT)")
    folder = tmp_path / "package"
    paths = pack_database(original, folder, part_bytes=1024)
    manifest = json.loads(paths[0].read_text())
    if fault == "truncated":
        paths[-1].unlink()
    elif fault == "tampered":
        body = bytearray(paths[-1].read_bytes())
        body[-1] ^= 1
        paths[-1].write_bytes(body)
    elif fault == "unsafe":
        manifest["parts"][0]["name"] = "../account.sqlite"
    else:
        manifest["raw_sha256"] = "wrong"
    paths[0].write_text(json.dumps(manifest))
    restored = tmp_path / "restored.sqlite"
    with pytest.raises(ValueError):
        unpack_database(folder, restored)
    assert not restored.exists() and not list(tmp_path.glob(".database-restore-*"))
