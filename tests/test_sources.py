from ecri import sources


def test_urls_name_a_commit_not_a_branch():
    for name in sources.SHA256:
        url = sources.url(name)
        assert sources.COMMIT in url
        assert "/main/" not in url


def test_status_reports_missing_changed_and_ok(tmp_path):
    name = "MetaData.csv"
    assert sources.status(tmp_path)[name] == "missing"

    (tmp_path / name).write_text("not the real file")
    assert sources.status(tmp_path)[name] == "changed"


def test_status_accepts_exact_bytes(tmp_path, monkeypatch):
    payload = b"SchoolID,FreeLunch\n1,0.3\n"
    (tmp_path / "MetaData.csv").write_bytes(payload)
    import hashlib

    monkeypatch.setitem(sources.SHA256, "MetaData.csv", hashlib.sha256(payload).hexdigest())
    assert sources.status(tmp_path)["MetaData.csv"] == "ok"


def test_every_pin_is_a_full_sha256():
    assert sources.SHA256
    for digest in sources.SHA256.values():
        assert len(digest) == 64 and int(digest, 16) >= 0
