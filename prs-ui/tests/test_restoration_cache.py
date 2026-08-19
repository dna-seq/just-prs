from prs_ui.mixin import missing_restoration_keys, restoration_cache_keys


def test_restoration_cache_keys_are_sample_aware() -> None:
    rows = [
        {"pgs_id": "PGS000001", "sample": "Mom", "score": 1.0},
        {"pgs_id": "PGS000001", "sample": "Dad", "score": 2.0},
        {"pgs_id": "", "sample": "Mom", "score": 3.0},
    ]

    assert restoration_cache_keys(rows) == {
        ("PGS000001", "Mom"),
        ("PGS000001", "Dad"),
    }


def test_missing_restoration_keys_empty_when_both_sides_cached() -> None:
    cached = [
        {"pgs_id": "PGS000001", "sample": "", "score": 1.0},
        {"pgs_id": "PGS000002", "sample": "", "score": 2.0},
    ]

    assert missing_restoration_keys(cached, ["PGS000001", "PGS000002"], [""]) == []


def test_missing_restoration_keys_lists_only_the_uncached_side() -> None:
    cached = [{"pgs_id": "PGS000001", "sample": "Mom", "score": 1.0}]

    missing = missing_restoration_keys(
        cached,
        ["PGS000001", "PGS000002"],
        ["Mom", "Dad"],
    )

    assert missing == [
        ("PGS000002", "Mom"),
        ("PGS000001", "Dad"),
        ("PGS000002", "Dad"),
    ]


def test_missing_restoration_keys_treats_empty_labels_as_single_sample() -> None:
    cached = [{"pgs_id": "PGS000001", "sample": "", "score": 1.0}]

    assert missing_restoration_keys(cached, ["PGS000001"], []) == []
    assert missing_restoration_keys(cached, ["PGS000002"], []) == [("PGS000002", "")]
