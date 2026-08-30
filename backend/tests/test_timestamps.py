"""Stored timestamps have to come back knowing what timezone they are in.

BSON has no timezone: Mongo stores UTC and, by default, hands it back as a
naive datetime. FastAPI then serialises `2026-08-29T17:34:09` with no offset,
and JavaScript parses an offset-less datetime as *local* time, so the
decision history was wrong by the reader's UTC offset. Five and a half hours
in India: long enough for "5 minutes ago" to render as "5h ago", or as a
moment in the future.
"""

import inspect

from app.db import mongo
from app.models.common import utcnow


def test_utcnow_is_timezone_aware():
    """The write side. Naive here and the offset is lost before Mongo sees it."""
    assert utcnow().tzinfo is not None


def test_the_client_is_built_timezone_aware():
    """The read side, which is where the bug actually lived.

    Writing an aware datetime is not enough. BSON drops the offset either way.
    `tz_aware=True` is what makes the driver reattach UTC on the way back, and
    without it every timestamp the frontend renders is silently shifted.
    """
    source = inspect.getsource(mongo.connect_to_mongo)

    assert "tz_aware=True" in source, (
        "the Mongo client must be tz_aware, or timestamps come back naive and "
        "the browser reads them as local time"
    )
