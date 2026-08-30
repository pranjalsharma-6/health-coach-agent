"""What happens when the database isn't there.

A bad MONGODB_URI used to surface as: one ERROR line, then "Application startup
complete", then a 500 and a 40-line traceback on every request. The line that
said what to fix was pymongo's "must be escaped according to RFC 3986", which
names a standard rather than a fix.

These tests pin the three things that changed: the error explains what to
change, a request answers 503 rather than 500, and the explanation stays out of
production responses.
"""

import httpx
import pytest
from asgi_lifespan import LifespanManager

from app.core.config import settings
from app.db import mongo

BAD_URI = "mongodb+srv://kaya:Secr@t123@cluster0.abc.mongodb.net/?retryWrites=true"


class TestCredentialDiagnosis:
    def test_names_the_character_and_its_replacement(self):
        hint = mongo.describe_credential_problem(BAD_URI)
        assert hint is not None
        assert "@ -> %40" in hint

    def test_never_echoes_the_password(self):
        """The hint goes to logs, which get pasted into issues and screenshots."""
        hint = mongo.describe_credential_problem(BAD_URI)
        assert "Secr" not in hint
        assert "t123" not in hint

    def test_splits_on_the_last_at_sign(self):
        """The @ separating credentials from the host is not the broken one.

        Splitting on the first @ would read the password as "Secr" and find
        nothing wrong with it. The failure would go undiagnosed.
        """
        assert mongo.describe_credential_problem(BAD_URI) is not None

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("mongodb+srv://u:a#b@h.net/", "# -> %23"),
            ("mongodb+srv://u:a/b@h.net/", "/ -> %2F"),
            ("mongodb+srv://u:a?b@h.net/", "? -> %3F"),
            ("mongodb+srv://u:a:b@h.net/", ": -> %3A"),
        ],
    )
    def test_covers_the_other_reserved_characters(self, raw, expected):
        assert expected in mongo.describe_credential_problem(raw)

    @pytest.mark.parametrize(
        "raw",
        [
            "mongodb+srv://user:p%40ss@cluster0.abc.mongodb.net/",  # already encoded
            "mongodb+srv://user:plainpass@cluster0.abc.mongodb.net/",
            "mongodb://localhost:27017",  # no credentials at all
            "not-a-uri",
        ],
    )
    def test_stays_quiet_when_the_uri_is_fine(self, raw):
        assert mongo.describe_credential_problem(raw) is None


@pytest.fixture
def disconnected(monkeypatch):
    """An app whose database never connected, with a known reason."""
    monkeypatch.setattr(mongo, "_database", None)
    monkeypatch.setattr(mongo, "_client", None)
    monkeypatch.setattr(mongo, "_connection_error", "MONGODB_URI is not set.")
    monkeypatch.setattr(mongo, "connect_to_mongo", _noop)
    monkeypatch.setattr(mongo, "close_mongo_connection", _noop)


async def _noop():
    return None


async def _client(app):
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )


class TestRequestsWhileDisconnected:
    async def test_a_request_answers_503_not_500(self, disconnected):
        from app.main import app

        async with LifespanManager(app):
            async with await _client(app) as c:
                response = await c.post(
                    "/api/v1/auth/register",
                    json={
                        "email": "a@b.com",
                        "password": "supersecret",
                        "full_name": "A B",
                    },
                )

        assert response.status_code == 503
        assert "database" in response.json()["detail"].lower()

    async def test_the_reason_reaches_the_developer(self, disconnected):
        from app.main import app

        async with LifespanManager(app):
            async with await _client(app) as c:
                response = await c.post(
                    "/api/v1/auth/register",
                    json={
                        "email": "a@b.com",
                        "password": "supersecret",
                        "full_name": "A B",
                    },
                )

        assert response.json()["reason"] == "MONGODB_URI is not set."

    async def test_the_reason_is_withheld_in_production(
        self, disconnected, monkeypatch
    ):
        """It names configuration. Locally that's the point; publicly it isn't."""
        monkeypatch.setattr(settings, "environment", "production")
        from app.main import app

        async with LifespanManager(app):
            async with await _client(app) as c:
                response = await c.post(
                    "/api/v1/auth/register",
                    json={
                        "email": "a@b.com",
                        "password": "supersecret",
                        "full_name": "A B",
                    },
                )

        assert response.status_code == 503
        assert "reason" not in response.json()

    async def test_health_reports_degraded_with_the_reason(self, disconnected):
        from app.main import app

        async with LifespanManager(app):
            async with await _client(app) as c:
                response = await c.get("/health")

        body = response.json()
        assert response.status_code == 200, "the pinger must still get a response"
        assert body["status"] == "degraded"
        assert body["database"]["reason"] == "MONGODB_URI is not set."


def test_get_database_raises_the_typed_error(monkeypatch):
    """A bare RuntimeError would be caught by the 500 handler, not the 503 one."""
    monkeypatch.setattr(mongo, "_database", None)
    monkeypatch.setattr(mongo, "_connection_error", "because reasons")

    with pytest.raises(mongo.DatabaseUnavailableError, match="because reasons"):
        mongo.get_database()


class TestConnectionFailureMessages:
    """DNS, firewall and auth failures arrive as one exception type.

    They look identical from the outside. A PyMongoError with a wall of
    topology description, and need completely different responses. A user
    whose Wi-Fi dropped was told to check their connection string.
    """

    @staticmethod
    def _describe(message, cls=Exception):
        return mongo.describe_connection_failure(cls(message))

    def test_dns_failure_points_at_the_network(self):
        """`getaddrinfo failed` is one layer below the connection string.
        The hostname was never even looked up."""
        message = self._describe(
            "ac-83tbkc0-shard-00-00.wpud7ke.mongodb.net:27017: "
            "[Errno 11001] getaddrinfo failed"
        )

        assert "resolved" in message
        assert "internet connection" in message

    def test_dns_failure_does_not_blame_the_connection_string(self):
        message = self._describe("[Errno 11001] getaddrinfo failed")
        assert "MONGODB_URI" not in message

    def test_dns_failure_mentions_campus_networks(self):
        """University and office networks commonly block port 27017, and a
        phone hotspot is the fastest way to tell."""
        message = self._describe("[Errno 11001] getaddrinfo failed")
        assert "hotspot" in message

    def test_a_timeout_points_at_network_access(self):
        """Resolved but silent is the Atlas IP allowlist, not DNS."""
        message = self._describe("connection attempt timed out")

        assert "Network Access" in message
        assert "internet connection" not in message

    def test_an_auth_failure_points_at_the_credentials(self):
        message = self._describe("bad auth: Authentication failed")

        assert "username or password" in message
        assert "percent-encoded" in message

    def test_an_unrecognised_failure_is_still_reported(self):
        message = self._describe("something entirely new went wrong")
        assert "something entirely new" in message

    def test_the_message_is_bounded(self):
        """A topology description runs to thousands of characters."""
        message = self._describe("x" * 5000)
        assert len(message) < 400


class TestSrvLookupFailure:
    """`mongodb+srv://` needs SRV and TXT records, not just a hostname.

    A network can resolve A records perfectly. Test-NetConnection succeeds,
    the browser works, and still drop SRV queries. Networks that intercept
    DNS commonly do. The failure says "The resolution lifetime expired ... The
    DNS operation timed out", which contains the word "timed out" and was
    therefore read as an unreachable server, sending the user to check the
    Atlas IP allowlist for a problem that was never on Atlas's side.
    """

    SRV_TIMEOUT = (
        "The resolution lifetime expired after 8.204 seconds: Server "
        "Do53:192.168.102.1@53 answered The DNS operation timed out.; Server "
        "Do53:8.8.8.8@53 answered The DNS operation timed out."
    )

    def _describe(self, message):
        return mongo.describe_connection_failure(Exception(message))

    def test_it_is_recognised_as_a_dns_failure(self):
        message = self._describe(self.SRV_TIMEOUT)

        assert "SRV" in message
        assert "Network Access" not in message, (
            "an SRV timeout is not an Atlas allowlist problem"
        )

    def test_it_explains_why_other_tools_look_fine(self):
        """The user had just proved the host was reachable on port 27017."""
        message = self._describe(self.SRV_TIMEOUT)
        assert "different record type" in message

    def test_it_names_the_non_srv_workaround(self):
        """The fix that needs no network cooperation."""
        message = self._describe(self.SRV_TIMEOUT)

        assert "non-SRV" in message
        assert "Atlas" in message

    def test_a_plain_server_timeout_still_points_at_network_access(self):
        """The distinction the ordering exists for."""
        message = self._describe("connection attempt timed out")

        assert "Network Access" in message
        assert "SRV" not in message
