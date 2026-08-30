"""CORS configuration must survive the ways people actually type a URL.

Closing the CORS loop is the step this project's deployment kept failing on,
and every failure was silent: the browser reports only that a header was
missing, the dashboard shows a value that looks right, and an exact-string
comparison rejects it anyway. A trailing slash is enough.

Nothing corrected here is a matter of taste. A browser's `Origin` header is
scheme://host[:port], lowercase, with no path. Always. So an allow list entry
that ends in a slash, carries quotes, or is uppercase is a typo that can never
match anything, and it is fixed rather than left to fail at 3am.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.middleware.cors import CORSMiddleware

from app.core.config import Settings, normalise_origin, settings

SITE = "https://health-coach-agent-two.vercel.app"


class TestOriginsAreNormalised:
    @pytest.mark.parametrize(
        "typed",
        [
            SITE,
            SITE + "/",                        # the classic, and it fails silently
            f"  {SITE}  ",                     # pasted with whitespace
            f'"{SITE}"',                       # quoted, as if it were a shell value
            f"'{SITE}'",
            SITE.upper(),                      # shouty paste
            "health-coach-agent-two.vercel.app",  # no scheme: can never match
        ],
    )
    def test_every_plausible_typo_lands_on_the_same_origin(self, typed):
        assert normalise_origin(typed) == SITE

    def test_a_port_is_kept(self):
        """localhost:3000 is a different origin from localhost."""
        assert normalise_origin("http://localhost:3000/") == "http://localhost:3000"

    def test_an_explicit_http_is_not_upgraded(self):
        """Only a *missing* scheme is filled in. http:// is a deliberate choice.
        Local development runs on it."""
        assert normalise_origin("http://localhost:3000") == "http://localhost:3000"


class TestTheSettingAccepts:
    @staticmethod
    def _origins(raw, monkeypatch):
        monkeypatch.setenv("CORS_ORIGINS", raw)
        return Settings().cors_origins

    def test_comma_separated(self, monkeypatch):
        assert self._origins(f"{SITE},http://localhost:3000", monkeypatch) == [
            SITE,
            "http://localhost:3000",
        ]

    def test_comma_separated_with_spaces_and_slashes(self, monkeypatch):
        assert self._origins(f"{SITE}/ , http://localhost:3000/", monkeypatch) == [
            SITE,
            "http://localhost:3000",
        ]

    def test_a_json_array(self, monkeypatch):
        """What a platform that round-trips its config through JSON hands back."""
        assert self._origins(f'["{SITE}/"]', monkeypatch) == [SITE]

    def test_a_single_value(self, monkeypatch):
        assert self._origins(SITE, monkeypatch) == [SITE]


class TestThePreflightActuallyPasses:
    """Parsing correctly and answering correctly are different claims.

    Only the second one unblocks a browser, so this drives the real wiring
    function against a real preflight request.
    """

    @staticmethod
    def _client(origins, monkeypatch):
        from app.main import add_cors

        monkeypatch.setattr(settings, "cors_origins", origins)

        app = FastAPI()

        @app.post("/api/v1/auth/register")
        async def _register():  # pragma: no cover - never invoked by a preflight
            return {"ok": True}

        add_cors(app)
        return TestClient(app)

    @pytest.mark.parametrize(
        "configured",
        [[SITE], [SITE, "http://localhost:3000"]],
    )
    def test_the_browsers_preflight_is_allowed(self, configured, monkeypatch):
        response = self._client(configured, monkeypatch).options(
            "/api/v1/auth/register",
            headers={
                "Origin": SITE,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )

        assert response.status_code == 200, response.text
        assert response.headers.get("access-control-allow-origin") == SITE

    def test_an_unrelated_site_is_still_refused(self, monkeypatch):
        """Normalising must not quietly widen the allow list into a wildcard."""
        response = self._client([SITE], monkeypatch).options(
            "/api/v1/auth/register",
            headers={
                "Origin": "https://not-your-site.example",
                "Access-Control-Request-Method": "POST",
            },
        )

        assert response.headers.get("access-control-allow-origin") is None

    def test_the_real_app_is_wired_the_same_way(self):
        """The factory is only worth testing if the app actually uses it."""
        import app.main as main_module

        assert any(
            middleware.cls is CORSMiddleware
            for middleware in main_module.app.user_middleware
        ), "the deployed app has no CORS middleware at all"


class TestHealthReportsIt:
    def test_the_running_origins_are_visible(self, monkeypatch):
        """The only version that matters is the one the process is holding, and
        no dashboard can show you that."""
        import app.main as main_module

        monkeypatch.setattr(settings, "cors_origins", [SITE])

        body = TestClient(main_module.app).get("/health").json()
        assert body["cors_origins"] == [SITE]
