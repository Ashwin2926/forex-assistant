from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    twelve_data_api_key: str
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db_name: str = "forex_assistant"
    forex_pairs: str = "EUR/USD,GBP/USD,USD/JPY,AUD/USD"
    env: str = "development"

    # Deriv paper trading (Phase 6). app_id 1089 is Deriv's public testing app ID —
    # register your own at api.deriv.com for anything beyond local development.
    # deriv_api_token MUST be generated from a virtual (demo) account — paper_trading.py
    # refuses to trade if the authorized account isn't virtual, but there's no substitute
    # for using a demo-account token in the first place.
    deriv_app_id: str = "1089"
    deriv_api_token: str = ""

    # Login credentials for the frontend/API. auth_password_hash is a bcrypt hash, never
    # the plaintext password — generate one with app.core.auth.hash_password. auth_secret_key
    # signs session tokens; if unset, AuthMiddleware fails closed (nothing can authenticate)
    # rather than falling back to some default secret.
    auth_username: str = ""
    auth_password_hash: str = ""
    auth_secret_key: str = ""

    # Static credential for the GitHub Actions ingestion cron (.github/workflows/keep-fresh.yml)
    # — it can't do an interactive login, so it sends this as X-Service-Token instead of a user
    # JWT. Unset by default, matching auth_secret_key's fail-closed behavior.
    automation_token: str = ""

    @property
    def pairs_list(self) -> list[str]:
        return [p.strip() for p in self.forex_pairs.split(",")]

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    return Settings()
