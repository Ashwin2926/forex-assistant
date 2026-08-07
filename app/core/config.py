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

    # Ingestion/scoring scheduling. The in-process APScheduler job (ENABLE_SCHEDULER=true)
    # only works on a host that stays alive between requests — it's useless on FastAPI
    # Cloud's free tier, which scales the app to zero on idle and kills any in-memory job.
    # Set ENABLE_SCHEDULER=false there and drive POST /cron/tick from an external scheduler
    # instead (e.g. GitHub Actions cron) so a request actually arrives on a schedule.
    # CRON_SECRET guards that endpoint from being triggered by anyone who finds the URL —
    # each ingest call spends Twelve Data quota, so it can't be left open.
    enable_scheduler: bool = True
    cron_secret: str = ""

    @property
    def pairs_list(self) -> list[str]:
        return [p.strip() for p in self.forex_pairs.split(",")]

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    return Settings()
