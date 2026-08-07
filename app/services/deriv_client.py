import asyncio
from contextlib import asynccontextmanager
from deriv_api import DerivAPI
from deriv_api.errors import ResponseError
from app.core.config import get_settings

settings = get_settings()

PAIR_TO_DERIV_SYMBOL = {
    "EUR/USD": "frxEURUSD",
    "GBP/USD": "frxGBPUSD",
    "USD/JPY": "frxUSDJPY",
    "AUD/USD": "frxAUDUSD",
}

CONNECT_TIMEOUT_SECONDS = 15


class DerivAuthError(Exception):
    """Raised when Deriv auth fails, times out, or the authorized account is not virtual/demo."""


@asynccontextmanager
async def deriv_session():
    """
    Opens a Deriv API connection, authorizes with the configured token, and closes
    the connection afterward no matter what. Refuses to yield a session at all if
    the authorized account isn't virtual — this is the one hard gate standing
    between paper_trading.py and placing an order on a real-money account, so it
    is intentionally not configurable or overridable.

    The authorize call is wrapped in a timeout: a flaky WebSocket handshake should
    surface as a clear error, not hang the calling request indefinitely.
    """
    if not settings.deriv_api_token:
        raise DerivAuthError(
            "DERIV_API_TOKEN is not set. Generate one from a Deriv DEMO account "
            "(Account Settings -> API token) and add it to .env."
        )

    api = DerivAPI(app_id=settings.deriv_app_id)
    try:
        try:
            auth = await asyncio.wait_for(
                api.authorize({"authorize": settings.deriv_api_token}), timeout=CONNECT_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            raise DerivAuthError(
                f"Timed out connecting to Deriv after {CONNECT_TIMEOUT_SECONDS}s. Try again — "
                "this is a network issue, not a problem with the token."
            )
        except ResponseError as e:
            raise DerivAuthError(
                f"Deriv rejected the API token: {e}. Generate a fresh one from your Deriv DEMO "
                "account under Account Settings -> API token, and make sure it has Trade permission."
            )
        account = auth["authorize"]
        if not account.get("is_virtual"):
            raise DerivAuthError(
                f"Refusing to trade: Deriv account {account.get('loginid')} is a REAL account, "
                "not virtual/demo. This project only ever trades on demo accounts — generate a "
                "new API token from a demo account instead."
            )
        yield api, account
    finally:
        try:
            await asyncio.wait_for(api.disconnect(), timeout=5)
        except Exception:
            pass  # best-effort cleanup; a stuck disconnect shouldn't mask the real error above


def deriv_symbol_for(pair: str) -> str:
    try:
        return PAIR_TO_DERIV_SYMBOL[pair]
    except KeyError:
        raise ValueError(
            f"No Deriv symbol mapping for pair '{pair}'. Known pairs: {list(PAIR_TO_DERIV_SYMBOL)}"
        )
