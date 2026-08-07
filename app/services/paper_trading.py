from datetime import datetime
from app.models.schemas import Signal, PaperTrade
from app.services.deriv_client import deriv_session, deriv_symbol_for

DEFAULT_TARGET_ATR_MULT = 1.5
DEFAULT_STOP_ATR_MULT = 1.0


async def execute_paper_trade(signal: Signal, stake: float = 10.0, multiplier: int = 100) -> PaperTrade:
    """
    Executes a directional live Signal as a Deriv Multipliers contract on the
    authorized demo account.

    Multipliers are a leveraged derivative, not a 1:1 unit trade, so our ATR-based
    target/stop (absolute price levels) are converted into the monetary take_profit/
    stop_loss amounts Multipliers actually take: pnl ~= stake * multiplier *
    (price_move / entry_price). This is an approximation of multiplier P&L mechanics
    for risk-sizing purposes — it doesn't model Deriv's own spread or commission, so
    actual fills will differ slightly from this estimate.
    """
    if signal.direction not in ("BUY", "SELL"):
        raise ValueError("Only BUY/SELL signals can be paper-traded, not HOLD.")
    if signal.target_price is None or signal.stop_price is None:
        raise ValueError("Signal is missing target_price/stop_price — attach ATR targets before trading it.")

    deriv_symbol = deriv_symbol_for(signal.pair)
    contract_type = "MULTUP" if signal.direction == "BUY" else "MULTDOWN"

    entry = signal.price_at_signal
    target_move_pct = (signal.target_price - entry) / entry
    stop_move_pct = (signal.stop_price - entry) / entry
    if signal.direction == "SELL":
        target_move_pct = -target_move_pct
        stop_move_pct = -stop_move_pct
    # both are now positive-favorable-direction fractions of price move

    take_profit_amount = round(stake * multiplier * target_move_pct, 2)
    stop_loss_amount = round(stake * multiplier * abs(stop_move_pct), 2)

    async with deriv_session() as (api, account):
        currency = account["currency"]
        loginid = account["loginid"]

        trade = PaperTrade(
            pair=signal.pair,
            interval=signal.interval,
            profile=signal.profile,
            direction=signal.direction,
            deriv_symbol=deriv_symbol,
            signal_price=entry,
            target_price=signal.target_price,
            stop_price=signal.stop_price,
            stake=stake,
            multiplier=multiplier,
            currency=currency,
            take_profit_amount=take_profit_amount,
            stop_loss_amount=stop_loss_amount,
            is_virtual=True,
            deriv_loginid=loginid,
            opened_at=datetime.utcnow(),
        )

        try:
            buy_resp = await api.buy({
                "buy": "1",
                "price": stake,
                "parameters": {
                    "amount": stake,
                    "basis": "stake",
                    "contract_type": contract_type,
                    "currency": currency,
                    "multiplier": multiplier,
                    "symbol": deriv_symbol,
                    "limit_order": {
                        "take_profit": take_profit_amount,
                        "stop_loss": stop_loss_amount,
                    },
                },
            })
            buy_info = buy_resp["buy"]
            trade.contract_id = buy_info.get("contract_id")
            trade.entry_spot = buy_info.get("start_spot")
            trade.buy_price = buy_info.get("buy_price")
        except Exception as e:
            trade.status = "error"
            trade.error = str(e)

    return trade


async def sync_open_trade(trade_doc: dict) -> dict:
    """
    Polls Deriv for a paper trade's current contract state. Returns the doc
    unchanged if the contract is still open (or the sync call itself fails —
    a transient API hiccup shouldn't corrupt stored trade state), otherwise
    returns it updated to "won"/"lost" with final P&L.
    """
    contract_id = trade_doc.get("contract_id")
    if not contract_id:
        return trade_doc

    try:
        async with deriv_session() as (api, _account):
            poc = await api.proposal_open_contract({"contract_id": contract_id})
    except Exception:
        return trade_doc

    info = poc.get("proposal_open_contract") or {}
    if not info.get("is_sold"):
        return trade_doc

    profit = float(info.get("profit", 0))
    trade_doc["status"] = "won" if profit > 0 else "lost"
    trade_doc["pnl"] = profit
    trade_doc["sell_price"] = info.get("sell_price")
    trade_doc["closed_at"] = datetime.utcnow()
    return trade_doc
