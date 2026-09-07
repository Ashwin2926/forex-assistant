import type { Direction, PaperTradeStatus, SignalStatus } from "@/lib/types";

const directionStyles: Record<Direction, string> = {
  BUY: "badge-buy",
  SELL: "badge-sell",
  HOLD: "badge-hold",
};

export function DirectionBadge({ direction }: { direction: Direction }) {
  return <span className={`badge ${directionStyles[direction]}`}>{direction}</span>;
}

const statusStyles: Record<SignalStatus, string> = {
  pending: "badge-hold",
  hit: "badge-buy",
  miss: "badge-sell",
  expired: "badge-neutral",
  // RL-only: the agent changed its mind before this signal ever resolved naturally -- distinct
  // from expired (a real timeout) so it doesn't visually read as a loss.
  superseded: "bg-violet-100 text-violet-700 dark:bg-violet-950 dark:text-violet-300",
};

export function StatusBadge({ status }: { status: SignalStatus }) {
  return <span className={`badge ${statusStyles[status]}`}>{status}</span>;
}

const paperTradeStatusStyles: Record<PaperTradeStatus, string> = {
  open: "badge-hold",
  won: "badge-buy",
  lost: "badge-sell",
  error: "badge-neutral",
};

export function PaperTradeStatusBadge({ status }: { status: PaperTradeStatus }) {
  return <span className={`badge ${paperTradeStatusStyles[status]}`}>{status}</span>;
}
