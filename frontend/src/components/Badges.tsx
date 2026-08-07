import type { Direction, PaperTradeStatus, SignalStatus } from "@/lib/types";

const directionStyles: Record<Direction, string> = {
  BUY: "bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300",
  SELL: "bg-rose-100 text-rose-800 dark:bg-rose-950 dark:text-rose-300",
  HOLD: "bg-zinc-100 text-zinc-600 dark:bg-zinc-800 dark:text-zinc-400",
};

export function DirectionBadge({ direction }: { direction: Direction }) {
  return (
    <span
      className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-semibold ${directionStyles[direction]}`}
    >
      {direction}
    </span>
  );
}

const statusStyles: Record<SignalStatus, string> = {
  pending: "bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300",
  hit: "bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300",
  miss: "bg-rose-100 text-rose-800 dark:bg-rose-950 dark:text-rose-300",
  expired: "bg-zinc-100 text-zinc-600 dark:bg-zinc-800 dark:text-zinc-400",
};

export function StatusBadge({ status }: { status: SignalStatus }) {
  return (
    <span
      className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ${statusStyles[status]}`}
    >
      {status}
    </span>
  );
}

const paperTradeStatusStyles: Record<PaperTradeStatus, string> = {
  open: "bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300",
  won: "bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300",
  lost: "bg-rose-100 text-rose-800 dark:bg-rose-950 dark:text-rose-300",
  error: "bg-zinc-200 text-zinc-700 dark:bg-zinc-800 dark:text-zinc-300",
};

export function PaperTradeStatusBadge({ status }: { status: PaperTradeStatus }) {
  return (
    <span
      className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ${paperTradeStatusStyles[status]}`}
    >
      {status}
    </span>
  );
}
