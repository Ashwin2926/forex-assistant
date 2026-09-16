import { redirect } from "next/navigation";

// The rule engine's own "Signal Feed" home page was removed along with the rest of that
// engine (see PROGRESS.md's rule-engine-removal entry) -- SMC consensus is the only live
// signal source now, and /trading-signals is already its practical, actionable view
// (position sizing included). Redirect rather than leave "/" 404ing or duplicating that
// page's content here.
export default function Home() {
  redirect("/trading-signals");
}
