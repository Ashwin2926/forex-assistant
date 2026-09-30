import { redirect } from "next/navigation";

// The dashboard is the landing page: open signals (any day) sized to your account, plus what
// every pair/interval last decided and why -- read-only, so it loads fast. /trading-signals
// (fresh consensus checks, which rarely fire) is still in the side nav.
export default function Home() {
  redirect("/dashboard");
}
