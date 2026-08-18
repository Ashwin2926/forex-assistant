import type { ReactNode } from "react";

// Temporarily disabled while backend auth is also disabled for debugging
// (see AuthMiddleware in app/main.py). Re-enable the login redirect once
// that's back.
export function AuthGuard({ children }: { children: ReactNode }) {
  return <>{children}</>;
}
