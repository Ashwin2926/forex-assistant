"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { clearToken, getToken } from "@/lib/auth";

export function AuthNav() {
  const router = useRouter();
  const [loggedIn, setLoggedIn] = useState(false);

  useEffect(() => {
    setLoggedIn(!!getToken());
  }, []);

  if (!loggedIn) return null;

  return (
    <button
      onClick={() => {
        clearToken();
        router.replace("/login");
      }}
      className="whitespace-nowrap text-sm transition-colors"
      style={{ color: "var(--text-muted)" }}
    >
      Log out
    </button>
  );
}
