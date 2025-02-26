import Link from "next/link"
import { ThemeToggle } from "./theme-toggle"

export function Header() {
  return (
    <header className="border-b">
      <div className="container mx-auto px-4 py-4 flex justify-between items-center">
        <Link href="/" className="text-2xl font-bold" style={{ color: "hsl(var(--gouda-yellow))" }}>
          :goudadocs
        </Link>
        <ThemeToggle />
      </div>
    </header>
  )
}

