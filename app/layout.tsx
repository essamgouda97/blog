import type React from "react"
import { Inter } from "next/font/google"
import "./globals.css"
import { ThemeProvider } from "@/components/theme-provider"
import { Header } from "@/components/header"
import { Analytics } from "@vercel/analytics/react"
import 'katex/dist/katex.min.css'

const inter = Inter({ subsets: ["latin"] })

export const metadata = {
  title: "GoudaDocs",
  description: "A personal blogging website",
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" suppressHydrationWarning className="dark">
      <head>
        <link
          rel="stylesheet"
          href="https://cdn.jsdelivr.net/npm/katex@0.16.0/dist/katex.min.css"
        />
      </head>
      <body className={`${inter.className} bg-background text-foreground`}>
        <ThemeProvider attribute="class" defaultTheme="dark" enableSystem disableTransitionOnChange>
          <Header />
          <main className="container mx-auto px-4 py-8">{children}</main>
          <Analytics />
        </ThemeProvider>
      </body>
    </html>
  )
}

