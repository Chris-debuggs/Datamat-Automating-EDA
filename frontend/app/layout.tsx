import type React from "react"
import type { Metadata } from "next"
import { GeistSans } from "geist/font/sans"
import { GeistMono } from "geist/font/mono"
import { Analytics } from "@vercel/analytics/next"
import "./globals.css"
import { DATAmatProvider } from "@/lib/datamat-context"

export const metadata: Metadata = {
  title: "DATAmat - AI-Powered Data Analysis",
  description: "Automated Exploratory Data Analysis platform powered by AI21 Labs",
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en">
      <body className={`font-sans antialiased ${GeistSans.variable} ${GeistMono.variable}`}>
        <DATAmatProvider>
          {children}
          <Analytics />
        </DATAmatProvider>
      </body>
    </html>
  )
}
