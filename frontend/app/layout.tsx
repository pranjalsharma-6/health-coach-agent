import type { Metadata } from "next";
import { Fraunces, Inter } from "next/font/google";

import { AuthProvider } from "@/lib/auth";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

// A serif for headings — it's a food app, not a spreadsheet.
const fraunces = Fraunces({
  subsets: ["latin"],
  variable: "--font-fraunces",
  display: "swap",
  axes: ["SOFT", "WONK"],
});

export const metadata: Metadata = {
  title: "Kaya — the nutrition coach that replans when life happens",
  description:
    "An autonomous nutrition coach. Tell it how you eat, log what you actually did, and it rewrites your week when you fall off — no willpower required.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`${inter.variable} ${fraunces.variable}`}>
      <body className="antialiased min-h-screen">
        <AuthProvider>{children}</AuthProvider>
      </body>
    </html>
  );
}
