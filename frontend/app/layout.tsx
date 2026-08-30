import type { Metadata } from "next";
import { Fraunces, Inter } from "next/font/google";

import { Footer } from "@/components/Footer";
import { AuthProvider } from "@/lib/auth";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

// A serif for headings. It's a food app, not a spreadsheet.
const fraunces = Fraunces({
  subsets: ["latin"],
  variable: "--font-fraunces",
  display: "swap",
  axes: ["SOFT", "WONK"],
});

export const metadata: Metadata = {
  title: "Kaya, the nutrition coach that replans when life happens",
  description:
    "An autonomous nutrition coach. Tell it how you eat, log what you actually did, and it rewrites your week when you fall off, no willpower required.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    // suppressHydrationWarning is required, not cosmetic: the inline script
    // below sets data-theme before React hydrates, and without this React
    // treats the attribute mismatch as a failed hydration and abandons it,
    // leaving every button and form on the site inert.
    <html
      lang="en"
      data-theme="light"
      suppressHydrationWarning
      className={`${inter.variable} ${fraunces.variable}`}
    >
      <head>
        {/* Resolve the theme before first paint. Without this the page renders
            light, then snaps to dark once React hydrates. The flash is brief
            and very visible on a dark-mode machine. */}
        <script
          dangerouslySetInnerHTML={{
            __html: `(function(){try{var t=localStorage.getItem("kaya-theme");if(t!=="dark"&&t!=="light"){t=matchMedia("(prefers-color-scheme: dark)").matches?"dark":"light"}document.documentElement.setAttribute("data-theme",t)}catch(e){}})()`,
          }}
        />
      </head>
      <body className="antialiased min-h-screen flex flex-col">
        <AuthProvider>
          <div className="flex-1">{children}</div>
          <Footer />
        </AuthProvider>
      </body>
    </html>
  );
}
