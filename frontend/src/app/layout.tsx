import type { Metadata } from "next";
import { Navbar } from "@/components/layout/Navbar";
import { Footer } from "@/components/layout/Footer";
import "./globals.css";

export const metadata: Metadata = {
  title: {
    default: "EthicsEngine \u2014 AI Ethics Benchmarking Platform",
    template: "%s | EthicsEngine",
  },
  description:
    "Independent ethical benchmarking for AI systems. Frontier model scores, HE-300 evaluations, and compliance-ready audit trails.",
  openGraph: {
    title: "EthicsEngine \u2014 AI Ethics Benchmarking",
    description: "How ethical is your AI? Independent HE-300 benchmark scores for every frontier model.",
    siteName: "EthicsEngine",
    url: "https://ethicsengine.org",
  },
  twitter: {
    card: "summary_large_image",
    title: "EthicsEngine",
    description: "Independent ethical benchmarking for AI systems.",
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen flex flex-col">
        <Navbar />
        <main className="flex-1">{children}</main>
        <Footer />
      </body>
    </html>
  );
}
