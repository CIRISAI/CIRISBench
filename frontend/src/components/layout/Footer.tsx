import Link from "next/link";

export function Footer() {
  return (
    <footer className="border-t border-border bg-bg-card mt-auto">
      <div className="mx-auto max-w-7xl px-6 py-12">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          {/* Brand */}
          <div className="md:col-span-1">
            <div className="flex items-center gap-2 mb-3">
              <div className="flex h-7 w-7 items-center justify-center rounded-md bg-accent font-bold text-white text-xs">
                EE
              </div>
              <span className="font-semibold">EthicsEngine</span>
            </div>
            <p className="text-sm text-text-secondary leading-relaxed">
              Independent ethical benchmarking for AI systems.
              Powered by the CIRIS Framework.
            </p>
          </div>

          {/* Platform */}
          <div>
            <h3 className="text-sm font-semibold mb-3">Platform</h3>
            <ul className="space-y-2 text-sm text-text-secondary">
              <li><Link href="/scores" className="hover:text-text transition-colors">Frontier Scores</Link></li>
              <li><Link href="/leaderboard" className="hover:text-text transition-colors">Leaderboard</Link></li>
              <li><Link href="/pricing" className="hover:text-text transition-colors">Pricing</Link></li>
              <li><Link href="/dashboard" className="hover:text-text transition-colors">Dashboard</Link></li>
            </ul>
          </div>

          {/* Resources */}
          <div>
            <h3 className="text-sm font-semibold mb-3">Resources</h3>
            <ul className="space-y-2 text-sm text-text-secondary">
              <li><a href="https://ciris.ai" target="_blank" rel="noopener noreferrer" className="hover:text-text transition-colors">CIRIS Framework</a></li>
              <li><a href="https://ciris.ai/ciris-scoring" target="_blank" rel="noopener noreferrer" className="hover:text-text transition-colors">CIRIS Scoring</a></li>
              <li><a href="https://github.com/CIRISAI/CIRISBench" target="_blank" rel="noopener noreferrer" className="hover:text-text transition-colors">GitHub</a></li>
              <li><a href="https://github.com/CIRISAI/CIRISBench/blob/main/README.md" target="_blank" rel="noopener noreferrer" className="hover:text-text transition-colors">Documentation</a></li>
            </ul>
          </div>

          {/* Legal */}
          <div>
            <h3 className="text-sm font-semibold mb-3">Company</h3>
            <ul className="space-y-2 text-sm text-text-secondary">
              <li><span>CIRIS L3C</span></li>
              <li><a href="mailto:hello@ethicsengine.org" className="hover:text-text transition-colors">Contact</a></li>
            </ul>
          </div>
        </div>

        <div className="mt-10 pt-6 border-t border-border-subtle flex flex-col sm:flex-row justify-between items-center gap-3">
          <p className="text-xs text-text-muted">
            &copy; {new Date().getFullYear()} CIRIS L3C. HE-300 Benchmark v1.0.
          </p>
          <p className="text-xs text-text-muted">
            Independent. Open-source. Auditable.
          </p>
        </div>
      </div>
    </footer>
  );
}
