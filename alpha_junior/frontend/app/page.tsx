import Link from 'next/link';
import { ArrowRight, TrendingUp, Shield, Users, BarChart3 } from 'lucide-react';

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-aj-navy-900">
      {/* Navigation */}
      <nav className="border-b border-aj-navy-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <span className="font-display text-xl font-bold text-white">
                Alpha <span className="text-aj-indigo-500">Junior</span>
              </span>
            </div>
            <div className="flex items-center space-x-4">
              <Link 
                href="/login" 
                className="text-gray-300 hover:text-white px-3 py-2 text-sm font-medium"
              >
                Sign In
              </Link>
              <Link 
                href="/register" 
                className="btn-primary"
              >
                Get Started
              </Link>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="relative py-20 lg:py-32 overflow-hidden">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <h1 className="text-4xl md:text-6xl font-display font-bold text-white mb-6">
              Institutional Fund
              <span className="text-gradient"> Management</span>
              <br />
              Platform
            </h1>
            <p className="text-xl text-gray-400 max-w-2xl mx-auto mb-8">
              Professional-grade fund management for accredited investors and fund managers. 
              Access private equity, hedge funds, and alternative investments.
            </p>
            <div className="flex justify-center space-x-4">
              <Link href="/register" className="btn-primary text-base px-8 py-3">
                Start Investing
                <ArrowRight className="ml-2 h-5 w-5" />
              </Link>
              <Link href="/funds" className="btn-secondary text-base px-8 py-3">
                Explore Funds
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-16 bg-aj-navy-800">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            <div className="text-center">
              <div className="text-3xl font-bold text-white mb-1">$2.5B+</div>
              <div className="text-sm text-gray-400">Assets Under Management</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-white mb-1">50+</div>
              <div className="text-sm text-gray-400">Active Funds</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-white mb-1">1,200+</div>
              <div className="text-sm text-gray-400">Investors</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-aj-gold-400 mb-1">24.8%</div>
              <div className="text-sm text-gray-400">Avg Annual Return</div>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-display font-bold text-white mb-4">
              Why Choose Alpha Junior?
            </h2>
            <p className="text-gray-400 max-w-2xl mx-auto">
              Built for professional investors who demand transparency, security, and performance.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
            <div className="card p-6 hover:border-aj-indigo-500 transition-colors">
              <div className="w-12 h-12 bg-aj-indigo-500/10 rounded-lg flex items-center justify-center mb-4">
                <TrendingUp className="h-6 w-6 text-aj-indigo-500" />
              </div>
              <h3 className="text-lg font-semibold text-white mb-2">Performance Tracking</h3>
              <p className="text-gray-400 text-sm">
                Real-time NAV updates, benchmark comparisons, and detailed analytics.
              </p>
            </div>

            <div className="card p-6 hover:border-aj-indigo-500 transition-colors">
              <div className="w-12 h-12 bg-aj-indigo-500/10 rounded-lg flex items-center justify-center mb-4">
                <Shield className="h-6 w-6 text-aj-indigo-500" />
              </div>
              <h3 className="text-lg font-semibold text-white mb-2">Bank-Grade Security</h3>
              <p className="text-gray-400 text-sm">
                2FA, encrypted data, and comprehensive audit logging for compliance.
              </p>
            </div>

            <div className="card p-6 hover:border-aj-indigo-500 transition-colors">
              <div className="w-12 h-12 bg-aj-indigo-500/10 rounded-lg flex items-center justify-center mb-4">
                <Users className="h-6 w-6 text-aj-indigo-500" />
              </div>
              <h3 className="text-lg font-semibold text-white mb-2">Accredited Access</h3>
              <p className="text-gray-400 text-sm">
                Rigorous KYC verification ensures only qualified investors participate.
              </p>
            </div>

            <div className="card p-6 hover:border-aj-indigo-500 transition-colors">
              <div className="w-12 h-12 bg-aj-indigo-500/10 rounded-lg flex items-center justify-center mb-4">
                <BarChart3 className="h-6 w-6 text-aj-indigo-500" />
              </div>
              <h3 className="text-lg font-semibold text-white mb-2">Diverse Strategies</h3>
              <p className="text-gray-400 text-sm">
                Access hedge funds, private equity, venture capital, and real estate.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 bg-gradient-to-b from-aj-navy-800 to-aj-navy-900">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="text-3xl font-display font-bold text-white mb-4">
            Ready to Start Investing?
          </h2>
          <p className="text-gray-400 mb-8">
            Join thousands of accredited investors accessing institutional-grade funds.
          </p>
          <div className="flex justify-center space-x-4">
            <Link href="/register" className="btn-primary text-base px-8 py-3">
              Create Account
            </Link>
            <Link href="/funds" className="btn-secondary text-base px-8 py-3">
              View Available Funds
            </Link>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-aj-navy-700 py-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid md:grid-cols-4 gap-8">
            <div>
              <span className="font-display text-lg font-bold text-white">
                Alpha <span className="text-aj-indigo-500">Junior</span>
              </span>
              <p className="text-gray-400 text-sm mt-2">
                Professional fund management platform for institutional investors.
              </p>
            </div>
            <div>
              <h4 className="font-semibold text-white mb-3">Platform</h4>
              <ul className="space-y-2 text-sm text-gray-400">
                <li><Link href="/funds" className="hover:text-white">Explore Funds</Link></li>
                <li><Link href="/about" className="hover:text-white">About Us</Link></li>
                <li><Link href="/pricing" className="hover:text-white">Pricing</Link></li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold text-white mb-3">Resources</h4>
              <ul className="space-y-2 text-sm text-gray-400">
                <li><Link href="/docs" className="hover:text-white">Documentation</Link></li>
                <li><Link href="/api" className="hover:text-white">API Reference</Link></li>
                <li><Link href="/help" className="hover:text-white">Help Center</Link></li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold text-white mb-3">Legal</h4>
              <ul className="space-y-2 text-sm text-gray-400">
                <li><Link href="/privacy" className="hover:text-white">Privacy Policy</Link></li>
                <li><Link href="/terms" className="hover:text-white">Terms of Service</Link></li>
                <li><Link href="/disclosures" className="hover:text-white">Disclosures</Link></li>
              </ul>
            </div>
          </div>
          <div className="border-t border-aj-navy-700 mt-8 pt-8 text-center text-sm text-gray-400">
            <p>© 2024 Alpha Junior. All rights reserved. Past performance is not indicative of future results.</p>
          </div>
        </div>
      </footer>
    </div>
  );
}
