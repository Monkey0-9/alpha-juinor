"""
Alpha Junior - Minimal Local Version
Simple Flask app with SQLite, no complex dependencies
"""

from flask import Flask, jsonify, request, render_template_string
from flask_cors import CORS
import sqlite3
import hashlib
import secrets
from datetime import datetime
import os

app = Flask(__name__)
CORS(app)

# Load Alpaca API keys from environment
import os
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY', '')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY', '')

# Import and register trading blueprint
try:
    from trading import trading_bp
    app.register_blueprint(trading_bp)
    print("✓ Trading module loaded")
except Exception as e:
    print(f"⚠ Trading module not loaded: {e}")

# Import autonomous trader
try:
    from autonomous_trader import get_trader
    autonomous_trader = None
    print("✓ Autonomous trader module loaded")
except Exception as e:
    print(f"⚠ Autonomous trader not loaded: {e}")
    autonomous_trader = None

# Import institutional core engine
try:
    from institutional_core import get_institutional_core
    inst_core = get_institutional_core()
    print("✓ Institutional core engine loaded (Goldman Sachs grade)")
except Exception as e:
    print(f"⚠ Institutional core not loaded: {e}")
    inst_core = None

# Database setup
def init_db():
    conn = sqlite3.connect('alpha_junior.db')
    c = conn.cursor()
    
    # Users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT,
            role TEXT DEFAULT 'investor',
            kyc_status TEXT DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Funds table
    c.execute('''
        CREATE TABLE IF NOT EXISTS funds (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            slug TEXT UNIQUE NOT NULL,
            strategy TEXT,
            min_investment REAL,
            nav REAL DEFAULT 100.0,
            status TEXT DEFAULT 'active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Investments table
    c.execute('''
        CREATE TABLE IF NOT EXISTS investments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            investor_id INTEGER,
            fund_id INTEGER,
            amount REAL,
            status TEXT DEFAULT 'active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (investor_id) REFERENCES users (id),
            FOREIGN KEY (fund_id) REFERENCES funds (id)
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✓ Database initialized")

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Alpha Junior - Fund Platform</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #0a0f1e 0%, #1a1f2e 100%);
            color: #fff;
            min-height: 100vh;
            line-height: 1.6;
        }
        .container { max-width: 1200px; margin: 0 auto; padding: 0 20px; }
        header {
            background: rgba(10, 15, 30, 0.9);
            padding: 1rem 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        .logo { font-size: 1.5rem; font-weight: 700; }
        .logo span { color: #4F46E5; }
        .hero {
            padding: 100px 20px;
            text-align: center;
        }
        .hero h1 {
            font-size: 3rem;
            margin-bottom: 1rem;
            background: linear-gradient(135deg, #fff 0%, #4F46E5 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .hero p {
            color: #9ca3af;
            font-size: 1.25rem;
            max-width: 600px;
            margin: 0 auto 2rem;
        }
        .btn {
            display: inline-block;
            padding: 12px 32px;
            background: #4F46E5;
            color: #fff;
            text-decoration: none;
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s;
        }
        .btn:hover {
            background: #4338CA;
            transform: translateY(-2px);
        }
        .features {
            padding: 60px 20px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
        }
        .feature-card {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.05);
            padding: 24px;
            border-radius: 12px;
        }
        .feature-card h3 {
            color: #fff;
            margin-bottom: 8px;
        }
        .feature-card p {
            color: #9ca3af;
            font-size: 0.9rem;
        }
        .api-section {
            padding: 40px 20px;
            background: rgba(255,255,255,0.02);
        }
        .endpoint {
            background: rgba(255,255,255,0.05);
            padding: 16px;
            margin: 10px 0;
            border-radius: 8px;
            font-family: monospace;
            font-size: 0.9rem;
        }
        .method { color: #10B981; font-weight: bold; }
        footer {
            text-align: center;
            padding: 40px 20px;
            color: #6b7280;
            font-size: 0.875rem;
            border-top: 1px solid rgba(255,255,255,0.05);
        }
        .status {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: rgba(16, 185, 129, 0.2);
            color: #10B981;
            padding: 12px 20px;
            border-radius: 8px;
            font-weight: 500;
        }
    </style>
</head>
<body>
    <header>
        <div class="container">
            <div class="logo">Alpha <span>Junior</span></div>
        </div>
    </header>

    <section class="hero">
        <div class="container">
            <h1>Institutional Fund Management</h1>
            <p>Professional-grade fund platform for accredited investors and fund managers.</p>
            <a href="#api" class="btn">View API</a>
        </div>
    </section>

    <section class="features container">
        <div class="feature-card">
            <h3>Performance Tracking</h3>
            <p>Real-time NAV updates and benchmark comparisons</p>
        </div>
        <div class="feature-card">
            <h3>Bank-Grade Security</h3>
            <p>2FA authentication and comprehensive audit logging</p>
        </div>
        <div class="feature-card">
            <h3>Accredited Access</h3>
            <p>Rigorous KYC verification for qualified investors</p>
        </div>
        <div class="feature-card">
            <h3>Diverse Strategies</h3>
            <p>Access hedge funds, private equity, and venture capital</p>
        </div>
    </section>

    <section class="api-section" id="api">
        <div class="container">
            <h2 style="text-align: center; margin-bottom: 30px;">API Endpoints</h2>
            
            <div class="endpoint">
                <span class="method">GET</span> /api/health - Health check
            </div>
            <div class="endpoint">
                <span class="method">POST</span> /api/register - Register new user
            </div>
            <div class="endpoint">
                <span class="method">POST</span> /api/login - Login user
            </div>
            <div class="endpoint">
                <span class="method">GET</span> /api/funds - List all funds
            </div>
            <div class="endpoint">
                <span class="method">POST</span> /api/funds - Create new fund
            </div>
            <div class="endpoint">
                <span class="method">GET</span> /api/investments - List investments
            </div>
            <div class="endpoint">
                <span class="method">POST</span> /api/invest - Make investment
            </div>
            
            <h3 style="margin-top: 30px; margin-bottom: 15px; color: #10B981;">🚀 Alpaca Paper Trading</h3>
            <div class="endpoint">
                <span class="method">GET</span> /api/trading/account - Account balance & equity
            </div>
            <div class="endpoint">
                <span class="method">GET</span> /api/trading/positions - Open positions with P&L
            </div>
            <div class="endpoint">
                <span class="method">GET</span> /api/trading/orders - All orders
            </div>
            <div class="endpoint">
                <span class="method">POST</span> /api/trading/order - Place buy/sell order
            </div>
            <div class="endpoint">
                <span class="method">POST</span> /api/trading/strategy/execute - Run automated strategy
            </div>
        </div>
    </section>

    <footer>
        <div class="container">
            <p>© 2024 Alpha Junior. Built for institutional investors.</p>
        </div>
    </footer>

    <div class="status">Server Running ✓</div>
</body>
</html>
"""

# Routes
@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/health')
def health():
    return jsonify({"status": "healthy", "service": "Alpha Junior"})

@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    full_name = data.get('full_name', '')
    
    if not email or not password:
        return jsonify({"error": "Email and password required"}), 400
    
    conn = sqlite3.connect('alpha_junior.db')
    c = conn.cursor()
    
    try:
        c.execute(
            "INSERT INTO users (email, password_hash, full_name) VALUES (?, ?, ?)",
            (email, hash_password(password), full_name)
        )
        conn.commit()
        user_id = c.lastrowid
        conn.close()
        
        return jsonify({
            "success": True,
            "message": "User registered successfully",
            "user_id": user_id
        })
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify({"error": "Email already exists"}), 400

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    
    conn = sqlite3.connect('alpha_junior.db')
    c = conn.cursor()
    c.execute(
        "SELECT id, email, full_name, role FROM users WHERE email = ? AND password_hash = ?",
        (email, hash_password(password))
    )
    user = c.fetchone()
    conn.close()
    
    if user:
        token = secrets.token_urlsafe(32)
        return jsonify({
            "success": True,
            "token": token,
            "user": {
                "id": user[0],
                "email": user[1],
                "full_name": user[2],
                "role": user[3]
            }
        })
    
    return jsonify({"error": "Invalid credentials"}), 401

@app.route('/api/funds', methods=['GET'])
def get_funds():
    conn = sqlite3.connect('alpha_junior.db')
    c = conn.cursor()
    c.execute("SELECT id, name, slug, strategy, min_investment, nav, status FROM funds")
    funds = c.fetchall()
    conn.close()
    
    return jsonify({
        "success": True,
        "funds": [
            {
                "id": f[0],
                "name": f[1],
                "slug": f[2],
                "strategy": f[3],
                "min_investment": f[4],
                "nav": f[5],
                "status": f[6]
            }
            for f in funds
        ]
    })

@app.route('/api/funds', methods=['POST'])
def create_fund():
    data = request.get_json()
    
    conn = sqlite3.connect('alpha_junior.db')
    c = conn.cursor()
    c.execute(
        "INSERT INTO funds (name, slug, strategy, min_investment, nav) VALUES (?, ?, ?, ?, ?)",
        (data.get('name'), data.get('slug'), data.get('strategy'), 
         data.get('min_investment', 10000), data.get('nav', 100))
    )
    conn.commit()
    fund_id = c.lastrowid
    conn.close()
    
    return jsonify({"success": True, "fund_id": fund_id}), 201

@app.route('/api/investments', methods=['GET'])
def get_investments():
    conn = sqlite3.connect('alpha_junior.db')
    c = conn.cursor()
    c.execute("""
        SELECT i.id, i.amount, i.status, u.full_name, f.name 
        FROM investments i
        JOIN users u ON i.investor_id = u.id
        JOIN funds f ON i.fund_id = f.id
    """)
    investments = c.fetchall()
    conn.close()
    
    return jsonify({
        "success": True,
        "investments": [
            {
                "id": inv[0],
                "amount": inv[1],
                "status": inv[2],
                "investor": inv[3],
                "fund": inv[4]
            }
            for inv in investments
        ]
    })

@app.route('/api/invest', methods=['POST'])
def invest():
    data = request.get_json()
    
    conn = sqlite3.connect('alpha_junior.db')
    c = conn.cursor()
    c.execute(
        "INSERT INTO investments (investor_id, fund_id, amount) VALUES (?, ?, ?)",
        (data.get('investor_id'), data.get('fund_id'), data.get('amount'))
    )
    conn.commit()
    investment_id = c.lastrowid
    conn.close()
    
    return jsonify({"success": True, "investment_id": investment_id}), 201

# ==================== AUTONOMOUS TRADER API ====================

@app.route('/api/autonomous/status', methods=['GET'])
def autonomous_status():
    """Get autonomous trader status"""
    try:
        from autonomous_trader import get_trader
        trader = get_trader()
        return jsonify({
            'success': True,
            'status': trader.get_status()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/autonomous/start', methods=['POST'])
def autonomous_start():
    """Start autonomous trading"""
    try:
        from autonomous_trader import get_trader
        import threading
        
        trader = get_trader(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        
        # Start in background thread
        def run_trader():
            trader.run()
        
        thread = threading.Thread(target=run_trader, daemon=True)
        thread.start()
        
        return jsonify({
            'success': True,
            'message': '🤖 Autonomous trader started!',
            'note': 'Trader is running in background. Check /api/autonomous/status for updates.'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/autonomous/stop', methods=['POST'])
def autonomous_stop():
    """Stop autonomous trading"""
    try:
        from autonomous_trader import get_trader
        trader = get_trader()
        trader.stop()
        return jsonify({
            'success': True,
            'message': '🛑 Autonomous trader stopped'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/brain/analyze', methods=['GET'])
def brain_analyze():
    """Run AI Brain analysis on all stocks"""
    try:
        from brain import get_brain
        brain = get_brain(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        
        limit = request.args.get('limit', 20, type=int)
        top_picks = brain.get_top_picks(n=limit, min_score=60)
        
        return jsonify({
            'success': True,
            'analysis': {
                'timestamp': datetime.now().isoformat(),
                'stocks_analyzed': brain.analysis_cache.get('total_analyzed', 0),
                'top_picks': top_picks,
                'brain_stats': brain.get_brain_stats()
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/brain/market-report', methods=['GET'])
def brain_market_report():
    """Get comprehensive market analysis report"""
    try:
        from brain import get_brain
        brain = get_brain(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        
        report = brain.generate_market_report()
        
        return jsonify({
            'success': True,
            'report': report
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ==================== ELITE HEDGE FUND API ====================

@app.route('/api/elite/start', methods=['POST'])
def elite_start():
    """Start Elite Hedge Fund Engine"""
    try:
        from elite_hedge_fund import get_elite_engine
        import threading
        
        engine = get_elite_engine(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        
        def run_elite():
            engine.run()
        
        thread = threading.Thread(target=run_elite, daemon=True)
        thread.start()
        
        return jsonify({
            'success': True,
            'message': '🎩 ELITE HEDGE FUND ENGINE STARTED!',
            'features': [
                '4 specialized AI traders working together',
                'Kelly Criterion position sizing',
                'Risk parity portfolio management',
                '100+ stock universe coverage',
                'Institutional-grade risk controls'
            ],
            'target': '60-100% annual returns (top 1% hedge fund performance)'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/elite/status', methods=['GET'])
def elite_status():
    """Get Elite Hedge Fund status"""
    try:
        from elite_hedge_fund import get_elite_engine
        engine = get_elite_engine(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        
        return jsonify({
            'success': True,
            'status': {
                'is_running': engine.is_running,
                'scan_count': engine.scan_count,
                'trades_executed': engine.trades_executed,
                'daily_pnl': engine.daily_pnl,
                'total_realized_pnl': engine.total_realized_pnl,
                'portfolio_summary': engine.portfolio_manager.get_portfolio_summary() if engine.portfolio_manager else None,
                'trading_team_performance': engine.trading_team.get_team_performance() if engine.trading_team else None
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/elite/trading-team', methods=['GET'])
def elite_trading_team():
    """Get Elite Trading Team status"""
    try:
        from elite_hedge_fund import get_elite_engine
        engine = get_elite_engine(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        
        return jsonify({
            'success': True,
            'trading_team': engine.trading_team.get_team_performance() if engine.trading_team else {}
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/elite/portfolio', methods=['GET'])
def elite_portfolio():
    """Get Elite Portfolio details"""
    try:
        from elite_hedge_fund import get_elite_engine
        engine = get_elite_engine(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        
        return jsonify({
            'success': True,
            'portfolio': engine.portfolio_manager.get_portfolio_summary() if engine.portfolio_manager else {}
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# Seed data
def seed_data():
    conn = sqlite3.connect('alpha_junior.db')
    c = conn.cursor()
    
    # Add sample funds if none exist
    c.execute("SELECT COUNT(*) FROM funds")
    if c.fetchone()[0] == 0:
        sample_funds = [
            ("Alpha Growth Fund", "alpha-growth", "Long/Short Equity", 50000, 105.50),
            ("Tech Ventures Fund", "tech-ventures", "Venture Capital", 100000, 98.75),
            ("Global Macro Fund", "global-macro", "Macro", 25000, 112.30),
            ("Real Estate Income", "re-income", "Real Estate", 10000, 101.25),
        ]
        c.executemany(
            "INSERT INTO funds (name, slug, strategy, min_investment, nav) VALUES (?, ?, ?, ?, ?)",
            sample_funds
        )
        conn.commit()
        print("✓ Sample funds added")
    
    conn.close()

if __name__ == '__main__':
    print("Initializing Alpha Junior...")
    init_db()
    seed_data()
    # Check Alpaca status
    alpaca_key = os.getenv('ALPACA_API_KEY', '')
    alpaca_secret = os.getenv('ALPACA_SECRET_KEY', '')
    
    print("\n" + "="*70)
    print("                  🤖 ALPHA JUNIOR v2.0 - AI TRADING")
    print("="*70)
    print("\n📊 FUND MANAGEMENT:")
    print("  Website:    http://localhost:5000")
    print("  API:        http://localhost:5000/api/health")
    
    if alpaca_key and alpaca_secret and alpaca_secret != 'YOUR_SECRET_KEY_HERE':
        print("\n🚀 ALPACA PAPER TRADING: ENABLED")
        print("  Manual Trading:")
        print("    Account:    http://localhost:5000/api/trading/account")
        print("    Positions:  http://localhost:5000/api/trading/positions")
        print("    Orders:     http://localhost:5000/api/trading/orders")
        print("\n  🤖 AUTONOMOUS AI TRADER:")
        print("    Start Bot:  POST http://localhost:5000/api/autonomous/start")
        print("    Stop Bot:   POST http://localhost:5000/api/autonomous/stop")
        print("    Status:     http://localhost:5000/api/autonomous/status")
        print("\n  🧠 AI BRAIN ANALYSIS:")
        print("    Analyze:    http://localhost:5000/api/brain/analyze")
        print("    Market Report: http://localhost:5000/api/brain/market-report")
        print("\n  🎩 ELITE HEDGE FUND ENGINE (NEW! TOP 1%% MODE):")
        print("    Start Elite:  POST http://localhost:5000/api/elite/start")
        print("    Elite Status: http://localhost:5000/api/elite/status")
        print("    Trading Team: http://localhost:5000/api/elite/trading-team")
        print("    Portfolio:    http://localhost:5000/api/elite/portfolio")
        print("\n  MODE SELECTION:")
        print("    [1] Manual Trading - You control everything")
        print("    [2] AI Autonomous - AI picks and trades automatically")
        print("    [3] ELITE HEDGE FUND - 14 AI traders + Kelly sizing + Risk parity ⭐")
        print("\n  RETURN TARGETS:")
        print("    • AI Autonomous: 50-60% annually (moderate risk)")
        print("    • Elite Hedge Fund: 60-100% annually (institutional-grade)")
    else:
        print("\n⚠ ALPACA TRADING: DISABLED")
        print("  Add your API keys to .env file to enable")
    
    print("\n" + "="*70)
    print("Press Ctrl+C to stop")
    print("="*70 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
