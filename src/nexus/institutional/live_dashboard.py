"""
=============================================================================
LIVE PAPER TRADING DASHBOARD
=============================================================================
Real-time monitoring dashboard for live trading system
Provides real-time updates on news, sentiment, trades, and portfolio
"""

import json
import time
from datetime import datetime
from pathlib import Path
from collections import deque
from dataclasses import dataclass


@dataclass
class DashboardUpdater:
    """Updates live dashboard with trading information."""
    
    def __init__(self, output_file: Path = None):
        self.output_file = output_file or Path("live_dashboard.json")
        self.news_cache = deque(maxlen=20)
        self.alerts_cache = deque(maxlen=50)
        self.trades_cache = deque(maxlen=100)
        
    def update_dashboard(self, news: list, sentiment: dict, alerts: list, portfolio: dict, market_data: dict):
        """Update the live dashboard with current data."""
        
        dashboard_data = {
            "timestamp": datetime.now().isoformat(),
            "status": "monitoring",
            "news_summary": {
                "total_articles": len(news),
                "bullish_count": len([a for a in news if a.get('sentiment') == 'POSITIVE']),
                "bearish_count": len([a for a in news if a.get('sentiment') == 'NEGATIVE']),
                "latest_articles": [n.to_dict() for n in news[-5:]] if news else []
            },
            "sentiment_summary": {
                "bullish_symbols": [s for s, v in sentiment.items() if v.get('sentiment_direction') == 'bullish'],
                "bearish_symbols": [s for s, v in sentiment.items() if v.get('sentiment_direction') == 'bearish'],
                "neutral_symbols": [s for s, v in sentiment.items() if v.get('sentiment_direction') == 'neutral']
            },
            "trading_alerts": {
                "total": len(alerts),
                "opportunities": len([a for a in alerts if a.get('alert_type') == 'opportunity']),
                "risks": len([a for a in alerts if a.get('alert_type') == 'risk']),
                "recent_alerts": [a.to_dict() for a in alerts[-10:]] if alerts else []
            },
            "portfolio": {
                "cash": portfolio.get('cash', 0),
                "positions": len(portfolio.get('positions', {})),
                "total_value": portfolio.get('value', 0),
                "positions_detail": portfolio.get('positions', {})
            },
            "market_data": market_data
        }
        
        # Write to file
        with open(self.output_file, 'w') as f:
            json.dump(dashboard_data, f, indent=2, default=str)
        
        return dashboard_data


class DashboardServer:
    """Simple HTTP server for live dashboard (optional)."""
    
    HTML_TEMPLATE = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Nexus Live Trading Monitor</title>
        <style>
            body {{ font-family: monospace; background: #1e1e1e; color: #00ff00; padding: 20px; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .header {{ text-align: center; margin-bottom: 30px; border-bottom: 2px solid #00ff00; padding-bottom: 10px; }}
            .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
            .widget {{ border: 1px solid #00ff00; padding: 15px; margin-bottom: 15px; }}
            .widget-title {{ font-weight: bold; margin-bottom: 10px; color: #ffff00; }}
            .bullish {{ color: #00ff00; }}
            .bearish {{ color: #ff0000; }}
            .neutral {{ color: #888888; }}
            .metric {{ display: flex; justify-content: space-between; padding: 5px 0; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #444; }}
            .update-time {{ text-align: center; color: #666; font-size: 0.9em; }}
        </style>
        <script>
            function loadDashboard() {{
                fetch('live_dashboard.json')
                    .then(r => r.json())
                    .then(data => updateUI(data))
                    .catch(e => console.error('Error loading dashboard:', e));
            }}
            
            setInterval(loadDashboard, 5000);  // Update every 5 seconds
            loadDashboard();
        </script>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>⚡ Nexus Institutional - Live Trading Monitor</h1>
                <p>Real-Time Paper Trading with News & Event Monitoring</p>
                <div class="update-time">Last update: <span id="timestamp">--</span></div>
            </div>
            
            <div class="grid">
                <div class="widget">
                    <div class="widget-title">📰 News Summary</div>
                    <div class="metric">
                        <span>Total Articles:</span>
                        <span id="news-total">0</span>
                    </div>
                    <div class="metric">
                        <span class="bullish">Bullish:</span>
                        <span class="bullish" id="news-bullish">0</span>
                    </div>
                    <div class="metric">
                        <span class="bearish">Bearish:</span>
                        <span class="bearish" id="news-bearish">0</span>
                    </div>
                </div>
                
                <div class="widget">
                    <div class="widget-title">📊 Trading Alerts</div>
                    <div class="metric">
                        <span>Total Alerts:</span>
                        <span id="alerts-total">0</span>
                    </div>
                    <div class="metric">
                        <span class="bullish">Opportunities:</span>
                        <span class="bullish" id="alerts-opportunities">0</span>
                    </div>
                    <div class="metric">
                        <span class="bearish">Risks:</span>
                        <span class="bearish" id="alerts-risks">0</span>
                    </div>
                </div>
                
                <div class="widget">
                    <div class="widget-title">📈 Portfolio Status</div>
                    <div class="metric">
                        <span>Cash:</span>
                        <span id="portfolio-cash">$0</span>
                    </div>
                    <div class="metric">
                        <span>Positions:</span>
                        <span id="portfolio-positions">0</span>
                    </div>
                    <div class="metric">
                        <span>Portfolio Value:</span>
                        <span id="portfolio-value">$0</span>
                    </div>
                </div>
                
                <div class="widget">
                    <div class="widget-title">💹 Market Sentiment</div>
                    <div class="metric">
                        <span class="bullish">Bullish:</span>
                        <span class="bullish" id="sentiment-bullish">--</span>
                    </div>
                    <div class="metric">
                        <span class="bearish">Bearish:</span>
                        <span class="bearish" id="sentiment-bearish">--</span>
                    </div>
                    <div class="metric">
                        <span class="neutral">Neutral:</span>
                        <span class="neutral" id="sentiment-neutral">--</span>
                    </div>
                </div>
            </div>
            
            <div class="widget">
                <div class="widget-title">🔔 Recent Trading Alerts</div>
                <table>
                    <thead>
                        <tr>
                            <th>Time</th>
                            <th>Symbol</th>
                            <th>Type</th>
                            <th>Action</th>
                            <th>Confidence</th>
                        </tr>
                    </thead>
                    <tbody id="alerts-table">
                        <tr><td colspan="5">Loading...</td></tr>
                    </tbody>
                </table>
            </div>
            
            <div class="widget">
                <div class="widget-title">📰 Latest News</div>
                <div id="news-list">Loading...</div>
            </div>
        </div>
        
        <script>
            function updateUI(data) {{
                document.getElementById('timestamp').innerText = new Date(data.timestamp).toLocaleTimeString();
                
                // News
                document.getElementById('news-total').innerText = data.news_summary.total_articles;
                document.getElementById('news-bullish').innerText = data.news_summary.bullish_count;
                document.getElementById('news-bearish').innerText = data.news_summary.bearish_count;
                
                // Alerts
                document.getElementById('alerts-total').innerText = data.trading_alerts.total;
                document.getElementById('alerts-opportunities').innerText = data.trading_alerts.opportunities;
                document.getElementById('alerts-risks').innerText = data.trading_alerts.risks;
                
                // Portfolio
                document.getElementById('portfolio-cash').innerText = '$' + data.portfolio.cash.toLocaleString();
                document.getElementById('portfolio-positions').innerText = data.portfolio.positions;
                document.getElementById('portfolio-value').innerText = '$' + data.portfolio.total_value.toLocaleString();
                
                // Sentiment
                document.getElementById('sentiment-bullish').innerText = data.sentiment_summary.bullish_symbols.join(', ');
                document.getElementById('sentiment-bearish').innerText = data.sentiment_summary.bearish_symbols.join(', ');
                document.getElementById('sentiment-neutral').innerText = data.sentiment_summary.neutral_symbols.join(', ');
                
                // Latest alerts
                let alertsHtml = '';
                for (let alert of data.trading_alerts.recent_alerts.slice(0, 10)) {{
                    alertsHtml += `
                        <tr>
                            <td>${{new Date(alert.timestamp).toLocaleTimeString()}}</td>
                            <td>${{alert.symbol}}</td>
                            <td>${{alert.type}}</td>
                            <td>${{alert.action}}</td>
                            <td>${{{(alert.confidence * 100).toFixed(0)}}%</td>
                        </tr>
                    `;
                }}
                document.getElementById('alerts-table').innerHTML = alertsHtml || '<tr><td colspan="5">No alerts</td></tr>';
                
                // Latest news
                let newsHtml = '';
                for (let article of data.news_summary.latest_articles.slice(0, 5)) {{
                    newsHtml += `
                        <div style="margin-bottom: 10px; padding-bottom: 10px; border-bottom: 1px solid #444;">
                            <div><strong>${{article.title}}</strong></div>
                            <div style="font-size: 0.9em; color: #aaa;">
                                ${{article.source}} | ${{article.sentiment}}
                            </div>
                        </div>
                    `;
                }}
                document.getElementById('news-list').innerHTML = newsHtml || 'No news';
            }}
        </script>
    </body>
    </html>
    """


def create_dashboard_html(output_file: Path = Path("live_dashboard.html")):
    """Create a standalone HTML dashboard."""
    with open(output_file, 'w') as f:
        f.write(DashboardServer.HTML_TEMPLATE)
    print(f"Dashboard created: {output_file}")
    print(f"Run 'python -m http.server 8000' and visit http://localhost:8000/live_dashboard.html")


if __name__ == "__main__":
    create_dashboard_html()
