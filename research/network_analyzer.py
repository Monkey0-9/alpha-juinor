import networkx as nx
import logging
import random
from typing import List, Dict

logger = logging.getLogger("NetworkAnalyzer")

class CorporateNetworkAnalyzer:
    """
    Analyzes relationships between insiders and institutions.
    Uses NetworkX to detect 'Smart Clusters' of activity.
    """
    def __init__(self):
        self.graph = nx.Graph()

    def ingest_dummy_data(self):
        """
        Load mock data for simulation/verification.
        Graph Nodes: People (Insiders) or Companies.
        Edges: Shared board membership, past employment, or co-investment.
        """
        # Mock: Insiders A, B, C are connected (e.g. board of Comp X)
        self.graph.add_edge("Insider_A", "Insider_B", weight=1.0)
        self.graph.add_edge("Insider_B", "Insider_C", weight=1.0)
        self.graph.add_edge("Insider_A", "Insider_C", weight=1.0)

        # Mock: They all bought "Target_Stock_Z" recently
        self.activity_log = [
            {"person": "Insider_A", "action": "BUY", "symbol": "XYZ", "amount": 50000},
            {"person": "Insider_B", "action": "BUY", "symbol": "XYZ", "amount": 75000},
            {"person": "Insider_C", "action": "BUY", "symbol": "XYZ", "amount": 20000},
            {"person": "Insider_D", "action": "SELL", "symbol": "ABC", "amount": 10000} # Unconnected
        ]

    def detect_anomalous_clusters(self) -> List[Dict]:
        """
        Run clique detection and correlate with recent activity.
        Logic: If > 2 members of a clique BUY the same stock -> Signal.
        """
        # 1. Find Cliques (tightly knit groups)
        cliques = list(nx.find_cliques(self.graph))
        signals = []

        for clique in cliques:
            if len(clique) < 2:
                continue

            # Check if members acted on same stock
            stock_actions = {} # {symbol: [list of buyers]}

            for person in clique:
                # Find recent actions for this person
                # (Inefficient O(N^2) for mock, optimize with dicts for prod)
                for event in self.activity_log:
                    if event["person"] == person and event["action"] == "BUY":
                        sym = event["symbol"]
                        if sym not in stock_actions:
                            stock_actions[sym] = []
                        stock_actions[sym].append(event)

            # Generate Signals
            for sym, events in stock_actions.items():
                if len(events) >= 2: # Threshold: 2+ connected insiders
                    total_vol = sum(e["amount"] for e in events)
                    signals.append({
                        "type": "CLUSTER_BUY",
                        "symbol": sym,
                        "cluster_size": len(events),
                        "total_volume": total_vol,
                        "conviction": min(1.0, len(events) * 0.2 + 0.5) # Simple score
                    })

        return signals

def get_network_analyzer():
    na = CorporateNetworkAnalyzer()
    na.ingest_dummy_data()
    return na
