
from typing import List
from alpha_agents.technical import (
    MomentumAgent, MeanReversionAgent, VolatilityAgent, TrendFollowingAgent,
    BreakoutAgent, RSIDivergenceAgent, MACDCrossoverAgent, BollingerBandwidthAgent
)
from alpha_agents.statistical_fundamental import (
    StatArbAgent, CointegrationAgent, PairsTradingAgent, KalmanFilterAgent,
    HurstExponentAgent, FractalDimensionAgent,
    FundamentalGrowthAgent, ValueInvestingAgent, QualityFactorAgent,
    MacroRegimeAgent, YieldCurveAgent, InflationHedgeAgent
)
from alpha_agents.alternative_advanced import (
    SentimentAnalysisAgent, NewsEventAgent, EarningsSurpriseAgent, OptionsFlowAgent,
    DarkPoolLiquidityAgent, StructuralBreakAgent, RegimeShiftAgent,
    ReinforcementLearningAgent, LSTMSequenceAgent, TransformerAttentionAgent,
    GraphNetworkAgent, AdversarialAttackAgent, EvolutionaryStrategyAgent
)
from alpha_agents.specialized_micro import (
    ESGScoreAgent, SupplyChainAgent, RegulatoryRiskAgent, MergerArbitrageAgent,
    SpinOffAgent, InsiderActivityAgent, ShortInterestAgent,
    OrderBookImbalanceAgent, TradeFlowToxicityAgent, SpreadCaptureAgent, LatencyArbitrageAgent,
    EnsembleVotingAgent, MixtureOfExpertsAgent, MetaLabelingAgent
)

class AlphaRegistry:
    @staticmethod
    def get_all_agents() -> List:
        return [
            # Technical
            MomentumAgent("MomentumAgent"), MeanReversionAgent("MeanReversionAgent"), VolatilityAgent("VolatilityAgent"),
            TrendFollowingAgent("TrendFollowingAgent"), BreakoutAgent("BreakoutAgent"), RSIDivergenceAgent("RSIDivergenceAgent"),
            MACDCrossoverAgent("MACDCrossoverAgent"), BollingerBandwidthAgent("BollingerBandwidthAgent"),
            # Statistical
            StatArbAgent("StatArbAgent"), CointegrationAgent("CointegrationAgent"), PairsTradingAgent("PairsTradingAgent"),
            KalmanFilterAgent("KalmanFilterAgent"), HurstExponentAgent("HurstExponentAgent"), FractalDimensionAgent("FractalDimensionAgent"),
            # Fundamental
            FundamentalGrowthAgent("FundamentalGrowthAgent"), ValueInvestingAgent("ValueInvestingAgent"), QualityFactorAgent("QualityFactorAgent"),
            MacroRegimeAgent("MacroRegimeAgent"), YieldCurveAgent("YieldCurveAgent"), InflationHedgeAgent("InflationHedgeAgent"),
            # Alternative
            SentimentAnalysisAgent("SentimentAnalysisAgent"), NewsEventAgent("NewsEventAgent"), EarningsSurpriseAgent("EarningsSurpriseAgent"),
            OptionsFlowAgent("OptionsFlowAgent"), DarkPoolLiquidityAgent("DarkPoolLiquidityAgent"), StructuralBreakAgent("StructuralBreakAgent"), RegimeShiftAgent("RegimeShiftAgent"),
            # Advanced
            ReinforcementLearningAgent("ReinforcementLearningAgent"), LSTMSequenceAgent("LSTMSequenceAgent"), TransformerAttentionAgent("TransformerAttentionAgent"),
            GraphNetworkAgent("GraphNetworkAgent"), AdversarialAttackAgent("AdversarialAttackAgent"), EvolutionaryStrategyAgent("EvolutionaryStrategyAgent"),
            # Specialized
            ESGScoreAgent("ESGScoreAgent"), SupplyChainAgent("SupplyChainAgent"), RegulatoryRiskAgent("RegulatoryRiskAgent"), MergerArbitrageAgent("MergerArbitrageAgent"),
            SpinOffAgent("SpinOffAgent"), InsiderActivityAgent("InsiderActivityAgent"), ShortInterestAgent("ShortInterestAgent"),
            # Micro
            OrderBookImbalanceAgent("OrderBookImbalanceAgent"), TradeFlowToxicityAgent("TradeFlowToxicityAgent"), SpreadCaptureAgent("SpreadCaptureAgent"), LatencyArbitrageAgent("LatencyArbitrageAgent"),
            # Meta
            EnsembleVotingAgent("EnsembleVotingAgent"), MixtureOfExpertsAgent("MixtureOfExpertsAgent"), MetaLabelingAgent("MetaLabelingAgent")
        ]
