import traceback
import logging
import numpy as np
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def run_agent(agent: Any, *args, **kwargs) -> Dict[str, Any]:
    """
    Runs an agent and returns a safe, standardized result dict.
    Contract:
    {
        'ok': bool,
        'mu': float|None,
        'sigma': float|None,
        'confidence': float,
        'error': str|None
    }
    """
    try:
        # Standardize call
        # Robust signature handling:
        # 1. AlphaAgents use evaluate(symbol, data)
        # 2. AlphaFamilies use generate_signal(data)

        if hasattr(agent, 'evaluate'):
            # If the first arg is NOT a symbol (string), we might be missing it
            # But we'll trust the caller to provide it if they use evaluate
            res = agent.evaluate(*args, **kwargs)
        else:
            # For generate_signal, if the first arg is a symbol (string), skip it
            agent_args = args
            if args and isinstance(args[0], str):
                agent_args = args[1:]
            res = agent.generate_signal(*agent_args, **kwargs)

        # Elite-Tier Upgrade: Enforce AlphaDistribution Contract
        # Elite-Tier Upgrade: Enforce AlphaDistribution Contract
        try:
            from contracts import AlphaDistribution
            from alpha_families.normalization import AlphaNormalizer

            normalizer = AlphaNormalizer()

            # Cases:
            # 1. Agent returns AlphaDistribution object (Goal)
            # 2. Agent returns dict (Legacy) -> Convert
            # 3. Agent returns float (Legacy) -> Convert
            # 4. Agent returns AgentResult (Legacy Agent) -> Convert

            dist_dict = None

            if isinstance(res, AlphaDistribution):
                # Convert to dict for repair
                dist_dict = {
                    'mu': res.mu,
                    'sigma': res.sigma,
                    'p_loss': res.p_loss,
                    'cvar_95': res.cvar_95,
                    'confidence': res.confidence
                }
            elif hasattr(res, 'mu') and hasattr(res, 'confidence'): # AgentResult or similar
                 dist_dict = {
                    'mu': float(res.mu),
                    'sigma': float(res.sigma) if hasattr(res, 'sigma') else 0.01,
                    'confidence': float(res.confidence),
                    # Heuristics for others
                    'p_loss': 0.5 - (float(res.mu)*2),
                    'cvar_95': -0.05
                 }
            elif isinstance(res, dict):
                mu = float(res.get('mu', res.get('signal', 0.0)))
                sigma = float(res.get('sigma', 0.01)) # Default small sigma
                confidence = float(res.get('confidence', 1.0))
                dist_dict = {
                    'mu': mu,
                    'sigma': sigma,
                    'confidence': confidence,
                    'p_loss': 0.5 - (mu * 2),
                    'cvar_95': -abs(sigma * 1.645)
                }
            elif isinstance(res, (float, int)):
                val = float(res)
                dist_dict = {
                    'mu': val,
                    'sigma': 0.01,
                    'confidence': 1.0,
                    'p_loss': 0.5,
                    'cvar_95': -0.01
                }

            # REPAIR PIPELINE
            current_price = 1.0
            # Try to extract price from args for specific normalization
            if len(args) > 1 and hasattr(args[1], 'iloc'): # Likely (symbol, df)
                 try:
                     current_price = float(args[1]['Close'].iloc[-1])
                 except: pass
            elif len(args) > 0 and hasattr(args[0], 'iloc'): # Likely (df)
                 try:
                     current_price = float(args[0]['Close'].iloc[-1])
                 except: pass

            if dist_dict:
                dist_dict = normalizer.repair_distribution(dist_dict, price=current_price)
                dist = AlphaDistribution(**dist_dict)
            else:
                dist = None


            # VALIDATION GATE
            if dist:
                if not dist.validate():
                    logger.error(f"[AGENT_RUNNER] Agent {agent_name} output failed AlphaDistribution validation")
                    return {
                        'ok': False,
                        'mu': 0.0,
                        'error': "CONTRACT_VIOLATION"
                    }

                # Success - return standardized dict for now (integration bridge)
                # In future this should return the object directly
                return {
                    'ok': True,
                    'mu': dist.mu,
                    'sigma': dist.sigma,
                    'confidence': dist.confidence,
                    'object': dist, # Pass full object
                    'error': None
                }
            else:
                 logger.error(f"[AGENT_RUNNER] Agent {agent_name} returned unparseable output: {type(res)}")
                 return {'ok': False, 'mu': 0.0, 'error': "INVALID_OUTPUT_TYPE"}

        except Exception as e_convert:
             logger.error(f"[AGENT_RUNNER] Contract conversion failed: {e_convert}")
             return {'ok': False, 'mu': 0.0, 'error': f"CONTRACT_ERROR: {e_convert}"}

    except Exception as e:
        # Robust name extraction
        try:
            agent_name = agent.__class__.__name__
        except:
            agent_name = str(agent)

        logger.warning(f"Agent {agent_name} failed: {e}")
        return {
            'ok': False,
            'mu': None,
            'sigma': None,
            'confidence': 0.0,
            'error': traceback.format_exc()
        }
