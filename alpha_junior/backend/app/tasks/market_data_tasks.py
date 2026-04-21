"""
Celery tasks for market data operations
Periodic NAV calculations, benchmark updates, economic indicators
"""

import logging
from datetime import datetime, date
from decimal import Decimal

from celery import Celery
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.session import AsyncSessionLocal
from app.services.market_data import market_data_service
from app.models.fund import Fund, FundPerformance
from app.models.audit import AuditLog, AuditAction

logger = logging.getLogger(__name__)

# Initialize Celery
# In production, this would be configured with Redis/RabbitMQ
celery_app = Celery(
    "alpha_junior",
    broker=settings.celery_broker,
    backend=settings.celery_broker,
)


@celery_app.task
async def update_fund_nav_task(fund_id: int):
    """
    Calculate and update fund NAV from holdings
    Triggered after market close or on-demand
    """
    async with AsyncSessionLocal() as db:
        try:
            # Get fund
            result = await db.execute(
                select(Fund).where(Fund.id == fund_id, Fund.is_deleted == False)
            )
            fund = result.scalar_one_or_none()
            
            if not fund:
                logger.warning(f"Fund {fund_id} not found")
                return
            
            # In production, fetch actual holdings from database
            # For now, simulate with example holdings
            example_holdings = [
                {"symbol": "AAPL", "quantity": 100, "asset_type": "stock"},
                {"symbol": "MSFT", "quantity": 50, "asset_type": "stock"},
                {"symbol": "BTC", "quantity": 2, "asset_type": "crypto"},
                {"symbol": "USD", "quantity": 50000, "asset_type": "cash"},
            ]
            
            # Calculate NAV
            nav = await market_data_service.calculate_fund_nav(example_holdings)
            
            if nav:
                old_nav = fund.nav
                fund.nav = nav
                fund.nav_updated_at = datetime.utcnow()
                
                # Create performance record
                performance = FundPerformance(
                    fund_id=fund_id,
                    date=date.today(),
                    nav=nav,
                    # Calculate returns if we have previous NAV
                )
                db.add(performance)
                
                # Audit log
                audit = AuditLog(
                    actor_id=None,  # System action
                    action=AuditAction.FUND_NAV_UPDATED,
                    target_type="fund",
                    target_id=fund_id,
                    payload={
                        "before": {"nav": str(old_nav)},
                        "after": {"nav": str(nav)},
                        "source": "market_data_api"
                    }
                )
                db.add(audit)
                
                await db.commit()
                logger.info(f"Updated NAV for fund {fund_id}: {nav}")
            else:
                logger.error(f"Failed to calculate NAV for fund {fund_id}")
        
        except Exception as e:
            logger.error(f"Error updating NAV for fund {fund_id}: {e}")
            await db.rollback()
            raise


@celery_app.task
async def update_all_fund_navs_task():
    """
    Update NAV for all active funds
    Scheduled to run daily at market close (4:30 PM ET)
    """
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(Fund.id).where(
                Fund.status == "active",
                Fund.is_deleted == False
            )
        )
        fund_ids = [row[0] for row in result.all()]
    
    # Queue individual tasks
    for fund_id in fund_ids:
        update_fund_nav_task.delay(fund_id)
    
    logger.info(f"Queued NAV updates for {len(fund_ids)} funds")


@celery_app.task
async def fetch_benchmark_data_task():
    """
    Fetch benchmark index data (S&P 500, NASDAQ, etc.)
    Store for performance comparison
    """
    try:
        benchmarks = await market_data_service.get_benchmark_quotes()
        
        # In production, store in database
        # For now, just log
        for benchmark in benchmarks:
            logger.info(
                f"Benchmark: {benchmark.name} = {benchmark.price} "
                f"({benchmark.daily_return:+.2f}%)"
            )
        
        return {
            "benchmarks": len(benchmarks),
            "timestamp": datetime.utcnow().isoformat(),
        }
    
    except Exception as e:
        logger.error(f"Error fetching benchmark data: {e}")
        raise


@celery_app.task
async def fetch_economic_indicators_task():
    """
    Fetch key economic indicators from FRED
    Used for macro analysis and risk assessment
    """
    try:
        indicators = await market_data_service.get_key_economic_indicators()
        
        logger.info(f"Economic indicators: {indicators}")
        
        # In production, store in economic_data table
        return indicators
    
    except Exception as e:
        logger.error(f"Error fetching economic indicators: {e}")
        raise


@celery_app.task
async def fetch_crypto_market_data_task():
    """
    Fetch top 100 cryptocurrency market data
    For crypto funds and diversification analysis
    """
    try:
        crypto_data = await market_data_service.get_crypto_market_data(
            vs_currency="usd",
            per_page=100
        )
        
        if crypto_data:
            logger.info(f"Fetched {len(crypto_data)} cryptocurrencies")
            
            # In production, store in crypto_prices table
            return {
                "count": len(crypto_data),
                "top_3": [c["name"] for c in crypto_data[:3]],
            }
    
    except Exception as e:
        logger.error(f"Error fetching crypto data: {e}")
        raise


@celery_app.task
async def fetch_financial_news_task():
    """
    Fetch financial news for sentiment analysis
    Store for fund manager insights
    """
    try:
        news = await market_data_service.get_financial_news(
            query="hedge fund OR private equity OR venture capital",
            page_size=20
        )
        
        if news:
            logger.info(f"Fetched {len(news)} financial news articles")
            
            # In production:
            # 1. Store articles in news_articles table
            # 2. Run sentiment analysis
            # 3. Send relevant news to fund managers
            
            return {
                "count": len(news),
                "headlines": [a["title"] for a in news[:5]],
            }
    
    except Exception as e:
        logger.error(f"Error fetching financial news: {e}")
        raise


# Celery Beat schedule (periodic tasks)
celery_app.conf.beat_schedule = {
    "update-all-fund-navs": {
        "task": "app.tasks.market_data_tasks.update_all_fund_navs_task",
        "schedule": "30 16 * * 1-5",  # 4:30 PM ET, Monday-Friday
    },
    "fetch-benchmark-data": {
        "task": "app.tasks.market_data_tasks.fetch_benchmark_data_task",
        "schedule": 300.0,  # Every 5 minutes during market hours
    },
    "fetch-economic-indicators": {
        "task": "app.tasks.market_data_tasks.fetch_economic_indicators_task",
        "schedule": 3600.0,  # Every hour
    },
    "fetch-crypto-data": {
        "task": "app.tasks.market_data_tasks.fetch_crypto_market_data_task",
        "schedule": 300.0,  # Every 5 minutes (crypto markets 24/7)
    },
    "fetch-financial-news": {
        "task": "app.tasks.market_data_tasks.fetch_financial_news_task",
        "schedule": 1800.0,  # Every 30 minutes
    },
}
