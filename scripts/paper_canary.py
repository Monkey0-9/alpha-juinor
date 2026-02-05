
import sys
import os
import time
import logging

# Add root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Moved import inside main/function to catch ImportError
# from orchestration.live_decision_loop import LiveDecisionLoop

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PaperCanary")

def run_canary():
    try:
        from orchestration.live_decision_loop import LiveDecisionLoop

        class LimitedLiveLoop(LiveDecisionLoop):
            """Subclass to run for a limited number of ticks."""
            def __init__(self, max_ticks=5, **kwargs):
                super().__init__(**kwargs)
                self.max_ticks = max_ticks
                self.ticks = 0

            def run(self):
                self.running = True
                logger.info(f"Starting Paper Canary for {self.max_ticks} ticks...")
                try:
                    while self.running and self.ticks < self.max_ticks:
                        self.ticks += 1
                        logger.info(f"Tick {self.ticks}/{self.max_ticks}")

                        # Run logic
                        if self._should_refresh_data():
                            self._refresh_market_data()

                        if self._is_market_hours():
                            self._compute_signals()

                        time.sleep(1.0) # Speed up for test

                    logger.info("Paper Canary completed successfully.")

                except Exception as e:
                    import traceback
                    logger.critical(f"Paper Canary crashed: {e}")
                    with open("runtime/canary_error.log", "w") as f:
                        traceback.print_exc(file=f)
                    raise

        canary = LimitedLiveLoop(
            max_ticks=5,
            tick_interval=1.0,
            data_refresh_interval_min=1,
            paper_mode=True,
            market_hours_only=False # Force run even if "closed"
        )
        canary.run()

    except Exception as e:
        import traceback
        logger.critical(f"Startup crash: {e}")
        with open("runtime/canary_error.log", "w") as f:
            traceback.print_exc(file=f)
        sys.exit(1)

if __name__ == "__main__":
    run_canary()
