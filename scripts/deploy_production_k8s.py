
import os
import sys
import subprocess
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PROD_DEPLOY")

def check_kubectl():
    try:
        subprocess.run(["kubectl", "version", "--client"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def deploy():
    logger.info("Starting Institutional Production Deployment...")
    
    if not check_kubectl():
        logger.error("kubectl not found. Institutional systems require Kubernetes.")
        logger.info("To install Kubernetes locally for testing, use Minikube or Docker Desktop.")
        return False

    # Check if we have a valid context
    try:
        context = subprocess.check_output(["kubectl", "config", "current-context"]).decode().strip()
        logger.info(f"Using Kubernetes context: {context}")
    except subprocess.CalledProcessError:
        logger.error("No Kubernetes context found. Run 'kubectl config use-context <name>'")
        return False

    manifests = [
        "infrastructure/kubernetes/deployment.yaml",
        "infrastructure/monitoring/observability-stack.yaml",
        "infrastructure/database/timescaledb-cluster.yaml"
    ]

    for manifest in manifests:
        if os.path.exists(manifest):
            logger.info(f"Applying manifest: {manifest}")
            try:
                subprocess.run(["kubectl", "apply", "-f", manifest], check=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to apply {manifest}: {e}")
                return False
        else:
            logger.warning(f"Manifest not found: {manifest}")

    logger.info("Deployment successful! Monitoring cluster health...")
    time.sleep(5)
    subprocess.run(["kubectl", "get", "pods", "-n", "quant-fund"])
    
    return True

if __name__ == "__main__":
    success = deploy()
    if not success:
        sys.exit(1)
