import os
import sys
import logging
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def export_all_onnx():
    export_lstm()
    export_transformer()
    export_ppo()
    logger.info("All models exported to ONNX")


def export_lstm():
    from train_lstm import PriceLSTM

    pt_path = "nexus/models/lstm_price_model.pt"
    onnx_path = "nexus/models/lstm_price_model.onnx"
    if not os.path.exists(pt_path):
        logger.warning("LSTM weights not found, skipping")
        return
    model = PriceLSTM(input_size=20, hidden_size=128, num_layers=3).to(device)
    model.load_state_dict(torch.load(pt_path, map_location=device))
    model.eval()
    dummy = torch.randn(1, 60, 20).to(device)
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["sequence"],
        output_names=["score"],
        dynamic_axes={"sequence": {0: "batch", 1: "seq_len"}},
        opset_version=17,
    )
    logger.info(f"LSTM exported to {onnx_path}")


def export_transformer():
    from train_transformer import MultiAssetTransformer

    pt_path = "nexus/models/transformer_multi_asset.pt"
    onnx_path = "nexus/models/transformer_multi_asset.onnx"
    if not os.path.exists(pt_path):
        logger.warning("Transformer weights not found, skipping")
        return
    model = MultiAssetTransformer(n_assets=10).to(device)
    model.load_state_dict(torch.load(pt_path, map_location=device))
    model.eval()
    dummy = torch.randn(1, 10, 60, 5).to(device)
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["multi_asset_sequence"],
        output_names=["asset_scores"],
        dynamic_axes={"multi_asset_sequence": {0: "batch"}},
        opset_version=17,
    )
    logger.info(f"Transformer exported to {onnx_path}")


def export_ppo():
    from train_ppo import PolicyNetwork

    pt_path = "nexus/models/ppo_trade_executor.pt"
    onnx_path = "nexus/models/ppo_trade_executor.onnx"
    if not os.path.exists(pt_path):
        logger.warning("PPO weights not found, skipping")
        return
    model = PolicyNetwork(state_dim=20).to(device)
    model.load_state_dict(torch.load(pt_path, map_location=device))
    model.eval()
    dummy = torch.randn(1, 20).to(device)
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["state"],
        output_names=["action_probs", "state_value"],
        dynamic_axes={"state": {0: "batch"}},
        opset_version=17,
    )
    logger.info(f"PPO exported to {onnx_path}")


if __name__ == "__main__":
    export_all_onnx()
