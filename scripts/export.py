import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import os
import onnx
import onnxruntime as ort
import numpy as np

from src.train.trainer import SquatFormTrainer

@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def export_model(cfg: DictConfig) -> None:
    print("Starting model export...")
    print(OmegaConf.to_yaml(cfg))

    # --- Load Model ---
    checkpoint_path = cfg.get('checkpoint_path', None)
    if not checkpoint_path:
        raise ValueError("Please provide a 'checkpoint_path' in the config or as a CLI argument to load the model.")

    print(f"Loading model from checkpoint: {checkpoint_path}")
    model = SquatFormTrainer.load_from_checkpoint(checkpoint_path, config=cfg, class_weights=None)
    model.eval() # Set model to evaluation mode
    model.freeze() # Freeze model parameters

    # --- Export Model ---
    output_path = cfg.export.output_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    dummy_input = torch.randn(cfg.export.input_shape) # (batch_size, sequence_length, input_dim)

    if cfg.export.format == "onnx":
        print(f"Exporting model to ONNX format at {output_path}...")
        torch.onnx.export(
            model.model, # Export the underlying nn.Module, not the LightningModule
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11, # Common opset version
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size', 1: 'sequence_length'}, # Allow dynamic batch and sequence length
                          'output': {0: 'batch_size'}}
        )
        print("ONNX export complete. Validating ONNX model...")
        
        # Validate ONNX model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model check passed.")

        # Optional: Run a quick inference with ONNX Runtime to verify
        print("Running ONNX Runtime inference for verification...")
        ort_session = ort.InferenceSession(output_path)
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
        ort_outputs = ort_session.run(None, ort_inputs)
        print(f"ONNX Runtime output shape: {ort_outputs[0].shape}")
        print("ONNX Runtime verification complete.")

    elif cfg.export.format == "tensorflowjs":
        print("TensorFlow.js export is not directly supported by PyTorch's export tools.")
        print("You would typically convert ONNX to TensorFlow SavedModel, then to TensorFlow.js.")
        print("Skipping TensorFlow.js export for now.")
        # Placeholder for future TensorFlow.js conversion
        # import onnx_tf
        # from onnx_tf.backend import prepare
        # tf_rep = prepare(onnx_model)
        # tf_rep.export_graph("path/to/tf_saved_model")
        # then use tensorflowjs_converter
    else:
        raise ValueError(f"Unsupported export format: {cfg.export.format}")

    print("Model export process finished.")

if __name__ == "__main__":
    export_model()
