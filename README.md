# onnx2mlx

![Muna logo](https://raw.githubusercontent.com/muna-ai/.github/main/banner.png)

Convert ONNX models into [MLX](https://github.com/ml-explore/mlx) callables for accelerating inference on Apple Silicon.

## Setup Instructions
Open a terminal and run the following command:
```bash
# Install onnx2mlx
$ pip install --upgrade onnx2mlx
```

## Converting from ONNX to MLX
Use the `onnx2mlx` function to create a callable that uses MLX to run the model:
```py
import mlx.core as mx
import onnx
from onnx2mlx import onnx2mlx

# Load an ONNX model
model = onnx.load("model.onnx")

# Convert to MLX
model_mlx = onnx2mlx(onnx_model)

# Run the MLX model
outputs = model_mlx(mx.array(...))
```

## Useful Links
- [Join our Slack community](https://muna.ai/slack).
- [Check out the docs](https://docs.muna.ai/onnx2mlx).
- [Read our blog](https://muna.ai/blog).
- Reach out to us at [hi@muna.ai](mailto:hi@muna.ai).