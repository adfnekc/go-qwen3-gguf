"""
Verify Q8_0 dequantization against Go implementation.

Reads Q8_0 tensors from GGUF using Python's gguf library,
dequantizes manually, and reports statistics that can be
compared with Go's check_weight_stats.go output.
"""
import struct
import sys

import numpy as np
import gguf


def dequantize_q8_0(data: np.ndarray, num_elements: int) -> np.ndarray:
    """Dequantize Q8_0 data (fp16 scale + 32 int8 per block)."""
    block_size = 32
    block_bytes = 34
    num_blocks = (num_elements + block_size - 1) // block_size

    result = np.zeros(num_elements, dtype=np.float32)

    raw = data.tobytes()
    for b in range(num_blocks):
        off = b * block_bytes
        # fp16 scale (2 bytes, little-endian)
        scale = struct.unpack('<e', raw[off:off + 2])[0]
        # 32 int8 values
        qs = struct.unpack(f'<{block_size}b', raw[off + 2:off + 2 + block_size])
        for i in range(block_size):
            idx = b * block_size + i
            if idx >= num_elements:
                break
            result[idx] = float(qs[i]) * scale

    return result


def tensor_stats(name: str, tensor, print_first_n: int = 5):
    """Print statistics for a Q8_0 tensor."""
    shape = list(tensor.shape)
    num_elements = int(np.prod(shape))

    print(f"\n=== {name} ===")
    print(f"  Shape: {shape}")
    print(f"  Type: {tensor.tensor_type}")
    print(f"  Raw bytes: {tensor.data.nbytes}")
    print(f"  Elements: {num_elements}")

    # Dequantize
    values = dequantize_q8_0(tensor.data, num_elements)

    # Reshape to match GGUF layout [dim0, dim1]
    values_2d = values.reshape(shape)

    rms = float(np.sqrt(np.mean(values ** 2)))
    print(f"  Overall RMS: {rms:.6f}")
    print(f"  Min: {float(values.min()):.6f}")
    print(f"  Max: {float(values.max()):.6f}")

    # Per-row RMS (along dim0)
    row_rms = np.sqrt(np.mean(values_2d ** 2, axis=1))
    avg_row_rms = float(np.mean(row_rms))
    print(f"  Avg row RMS: {avg_row_rms:.6f}")

    # First few values of row 0
    print(f"  Row 0 first {print_first_n}: {values_2d[0, :print_first_n].tolist()}")
    print(f"  Row 0 RMS: {float(np.sqrt(np.mean(values_2d[0] ** 2))):.6f}")


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "/mnt/d/model/Qwen3-0.6B-Q8_0.gguf"
    reader = gguf.GGUFReader(model_path)

    # Check specific tensors
    for layer in range(5):
        for tname in ["ffn_gate", "ffn_up", "ffn_down"]:
            name = f"blk.{layer}.{tname}.weight"
            for tensor in reader.tensors:
                if tensor.name == name:
                    tensor_stats(name, tensor)
                    break

    # Also check attention weights for layer 2
    for tname in ["attn_q", "attn_k", "attn_v", "attn_output"]:
        name = f"blk.2.{tname}.weight"
        for tensor in reader.tensors:
            if tensor.name == name:
                tensor_stats(name, tensor, print_first_n=10)
                break


if __name__ == "__main__":
    main()
