# Golden reference data (spec 13)

Committed `.npz` files are loaded by `tests/fixed_point/test_golden_data.py`:

- `requantize_basic.npz` — `requantize_int` half-to-even
- `conv2d_basic.npz` — Conv2d INT16 kernel
- `round_shift_basic.npz` — signed round-shift

Regenerate after intentional algorithm changes:

```bash
export PYTHONPATH=/path/to/aimet_rx
python3 tests/fixed_point/data/generate_goldens.py
```
