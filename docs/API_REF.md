# HolographicSoulUnit API Reference

## Methods

### `_apply_projection(X)`
Projects raw input into the resonance space.
- **X**: `torch.Tensor` of shape `(batch_size, features)`.
- **Returns**: `torch.Tensor` of projected features.

### `_update_resonance_kernels(y)`
Updates internal kernels based on target signals.
- **y**: `torch.Tensor` of target labels or signal values...