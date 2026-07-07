"""HRF feature flags and defaults."""

# Feature toggles
ENABLE_SIGMOID_KERNEL = True
ENABLE_WAVELET_RESONANCE = False

# Sigmoid kernel defaults for lightweight SVC
SIGMOID_KERNEL_DEFAULTS = {
    'C': 1.0,
    'gamma': 'scale',
    'coef0': 0.0,
}

# Wavelet resonance defaults
WAVELET_DEFAULTS = {
    'wavelet': 'db1',
    'level': 1,
}
