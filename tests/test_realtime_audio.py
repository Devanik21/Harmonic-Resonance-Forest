"""
tests/test_realtime_audio.py

Unit tests for the block-based real-time audio engine: effect correctness,
state continuity across block boundaries, thread-safe live parameter
updates, and a real-time headroom benchmark.

Run from repo root: pytest tests/test_realtime_audio.py -v
"""

import sys
import threading
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from realtime_audio import (  # noqa: E402
    BiquadFilter,
    Delay,
    EffectChain,
    Gain,
    StreamProcessor,
    Tremolo,
    iter_blocks,
)

SR = 44100
BLOCK = 256


def sine(freq, duration=0.5, sr=SR):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def run_stream(chain, signal, block_size=BLOCK):
    proc = StreamProcessor(chain, block_size=block_size)
    out = np.concatenate(list(proc.process_stream(iter_blocks(signal, block_size))))
    return out[: len(signal)], proc


def dominant_freq(signal, sr=SR):
    spectrum = np.abs(np.fft.rfft(signal))
    return np.fft.rfftfreq(len(signal), 1 / sr)[np.argmax(spectrum)]


class TestIterBlocks:
    def test_exact_division(self):
        blocks = list(iter_blocks(np.arange(1024, dtype=np.float32), 256))
        assert len(blocks) == 4
        assert all(len(b) == 256 for b in blocks)

    def test_pad_last(self):
        blocks = list(iter_blocks(np.ones(300, dtype=np.float32), 256))
        assert len(blocks) == 2
        assert blocks[1][:44].sum() == 44  # data
        assert blocks[1][44:].sum() == 0  # padding

    def test_drop_last(self):
        blocks = list(iter_blocks(np.ones(300, dtype=np.float32), 256, pad_last=False))
        assert len(blocks) == 1


class TestGain:
    def test_attenuation(self):
        out, _ = run_stream(EffectChain([Gain(gain_db=-6.0)]), sine(440))
        ratio = np.abs(out).max() / np.abs(sine(440)).max()
        assert ratio == pytest.approx(10 ** (-6 / 20), abs=1e-3)

    def test_unity(self):
        signal = sine(440)
        out, _ = run_stream(EffectChain([Gain(gain_db=0.0)]), signal)
        np.testing.assert_allclose(out, signal, atol=1e-6)


class TestBiquadFilter:
    def test_lowpass_attenuates_high_freq(self):
        signal = sine(440) + sine(8000)
        out, _ = run_stream(EffectChain([BiquadFilter(SR, 1000, "lowpass")]), signal)
        assert dominant_freq(out) == pytest.approx(440, abs=5)

    def test_highpass_attenuates_low_freq(self):
        signal = sine(440) + sine(8000)
        out, _ = run_stream(EffectChain([BiquadFilter(SR, 3000, "highpass")]), signal)
        assert dominant_freq(out) == pytest.approx(8000, abs=5)

    def test_streaming_equals_batch(self):
        """Filter memory across blocks must make streaming output identical
        to processing the entire signal as one block."""
        signal = sine(440, duration=0.2)
        streamed, _ = run_stream(EffectChain([BiquadFilter(SR, 1000, "lowpass")]), signal)
        batch = BiquadFilter(SR, 1000, "lowpass").process(signal)
        np.testing.assert_allclose(streamed, batch[: len(streamed)], atol=1e-5)

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError):
            BiquadFilter(SR, 1000, "bandstop")


class TestDelay:
    def test_echo_appears_after_delay(self):
        impulse = np.zeros(SR // 2, dtype=np.float32)
        impulse[0] = 1.0
        out, _ = run_stream(EffectChain([Delay(SR, delay_ms=100, feedback=0.0, mix=0.5)]), impulse)
        delay_samples = int(SR * 0.1)
        assert out[0] == pytest.approx(0.5, abs=1e-4)  # dry portion
        assert out[delay_samples] == pytest.approx(0.5, abs=1e-4)  # echo

    def test_feedback_produces_repeats(self):
        impulse = np.zeros(SR, dtype=np.float32)
        impulse[0] = 1.0
        out, _ = run_stream(EffectChain([Delay(SR, delay_ms=100, feedback=0.5, mix=1.0)]), impulse)
        d = int(SR * 0.1)
        first, second = out[d], out[2 * d]
        assert first > 0 and second > 0
        assert second == pytest.approx(first * 0.5, rel=0.05)


class TestTremolo:
    def test_amplitude_modulated(self):
        signal = np.ones(SR // 2, dtype=np.float32)
        out, _ = run_stream(EffectChain([Tremolo(SR, rate_hz=4.0, depth=1.0)]), signal)
        assert out.max() > 0.95
        assert out.min() < 0.05

    def test_phase_continuous_across_blocks(self):
        """The LFO must not click at block boundaries: the max sample-to-
        sample jump in the envelope stays at the smooth theoretical bound."""
        signal = np.ones(SR // 2, dtype=np.float32)
        out, _ = run_stream(EffectChain([Tremolo(SR, rate_hz=4.0, depth=1.0)]), signal)
        max_jump = np.abs(np.diff(out)).max()
        theoretical = 2 * np.pi * 4.0 / SR  # LFO slope bound per sample
        assert max_jump < theoretical * 1.5


class TestLiveControl:
    def test_set_param_takes_effect_mid_stream(self):
        chain = EffectChain([Gain(gain_db=0.0)])
        proc = StreamProcessor(chain, block_size=BLOCK)
        signal = np.ones(BLOCK * 4, dtype=np.float32)
        blocks = list(iter_blocks(signal, BLOCK))

        out_first = proc.process_block(blocks[0])
        chain.effects[0].set_param("gain_db", -20.0)
        out_second = proc.process_block(blocks[1])

        assert out_first.max() == pytest.approx(1.0, abs=1e-6)
        assert out_second.max() == pytest.approx(0.1, abs=1e-3)

    def test_set_param_unknown_name_rejected(self):
        with pytest.raises(AttributeError):
            Gain().set_param("volume", 3)

    def test_concurrent_param_updates_do_not_corrupt(self):
        chain = EffectChain([BiquadFilter(SR, 1000, "lowpass")])
        proc = StreamProcessor(chain, block_size=BLOCK)
        signal = sine(440, duration=1.0)
        stop = threading.Event()

        def tweak():
            cutoff = 500
            while not stop.is_set():
                chain.effects[0].set_param("cutoff_hz", cutoff)
                cutoff = 500 + (cutoff + 137) % 4000

        t = threading.Thread(target=tweak)
        t.start()
        try:
            out = np.concatenate(list(proc.process_stream(iter_blocks(signal, BLOCK))))
        finally:
            stop.set()
            t.join()
        assert np.all(np.isfinite(out))

    def test_chain_add_remove(self):
        chain = EffectChain()
        g = Gain(gain_db=-6.0)
        chain.add(g)
        assert len(chain.effects) == 1
        chain.remove(g)
        out = chain.process(np.ones(8, dtype=np.float32))
        np.testing.assert_allclose(out, 1.0)


class TestPerformance:
    def test_block_latency_model(self):
        proc = StreamProcessor(EffectChain(), block_size=256)
        assert proc.latency_ms(44100) == pytest.approx(5.8, abs=0.1)

    def test_realtime_headroom(self):
        """A full effect chain must process a 256-sample block faster than
        the 5.8 ms of audio it represents, with generous CI margin."""
        chain = EffectChain([
            Gain(-3.0),
            BiquadFilter(SR, 2000, "lowpass"),
            Delay(SR, delay_ms=120),
            Tremolo(SR, rate_hz=5.0),
        ])
        signal = sine(440, duration=1.0)
        _, proc = run_stream(chain, signal)
        block_audio_ms = proc.latency_ms(SR)
        assert proc.stats["avg_block_time_ms"] < block_audio_ms
        # iter_blocks zero-pads the final partial block
        assert proc.stats["blocks"] == -(-len(signal) // BLOCK)

    def test_invalid_block_size(self):
        with pytest.raises(ValueError):
            StreamProcessor(EffectChain(), block_size=0)
