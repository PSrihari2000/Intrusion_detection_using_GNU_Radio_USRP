import numpy as np
from gnuradio import gr

class blk(gr.sync_block):
    """Intrusion Detector - 1M samp_rate, no resampler needed"""
    def __init__(self, threshold=0.3, std_threshold=0.05,
                 baseline_samples=100000, samp_rate=1000000, learning_rate=0.95):
        gr.sync_block.__init__(
            self,
            name='Fast Intrusion Detector',
            in_sig=[np.float32],
            out_sig=[np.float32]
        )
        self.threshold = float(threshold)
        self.std_threshold = float(std_threshold)
        self.baseline_samples = int(baseline_samples)
        self.samp_rate = float(samp_rate)
        self.learning_rate = float(learning_rate)
        self._buf = np.empty(self.baseline_samples, dtype=np.float32)
        self._buf_idx = 0
        self.baseline_power = 0.0
        self.baseline_std = 0.0
        self.armed = False
        self.smooth_ratio = 1.0
        self.phase = 0.0
        self.detect_count = 0

    def work(self, input_items, output_items):
        inp = input_items[0]
        out = output_items[0]
        n = len(inp)
        power = float(np.mean(inp))
        std   = float(np.std(inp))
        if not self.armed:
            space = self.baseline_samples - self._buf_idx
            chunk = min(n, space)
            self._buf[self._buf_idx:self._buf_idx + chunk] = inp[:chunk]
            self._buf_idx += chunk
            if self._buf_idx >= self.baseline_samples:
                self.baseline_power = float(np.mean(self._buf))
                self.baseline_std   = float(np.std(self._buf))
                self.smooth_ratio   = 1.0
                self._buf = None
                self.armed = True
            out[:] = 0.0
            return n
        ratio = power / (self.baseline_power + 1e-9)
        ratio = min(ratio, 10.0)
        self.smooth_ratio = (self.learning_rate * self.smooth_ratio +
                            (1.0 - self.learning_rate) * ratio)
        ratio_alert = self.smooth_ratio > 1.0 + self.threshold
        std_alert   = std > self.baseline_std * (1.0 + self.std_threshold)
        intrusion   = ratio_alert and std_alert
        power_gate  = power > self.baseline_power * 9.0
        if intrusion and power_gate:
            phase_step = 2.0 * np.pi * 1000.0 / self.samp_rate
            np.sin(self.phase + phase_step *
                   np.arange(n, dtype=np.float32), out=out)
            out *= 0.5
            self.phase = (self.phase + phase_step * n) % (2.0 * np.pi)
        else:
            out[:] = 0.0
            self.phase = 0.0
        return n