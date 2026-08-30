# Real-Time Intrusion Detection Using GNU Radio and USRP

Design and Simulation Laboratory project (EC6P002), School of Electrical and Computer
Sciences, Indian Institute of Technology Bhubaneswar — Academic Year 2025–26.

A pair of USRP software-defined radios are used to build a simple bistatic intrusion
detector: one USRP continuously transmits a sinusoidal carrier, the other receives it and
flags an intrusion when both the received signal power and its statistical variation
deviate significantly from a learned baseline.

**Author:** P Srihari

## How it works

1. **Transmitter** (`Tx.grc`) emits a continuous sine wave carrier via a USRP Sink.
2. **Receiver** (`Rx.grc`) captures the signal via a USRP Source, computes received
   signal power (magnitude-squared) and its standard deviation, and feeds them into a
   custom embedded Python block.
3. **Detector block** (`epy_block_1_0_dyhwg5t7.py`), embedded in the receiver flowgraph:
   - **Learning phase:** collects a baseline of samples (default 100,000) and computes
     the baseline mean power and standard deviation of the static environment.
   - **Detection phase:** tracks a smoothed power ratio (current power vs. baseline,
     exponentially smoothed) and compares live standard deviation against the baseline.
     An intrusion is flagged only when *both* the smoothed power ratio and the standard
     deviation exceed their respective thresholds, with an additional power gate — this
     dual-condition check reduces false detections from ordinary channel noise.
   - When an intrusion is detected, the block outputs an audible alert tone; otherwise
     it outputs silence.

Movement or obstruction in the propagation path perturbs the received signal through
reflection, scattering, and attenuation, which shows up as a deviation from the learned
static-environment baseline.

## Files

| File | Description |
|---|---|
| `Tx.grc` | GNU Radio Companion flowgraph for the transmitter (continuous sine wave carrier, USRP Sink) |
| `Rx.grc` | GNU Radio Companion flowgraph for the receiver (USRP Source, AGC, power/std computation, embedded detector block, spectrum/time-domain GUI sinks) |
| `epy_block_1_0_dyhwg5t7.py` | Embedded Python block implementing the baseline-learning and intrusion-detection logic |

## Requirements

- [GNU Radio](https://www.gnuradio.org/) (with GUI/QT support)
- [UHD](https://files.ettus.com/manual/) (USRP Hardware Driver)
- Two Ettus Research USRP devices (one as transmitter, one as receiver)

## Running

1. Open `Tx.grc` in GNU Radio Companion, connect the transmitting USRP, and run it to
   start emitting the carrier.
2. Open `Rx.grc` in GNU Radio Companion, connect the receiving USRP, and run it. The
   flowgraph first runs a learning phase to establish the baseline, then continuously
   monitors for intrusions.
3. Observe the time-domain and frequency-domain GUI sinks; an intrusion event produces
   an audible tone and a visible spike relative to the learned baseline.

## Results summary

Measurements were taken at transmitter–receiver separations of 85 cm, 1 m, and 1.5 m:

| Distance | Peak Power | Noise Floor |
|---|---|---|
| 85 cm | −52 dB | −125 dB |
| 1.00 m | −62 dB | −125 dB |
| 1.50 m | −74 dB | −126 dB |

As distance increases, received signal peak power decreases (path loss) while the noise
floor stays roughly constant, reducing SNR at larger separations. The detector reliably
flagged intrusions at all three distances tested.

## References

1. [GNU Radio Documentation](https://www.gnuradio.org/doc/)
2. [Ettus Research, USRP Hardware Driver and Manual](https://files.ettus.com/manual/)
3. M. Skolnik, *Introduction to Radar Systems*, 3rd Edition, McGraw-Hill, 2001.
4. Simon Haykin, *Communication Systems*, 5th Edition, Wiley, 2009.
5. T. S. Rappaport, *Wireless Communications: Principles and Practice*, 2nd Edition, Prentice Hall, 2002.
