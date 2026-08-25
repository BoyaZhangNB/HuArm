"""synthesis.py -- real-time erhu synthesis driven by `ErhuEnv` bow state.

The sound generator is Faust's shipped bowed-string physical model from
`physmodels.lib` (grame-cncm/faustlibraries, part of the official Faust
distribution:
https://github.com/grame-cncm/faustlibraries/blob/master/physmodels.lib).
It is a real, shipped, open-source digital waveguide -- not a neural net --
whose native parameters are exactly bow pressure, bow velocity and bow
position (plus string length for pitch), which is why it can be driven
straight off the quantities `ErhuEnv` already measures. It is compiled at
runtime via `dawdreamer` (https://github.com/DBraun/DawDreamer), which embeds
the Faust compiler.

    pip install dawdreamer sounddevice

Erhu, not violin
----------------
`pm.violin_ui` is the library's ready-made instrument, but a violin is the
wrong instrument here, so this module keeps the physics and swaps the parts
that are not shared. The chain is assembled from the same `pm` primitives
`pm.violinModel` uses --

    erhuNuts : pm.violinBowedString(...) : erhuBridge : erhuBody : pm.out

-- with three erhu-specific substitutions:

  * `erhuBody` replaces `pm.violinBody` (which is literally one
    `fi.resonbp(500, 2, 1)`). An erhu's resonator is a small hexagonal barrel
    closed at the front by a stretched python-skin membrane and open at the
    back, so instead of a violin's broad wooden plate response it gets a
    dry direct path plus a bank of resonators covering the barrel's air
    resonance, the first membrane mode, the ~1.2 kHz nasal formant the
    instrument is recognised by, and the upper membrane cluster.
  * `erhuBridge` reflects less and damps more than `pm.violinBridge`: the
    erhu's tiny wooden bridge sits directly on the membrane, which radiates
    efficiently, so the string loses energy fast (t60 ~ 1.4 s here).
  * `erhuNuts` stands in for the *qianjin*, the cloth loop that binds the
    string to the neck. Cloth is a lossier, duller termination than a
    violin's nut, so it gets lower brightness and higher absorption.

On top of the waveguide sit three things a bare model does not give you and
a listener immediately misses: hair noise injected into the bow velocity
(irregular stick-slip, scaled by bow pressure), a slow random pitch flutter
of a fraction of a percent, and a small room. All three are parameters and
can be set to zero.

Pitch is fixed at A4 -- `string_len` is deliberately not wired to anything
(see `A4_HZ` / `freq_to_length`).

Driving it from `ErhuEnv`
-------------------------
`BowMapping` converts the env's physical measurements into the model's
normalised controls:

    state.metrics["bow_a_force_ema"]  N      -> bow pressure
    state.metrics["bow_vel_ema"]      m/s    -> bow velocity (magnitude)
    env._bow_stroke_position(...)     [-1,1] -> bow position

See `BowMapping` for why bow pressure and velocity are mapped into narrow
bands rather than the model's full 0-1 range, and why the bow *position*
range is narrow too.

Real-time rendering
-------------------
DawDreamer is an offline renderer: `RenderEngine.render()` resets all DSP
state and re-renders from t=0 every call, so the obvious "render one small
block per control step and concatenate" loop does not produce continuous
audio -- it produces the same block over and over, combed at the block rate.

`ErhuSynth` therefore streams by *anchored re-render*: it keeps the whole
control history as PPQN-rate automation, re-renders the entire timeline for
every audio chunk, and emits only the newly grown tail. Because the DSP is
deterministic and every render shares the same prefix, consecutive chunks
line up sample-exactly -- there is no crossfade and no discontinuity. The
cost is that render time grows with session length: end to end (automation
upload, render, buffer copy) the model runs a few hundred times faster than
real time, so on the machine this was written on a 50 ms chunk carries about
15 s of history before it stops keeping up, and `max_window_s` defaults to
well under that. `ErhuSynth.render_load` reports the measured margin --
render time over chunk time, so anything approaching 1.0 means the window is
too long. `ErhuSynth` re-anchors -- drops the history and starts a new
timeline -- as soon as the string falls silent, which on a bowing task
happens every time the bow leaves the string, and is inaudible because there
is nothing to cut. If
`max_window_s` of *continuously sounding* audio goes by without such a
moment, it re-anchors anyway: the new timeline is warmed up by replaying the
last `warmup_s` of control history so it reaches the same steady state, and
the two are equal-power crossfaded over one chunk.

Usage
-----
    from synthesis import ErhuSynth

    with ErhuSynth() as synth:          # opens the audio device
        while running:
            ...
            synth.update(force_n=..., speed_mps=..., bow_x=...)

or, straight off a Brax `State` (see `teleop.py`):

            synth.update_from_state(state, bow_x=...)

Run this file directly to hear a demo bow stroke, or to render it to a .wav
without touching an audio device:

    python synthesis.py --play
    python synthesis.py --wav erhu_demo.wav --analyse
"""

from __future__ import annotations

import argparse
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

SAMPLE_RATE = 44100
BLOCK_SIZE = 128          # dawdreamer's internal render block

# Automation is fed to Faust as PPQN data with the engine at 60 BPM, so one
# "beat" is one second and `CONTROL_RATE` pulses-per-quarter-note is exactly
# `CONTROL_RATE` control frames per second. Keeping the automation at control
# rate rather than audio rate makes the arrays ~440x smaller, which matters
# because the real-time path re-uploads the whole history every chunk.
CONTROL_RATE = 100
CONTROL_BPM = 60.0

A4_HZ = 440.0
SPEED_OF_SOUND = 340.0    # m/s, matches pm.speedOfSound

# Extra loop delay contributed by the terminations, the bow junction and the
# one-sample delay `pm.chain` inserts per link -- i.e. everything in the
# waveguide's feedback path that is not the two string segments. The model's
# sounding period is `l2s(length) + LOOP_EXTRA_SAMPLES` samples, so tuning it
# needs that offset subtracted; `pm.violinModel` does the same thing with its
# hard-coded `stringTuning = 0.08`. Measured (see `calibrate_loop_extra`),
# and re-measured at construction unless `calibrate=False`.
LOOP_EXTRA_SAMPLES = 12.0


def freq_to_length(freq_hz: float, loop_extra_samples: float = LOOP_EXTRA_SAMPLES,
                   sample_rate: int = SAMPLE_RATE) -> float:
    """String length (meters) that makes the model sound at `freq_hz`.

    The waveguide's round trip is `length * SR / SPEED_OF_SOUND` samples of
    string plus a fixed `loop_extra_samples` of filter/junction delay, so a
    naive `SPEED_OF_SOUND / freq` is sharp by that offset -- about 1.9
    semitones at A4, which is audible as badly out of tune rather than as a
    subtle detune.
    """
    loop_samples = sample_rate / float(freq_hz)
    return max(1e-3, (loop_samples - loop_extra_samples) * SPEED_OF_SOUND / sample_rate)


# ---------------------------------------------------------------------------
# The DSP. `%(...)s` slots are filled in by `build_dsp` so the erhu's
# termination coefficients and body modes stay editable from Python.
# ---------------------------------------------------------------------------

DSP_TEMPLATE = r'''
import("stdfaust.lib");
pm = library("physmodels.lib");
bf = pm.bridgeFilter;

// ---- controls -------------------------------------------------------------
// Labels here are the public parameter names; dawdreamer prefixes them with
// a group path, which `ErhuSynth` resolves by label rather than hard-coding.
length_  = hslider("length",   %(LENGTH)r, 0.02, 2.0, 0.000001) : si.smoo;
bowVel   = hslider("velocity", 0.0,  0.0,  1.0, 0.000001) : si.smoo;
bowPress = hslider("pressure", %(PRESSURE)r, 0.0, 1.0, 0.000001) : si.smoo;
bowPos   = hslider("position", %(POSITION)r, 0.05, 0.95, 0.000001) : si.smoo;
gain     = hslider("gain",     0.0,  0.0,  1.0, 0.000001) : si.smoo;
hair     = hslider("hair",     %(HAIR)r, 0.0, 1.0, 0.000001) : si.smoo;
flutter  = hslider("flutter",  %(FLUTTER)r, 0.0, 0.05, 0.000001) : si.smoo;
room     = hslider("room",     %(ROOM)r, 0.0, 1.0, 0.000001) : si.smoo;

// ---- erhu-specific parts of the chain ------------------------------------
// The qianjin: a cloth loop, so duller and lossier than a violin's nut
// (pm.violinNuts uses bf(0.6, 0.1)).
erhuNuts   = pm.lTermination(-bf(%(NUT_BRIGHT)r, %(NUT_ABSORB)r), pm.basicBlock);

// Bridge on a stretched membrane: radiates hard, so the string's t60 is
// short (absorption a -> t60 = (1-a)*20 s). pm.violinBridge uses bf(0.2, 0.9).
erhuBridge = pm.rTermination(pm.basicBlock, -bf(%(BRG_BRIGHT)r, %(BRG_ABSORB)r)) : _, _, _;

// Body. Only the middle (right-going / transmitted) signal reaches the
// output bus through pm.out, exactly as in pm.violinBody; the left-going
// wave is fed 0 by pm.endChain's closeIns, so `_` there is a pass-through,
// not an unfiltered leak.
erhuBody = _, transmittance, _
with {
    transmittance = _ <: (_ * %(BODY_DIRECT)r, modes) :> _
                      : fi.highpass(2, %(BODY_HP)r)
                      : fi.lowpass(2, %(BODY_LP)r);
    modes = _ <: (%(BODY_MODES)s) :> _ * %(BODY_MODE_GAIN)r;
};

erhuModel(stringLength, bowPressure, bowVelocity, bowPosition) =
    pm.endChain(pm.chain(
        erhuNuts :
        pm.violinBowedString(stringLength, bowPressure, bowVelocity, bowPosition) :
        erhuBridge :
        erhuBody :
        pm.out
    ));

// ---- the things a bare waveguide does not give you ------------------------
// Hair noise belongs on the bow velocity, not on the output: what is
// irregular in a real bow stroke is the stick-slip itself. More rosin grip
// (higher pressure) means more of it.
excitation = bowVel * (1.0 + hair * (0.4 + 0.6 * bowPress) * grit)
with {
    grit = no.noise : fi.lowpass(1, 5000);
};

// A fraction of a percent of slow random pitch drift. Without it a sustained
// waveguide tone is recognisably synthetic.
soundingLength = length_ * (1.0 + flutter * no.lfnoise(4.7));

dry = erhuModel(soundingLength, bowPress, excitation, bowPos) * gain * %(OUT_TRIM)r;

// A small room. The erhu is never heard anechoically and the model has no
// radiation of its own.
verb = re.mono_freeverb(0.62, 0.42, 0.55, 0);

process = dry <: (_ * (1.0 - room), verb * room) :> _ : co.limiter_1176_R4_mono;
'''


@dataclass
class ErhuTone:
    """Physical/timbral constants of the instrument itself, as opposed to the
    bowing gesture. Defaults were tuned by ear and by checking the harmonic
    envelope and spectral centroid of a sustained A4 (see `--analyse`)."""

    # Terminations. `bridgeFilter(brightness, absorption)`: brightness damps
    # highs (0 = dark), absorption sets the string's t60 = (1 - a) * 20 s.
    nut_brightness: float = 0.60      # qianjin (cloth loop)
    nut_absorption: float = 0.40
    bridge_brightness: float = 0.70   # small bridge on the python skin
    bridge_absorption: float = 0.93   # t60 ~ 1.4 s

    # Body: a dry direct path plus resonant modes, as (freq_hz, Q, gain).
    body_direct: float = 0.45
    body_mode_gain: float = 0.85
    body_modes: Sequence[tuple] = field(default_factory=lambda: (
        (320.0,  2.5, 0.55),   # barrel air resonance
        (520.0,  3.0, 0.70),   # first membrane mode
        (1200.0, 2.0, 1.00),   # the nasal formant the erhu is known by
        (2600.0, 2.5, 0.55),   # upper membrane cluster
        (3900.0, 3.0, 0.25),   # rosin / brightness
    ))
    body_highpass: float = 130.0
    body_lowpass: float = 7500.0

    # Extras (see the module docstring). Set any of them to 0.0 to disable.
    hair_noise: float = 0.08          # stick-slip irregularity
    pitch_flutter: float = 0.0015     # +/- 0.15 % slow random detune
    room: float = 0.12                # reverb wet fraction

    # Fixed output trim so `gain` can stay a musical 0-1 control: the raw
    # model peaks around 3.6 with the bow at full tilt.
    out_trim: float = 0.22


def build_dsp(tone: ErhuTone, init_length: float, init_pressure: float,
              init_position: float) -> str:
    """Fill `DSP_TEMPLATE` in from `tone`. Kept a plain function so the
    generated Faust source can be printed and pasted into a Faust IDE."""
    modes = ", ".join(
        f"fi.resonbp({f!r}, {q!r}, {g!r})" for f, q, g in tone.body_modes
    )
    return DSP_TEMPLATE % dict(
        LENGTH=float(init_length),
        PRESSURE=float(init_pressure),
        POSITION=float(init_position),
        HAIR=float(tone.hair_noise),
        FLUTTER=float(tone.pitch_flutter),
        ROOM=float(tone.room),
        NUT_BRIGHT=float(tone.nut_brightness),
        NUT_ABSORB=float(tone.nut_absorption),
        BRG_BRIGHT=float(tone.bridge_brightness),
        BRG_ABSORB=float(tone.bridge_absorption),
        BODY_DIRECT=float(tone.body_direct),
        BODY_MODES=modes,
        BODY_MODE_GAIN=float(tone.body_mode_gain),
        BODY_HP=float(tone.body_highpass),
        BODY_LP=float(tone.body_lowpass),
        OUT_TRIM=float(tone.out_trim),
    )


# ---------------------------------------------------------------------------
# Env state -> model controls.
# ---------------------------------------------------------------------------

@dataclass
class BowMapping:
    """Turns `ErhuEnv`'s measured bow state into the waveguide's controls.

    The defaults are keyed to `ErhuEnv`'s own scales: `force_ref` is its
    `traj_p_max` (3 N) and `speed_ref` is a little over its `traj_v_limit`
    (0.1 m/s), so a policy or operator tracking the scripted reference stroke
    sweeps most of the expressive range.

    Two of the ranges are deliberately narrow:

    `pressure_range` and `velocity_range` -- the STK-style bow table
    `pm.violinBowTable` uses sustains clean Helmholtz motion only over part of
    its nominal 0-1 span. Below about 0.5 bow pressure, or below about 0.25
    bow velocity, the waveguide period-triples and locks a fifth-plus-two-
    octaves *below* the string (146.7 Hz for an A4): not the airy "not enough
    bow" sound a real light stroke makes, just a wrong note. Both controls are
    therefore mapped into the stable band, and softness is expressed the way
    it actually sounds -- through amplitude (`quiet_gain`) and through extra
    hair noise when the bow is under-weighted for its speed (`_bow_tone`).
    Bow velocity is still hard-gated to zero when the hair is off the string
    or not moving, so a stroke starts with a real attack instead of fading up
    from an already-sounding string.

    `position_range` -- on an erhu the hair is threaded *between* the two
    strings and stays there, so unlike a violinist's the contact point on the
    string barely moves; what `bow_x` measures is travel along the *hair*
    (frog to tip), which changes hair tension and angle and so the tone only
    slightly. Mapping it across the model's full sul tasto / sul ponticello
    span would be a violinist's gesture, not an erhu player's.
    """

    force_ref: float = 3.0            # N mapping to full bow pressure
    force_floor: float = 0.05         # N below which the hair is off the string
    speed_ref: float = 0.12           # m/s mapping to full bow speed
    speed_floor: float = 0.004        # m/s below which nothing is sustained

    pressure_range: tuple = (0.55, 0.95)
    velocity_range: tuple = (0.25, 0.85)
    position_range: tuple = (0.78, 0.86)
    quiet_gain: float = 0.30          # output level at the slowest sounding bow

    # Schelleng's minimum bow force grows with bow speed: bow fast and light
    # and the string never latches into Helmholtz motion, it just hisses.
    # `min_force_per_speed` is that slope in N per (m/s).
    min_force_per_speed: float = 12.0
    surface_noise: float = 0.35       # extra hair noise when under-bowed

    master_gain: float = 0.9

    def __call__(self, force_n: float, speed_mps: float, bow_x: float = 0.0) -> dict:
        """Returns a dict of Faust parameter label -> value."""
        force = max(0.0, float(force_n))
        speed = abs(float(speed_mps))

        p_norm = _clip01(force / self.force_ref) ** 0.5
        pressure = _lerp(self.pressure_range, p_norm)

        s_norm = _clip01((speed - self.speed_floor)
                         / max(1e-9, self.speed_ref - self.speed_floor))
        position = _lerp(self.position_range, _clip01((float(bow_x) + 1.0) * 0.5))

        # Bowing at all is a yes/no thing: hair on the string and moving, or
        # no excitation and the string left to ring down on its own.
        bowing = force > self.force_floor and speed > self.speed_floor
        velocity = _lerp(self.velocity_range, s_norm ** 0.7) if bowing else 0.0

        tone = self._bow_tone(force, speed)
        loud = _lerp((self.quiet_gain, 1.0), s_norm ** 0.6)
        return {
            "pressure": pressure,
            "velocity": velocity,
            "position": position,
            "gain": self.master_gain * tone * loud * (1.0 if bowing else 0.0),
            "hair": self.surface_noise * (1.0 - tone),
        }

    def _bow_tone(self, force_n: float, speed_mps: float) -> float:
        """0 = under-bowed (surface sound only), 1 = properly gripping.

        Smooth so a stroke fades in and out rather than switching."""
        f_min = self.min_force_per_speed * speed_mps
        if f_min <= 1e-9:
            return 1.0
        return _smoothstep(_clip01(force_n / f_min))


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))


def _lerp(rng: tuple, t: float) -> float:
    lo, hi = rng
    return lo + (hi - lo) * t


def _smoothstep(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


# ---------------------------------------------------------------------------
# The synth.
# ---------------------------------------------------------------------------

class ErhuSynth:
    """Faust erhu waveguide, playable in real time from a control loop.

    Not started by construction -- call `start()` (or use it as a context
    manager) to open the audio device and begin streaming. `update()` is
    thread-safe and non-blocking; call it as often as you like, the renderer
    samples the latest values at `CONTROL_RATE`.
    """

    LABELS = ("length", "velocity", "pressure", "position", "gain", "hair",
              "flutter", "room")
    # Parameters the real-time path automates. `length` stays a static
    # parameter: pitch is fixed at A4 for now.
    AUTOMATED = ("velocity", "pressure", "position", "gain", "hair")

    def __init__(self, freq_hz: float = A4_HZ, tone: Optional[ErhuTone] = None,
                 mapping: Optional[BowMapping] = None,
                 sample_rate: int = SAMPLE_RATE, chunk_s: float = 0.05,
                 max_window_s: float = 6.0, warmup_s: float = 0.35,
                 silence_rms: float = 2e-4, device=None, calibrate: bool = True):
        import dawdreamer as daw  # imported here so `--help` works without it

        self.sr = int(sample_rate)
        self.tone = tone or ErhuTone()
        self.mapping = mapping or BowMapping()
        self.freq_hz = float(freq_hz)
        self.device = device

        self.chunk_frames = max(1, int(round(chunk_s * CONTROL_RATE)))
        self.chunk_s = self.chunk_frames / CONTROL_RATE
        self.chunk_samples = int(round(self.chunk_s * self.sr))
        self.max_window_s = float(max_window_s)
        self.warmup_frames = max(1, int(round(warmup_s * CONTROL_RATE)))
        self.silence_rms = float(silence_rms)

        self.loop_extra_samples = LOOP_EXTRA_SAMPLES
        self._engine = daw.RenderEngine(self.sr, BLOCK_SIZE)
        self._engine.set_bpm(CONTROL_BPM)
        self._fp = self._engine.make_faust_processor("erhu")

        length = freq_to_length(self.freq_hz, self.loop_extra_samples, self.sr)
        code = build_dsp(self.tone, length,
                         self.mapping.pressure_range[0],
                         sum(self.mapping.position_range) / 2.0)
        if not self._fp.set_dsp_string(code):
            raise RuntimeError("failed to set the erhu Faust DSP")
        if not self._fp.compile():
            raise RuntimeError("failed to compile the erhu Faust DSP")
        self._engine.load_graph([(self._fp, [])])

        # dawdreamer namespaces parameters under a group path; resolve by the
        # `hslider` label so the paths are not hard-coded.
        desc = self._fp.get_parameters_description()
        self._param = {d["label"]: d["name"] for d in desc}
        missing = [l for l in self.LABELS if l not in self._param]
        if missing:
            raise RuntimeError(f"erhu DSP is missing parameters: {missing}")

        if calibrate:
            self.loop_extra_samples = self.calibrate_loop_extra()
            length = freq_to_length(self.freq_hz, self.loop_extra_samples, self.sr)
        self._length = length
        self._fp.set_parameter(self._param["length"], self._length)

        # Control state, shared with the render thread.
        self._lock = threading.Lock()
        self._target = self.mapping(0.0, 0.0, 0.0)
        self._prev = dict(self._target)

        self._auto = {k: np.zeros(0, dtype=np.float32) for k in self.AUTOMATED}
        self._emitted = 0          # samples of the current anchor already played
        self._max_frames = int(self.max_window_s * CONTROL_RATE)
        self._thread = None
        self._stop_evt = threading.Event()
        self._stream = None
        self.underruns = 0        # device underflows reported by sounddevice
        self.render_s = 0.0       # wall time of the most recent chunk render
        self.render_load = 0.0    # render_s / chunk_s; >1 means we cannot keep up

    # -- tuning -------------------------------------------------------------

    def calibrate_loop_extra(self) -> float:
        """Measure the waveguide's non-string loop delay, in samples.

        Renders two sustained tones at known string lengths and reads their
        periods back; the delay that is *not* proportional to length is the
        offset `freq_to_length` has to subtract. Costs a few milliseconds and
        keeps A4 in tune even if the terminations above are edited (each
        `bridgeFilter` and each `pm.chain` link contributes delay).
        """
        extras = []
        for length in (0.5, 0.9):
            self._set_static(length=length, velocity=0.5, pressure=0.7,
                             position=0.82, gain=0.6, hair=0.0, flutter=0.0,
                             room=0.0)
            self._engine.render(2.0)
            x = self._engine.get_audio()[0]
            loop = self.sr / _fundamental(x[int(1.0 * self.sr):int(2.0 * self.sr)], self.sr)
            extras.append(loop - length * self.sr / SPEED_OF_SOUND)
        # Restore the parameters the render above trampled.
        self._set_static(velocity=0.0, gain=0.0, hair=self.tone.hair_noise,
                         flutter=self.tone.pitch_flutter, room=self.tone.room)
        return float(np.mean(extras))

    def _set_static(self, **kw):
        for label, value in kw.items():
            self._fp.set_parameter(self._param[label], float(value))

    # -- control ------------------------------------------------------------

    def update(self, force_n: float, speed_mps: float, bow_x: float = 0.0) -> dict:
        """Set the current bow gesture, in the env's physical units.

        `force_n`  -- bow-hair/string contact force, newtons.
        `speed_mps`-- lateral bow speed, m/s (sign ignored: an up-bow and a
                      down-bow of the same speed sound the same).
        `bow_x`    -- normalised position along the hair, -1 (frog) to +1
                      (tip).

        Returns the model controls it mapped to, which is handy for logging.
        """
        controls = self.mapping(force_n, speed_mps, bow_x)
        with self._lock:
            self._target = controls
        return controls

    def update_from_state(self, state, bow_x: float = 0.0) -> dict:
        """`update()` straight off a Brax `State` from `ErhuEnv.step`.

        Uses the low-pass filtered metrics (`bow_a_force_ema`,
        `bow_vel_ema`) rather than the raw per-step values: the raw contact
        force and the finite-differenced velocity are noisy enough at 25 Hz
        that feeding them in unfiltered makes the tone flutter in a way the
        arm is not actually doing. `teleop.py` logs the same two.
        """
        m = state.metrics
        return self.update(float(m["bow_a_force_ema"]), float(m["bow_vel_ema"]), bow_x)

    # -- real-time streaming ------------------------------------------------

    def start(self) -> "ErhuSynth":
        import sounddevice as sd

        if self._thread is not None:
            return self
        self._stream = sd.OutputStream(
            samplerate=self.sr, channels=1, dtype="float32",
            blocksize=self.chunk_samples, device=self.device, latency="low",
        )
        self._stream.start()
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run, name="erhu-render",
                                        daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    def _run(self):
        while not self._stop_evt.is_set():
            try:
                self._run_once()
            except Exception:
                if self._stop_evt.is_set():
                    break
                raise

    def _run_once(self):
        """Render and emit exactly one chunk. Split out of `_run` so the
        streaming path can be exercised without an audio device."""
        t0 = time.perf_counter()
        for label, col in self._next_frames().items():
            self._auto[label] = np.concatenate([self._auto[label], col])

        audio = self._render_current()
        # dawdreamer rounds a render up to whole internal blocks, so the
        # buffer is occasionally a sample shorter than `chunk_samples` times
        # the chunk count. Emit whatever is actually new rather than assuming
        # a fixed stride, or that sample is silently dropped and the stream
        # picks up a click.
        out = audio[self._emitted:]
        self._emitted = audio.size

        n_frames = len(self._auto["gain"])
        quiet = (out.size == 0
                 or float(np.sqrt(np.mean(out.astype(np.float64) ** 2))) < self.silence_rms)
        if quiet and n_frames > self.warmup_frames:
            # Free re-anchor: nothing is sounding, so nothing is cut.
            self._reanchor(keep_frames=0)
        elif n_frames >= self._max_frames:
            out = _crossfade(out, self._reanchor(self.warmup_frames, out.size))

        # Re-rendering the whole anchored timeline is what costs time here, so
        # this is the number that says how close to the wire the current
        # window length is running; `_stream.write` below blocks until the
        # device has room and is expected to take ~`chunk_s`.
        self.render_s = time.perf_counter() - t0
        self.render_load = self.render_s / self.chunk_s
        if self._stream.write(np.ascontiguousarray(out.reshape(-1, 1))):
            self.underruns += 1

    def _next_frames(self) -> dict:
        """One chunk's worth of automation, ramped from the previous chunk's
        values to the current target so a 25 Hz control loop does not step the
        bow in audible jumps (`si.smoo` in the DSP smooths what is left)."""
        with self._lock:
            target = dict(self._target)
        prev, self._prev = self._prev, target
        n = self.chunk_frames
        ramp = np.linspace(1.0 / n, 1.0, n, dtype=np.float32)
        return {k: (prev.get(k, target[k]) + (target[k] - prev.get(k, target[k])) * ramp)
                   .astype(np.float32)
                for k in self.AUTOMATED}

    def _render_current(self) -> np.ndarray:
        n_frames = len(self._auto["gain"])
        for label in self.AUTOMATED:
            self._fp.set_automation(self._param[label], self._auto[label],
                                    ppqn=CONTROL_RATE)
        self._engine.render(n_frames / CONTROL_RATE)
        return self._engine.get_audio()[0]

    def _reanchor(self, keep_frames: int, n_out: int = 0) -> np.ndarray:
        """Start a fresh timeline, optionally warmed up by replaying the last
        `keep_frames` of control history.

        Returns the new timeline's last `n_out` samples, which cover the same
        wall-clock instant as the chunk the old timeline just produced and so
        are what that chunk crossfades into. Empty when `keep_frames` is 0,
        i.e. when we re-anchored into silence and there is nothing to fade.
        """
        if keep_frames <= 0:
            self._auto = {k: np.zeros(0, dtype=np.float32) for k in self.AUTOMATED}
            self._emitted = 0
            return np.zeros(0, dtype=np.float32)
        self._auto = {k: v[-keep_frames:].copy() for k, v in self._auto.items()}
        audio = self._render_current()
        self._emitted = audio.size
        return audio[audio.size - n_out:] if n_out else np.zeros(0, dtype=np.float32)

    # -- offline ------------------------------------------------------------

    def render(self, duration_s: float, force_fn: Callable[[float], float],
               speed_fn: Callable[[float], float],
               bow_x_fn: Optional[Callable[[float], float]] = None) -> np.ndarray:
        """Render `duration_s` seconds offline, in one shot.

        The callables take seconds and return the same physical units
        `update()` takes, and go through the same `BowMapping`, so an offline
        render and the live stream sound identical for identical gestures.
        """
        n = int(round(duration_s * CONTROL_RATE))
        t = np.arange(n) / CONTROL_RATE
        auto = {k: np.zeros(n, dtype=np.float32) for k in self.AUTOMATED}
        for i, ti in enumerate(t):
            c = self.mapping(force_fn(ti), speed_fn(ti),
                             bow_x_fn(ti) if bow_x_fn else 0.0)
            for k in self.AUTOMATED:
                auto[k][i] = c[k]
        self._set_static(length=self._length, flutter=self.tone.pitch_flutter,
                         room=self.tone.room)
        for label in self.AUTOMATED:
            self._fp.set_automation(self._param[label], auto[label], ppqn=CONTROL_RATE)
        self._engine.render(duration_s)
        return self._engine.get_audio()[0][:int(duration_s * self.sr)]

    def save_wav(self, path, audio: np.ndarray, normalize: bool = False):
        from scipy.io import wavfile
        x = np.asarray(audio, dtype=np.float64)
        if normalize:
            x = x / (np.max(np.abs(x)) + 1e-9) * 0.9
        x = np.clip(x, -1.0, 1.0)
        wavfile.write(str(path), self.sr, (x * 32767).astype(np.int16))


def _crossfade(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Equal-power crossfade from `a` into `b` over their common length."""
    n = min(a.size, b.size)
    if n == 0:
        return a
    ramp = np.linspace(0.0, math.pi / 2, n, dtype=np.float32)
    return a[:n] * np.cos(ramp) + b[:n] * np.sin(ramp)


def _fundamental(x: np.ndarray, sr: int, lo: float = 120.0, hi: float = 1500.0) -> float:
    """Autocorrelation pitch estimate with parabolic interpolation."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    ac = np.correlate(x, x, "full")[x.size - 1:]
    n1, n2 = int(sr / hi), min(int(sr / lo), ac.size - 2)
    k = n1 + int(np.argmax(ac[n1:n2]))
    a, b, c = ac[k - 1], ac[k], ac[k + 1]
    denom = a - 2 * b + c
    k = k + (0.5 * (a - c) / denom if abs(denom) > 1e-12 else 0.0)
    return sr / k


# ---------------------------------------------------------------------------
# Demo / self-check.
# ---------------------------------------------------------------------------

def _demo_gestures():
    """A phrase that exercises the whole mapping: a swell, a lift, a light
    under-bowed stroke, and a heavy one."""
    def force_fn(t):
        if t < 0.15:  return 0.0
        if t < 2.2:   return 2.2 * min((t - 0.15) / 0.4, 1.0)
        if t < 2.9:   return 0.0                       # bow off the string
        if t < 4.2:   return 0.25                      # under-bowed: surface sound
        if t < 6.0:   return 2.8
        return 0.0

    def speed_fn(t):
        if t < 0.15 or (2.2 <= t < 2.9) or t >= 6.0:
            return 0.0
        return 0.07 + 0.02 * math.sin(2 * math.pi * 0.8 * t)

    def bow_x_fn(t):
        return math.sin(2 * math.pi * 0.25 * t)        # slow travel along the hair

    return force_fn, speed_fn, bow_x_fn


def _analyse(x: np.ndarray, sr: int, f0: float):
    seg = x[int(1.0 * sr):int(2.0 * sr)]
    w = np.hanning(seg.size)
    power = np.abs(np.fft.rfft(seg * w)) ** 2
    f = np.fft.rfftfreq(seg.size, 1 / sr)
    centroid = float((f * power).sum() / power.sum())
    bands = []
    for k in range(1, 15):
        sel = (f > (k - 0.45) * f0) & (f < (k + 0.45) * f0)
        bands.append(power[sel].sum())
    bands = np.array(bands)
    print(f"  measured f0 : {_fundamental(seg, sr):.2f} Hz  (target {f0:.2f})")
    print(f"  centroid    : {centroid:.0f} Hz")
    print(f"  peak / rms  : {np.abs(seg).max():.3f} / {np.sqrt((seg**2).mean()):.3f}")
    print("  partials dB :", np.round(10 * np.log10(bands / bands.max() + 1e-14), 1))


def main():
    ap = argparse.ArgumentParser(description="erhu physical-model synthesis")
    ap.add_argument("--wav", default="erhu_demo.wav", help="output .wav path")
    ap.add_argument("--seconds", type=float, default=7.0)
    ap.add_argument("--freq", type=float, default=A4_HZ)
    ap.add_argument("--analyse", action="store_true",
                    help="also print a spectral report of a sustained tone")
    ap.add_argument("--play", action="store_true",
                    help="stream the demo phrase live instead of writing a .wav")
    ap.add_argument("--print-dsp", action="store_true", help="dump the Faust source")
    args = ap.parse_args()

    if args.print_dsp:
        print(build_dsp(ErhuTone(), freq_to_length(args.freq), 0.7, 0.82))
        return

    synth = ErhuSynth(freq_hz=args.freq)
    print(f"loop delay offset: {synth.loop_extra_samples:.2f} samples "
          f"-> A4 string length {synth._length:.5f} m")

    force_fn, speed_fn, bow_x_fn = _demo_gestures()

    if args.play:
        with synth:
            t0 = time.time()
            while time.time() - t0 < args.seconds:
                t = time.time() - t0
                synth.update(force_fn(t), speed_fn(t), bow_x_fn(t))
                time.sleep(0.02)
            synth.update(0.0, 0.0, 0.0)
            time.sleep(1.5)
        print(f"underruns: {synth.underruns}")
        return

    audio = synth.render(args.seconds, force_fn, speed_fn, bow_x_fn)
    synth.save_wav(args.wav, audio)
    print(f"wrote {args.wav}  ({audio.size / synth.sr:.2f} s, "
          f"peak {np.abs(audio).max():.3f})")

    if args.analyse:
        sustained = synth.render(3.0, lambda t: 2.2, lambda t: 0.07)
        _analyse(sustained, synth.sr, args.freq)


if __name__ == "__main__":
    main()
