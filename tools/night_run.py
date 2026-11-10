"""FYR_SPEEDZ 'Night Run' - original loop, synthesized from scratch (no samples).
122 BPM, A minor, Am-F-C-G. 32 bars, seamless loop.
Usage: python3 night_run.py out.wav
"""
import sys
import numpy as np
from scipy.signal import butter, sosfilt, fftconvolve

SR = 44100
BPM = 122
BEAT = 60 / BPM
BAR = 4 * BEAT
BARS = 32
TAIL_BARS = 2
N = int(SR * BAR * (BARS + TAIL_BARS))
LOOP_N = int(SR * BAR * BARS)
rng = np.random.default_rng(71)

def t_of(n): return np.arange(n) / SR
def mtof(m): return 440.0 * 2 ** ((m - 69) / 12)
def lp(x, fc, order=2): return sosfilt(butter(order, fc, 'low', fs=SR, output='sos'), x)
def hp(x, fc, order=2): return sosfilt(butter(order, fc, 'high', fs=SR, output='sos'), x)
def bp(x, lo, hi): return sosfilt(butter(2, [lo, hi], 'band', fs=SR, output='sos'), x)

def saw(f, n, phase=0.0):
    ph = (phase + f * t_of(n)) % 1.0
    return 2 * ph - 1

def add(buf, sig, start):
    s = int(start * SR)
    if s >= len(buf): return
    e = min(len(buf), s + len(sig))
    buf[s:e] += sig[:e - s]

# A minor: Am F C G  (root midi in octave 2/3)
CHORDS = [
    (45, [57, 60, 64, 69]),  # Am
    (41, [53, 57, 60, 65]),  # F
    (48, [55, 60, 64, 67]),  # C
    (43, [55, 59, 62, 67]),  # G
]
def chord_at(bar): return CHORDS[(bar // 2) % 4]  # 2 bars per chord

# ---------- section energy (0..1 per bar) ----------
def energy(bar):
    if bar < 8: return 0.55       # filtered intro
    if bar < 24: return 1.0
    return 1.0

kick = np.zeros(N); clap = np.zeros(N); hats = np.zeros(N)
bass = np.zeros(N); padL = np.zeros(N); padR = np.zeros(N)
arp = np.zeros(N); lead = np.zeros(N)

# ---------- drums ----------
def kick_hit():
    n = int(0.45 * SR); t = t_of(n)
    f = 45 + 105 * np.exp(-t * 28)
    ph = 2 * np.pi * np.cumsum(f) / SR
    body = np.sin(ph) * np.exp(-t * 7.5)
    click = lp(rng.standard_normal(n), 4000) * np.exp(-t * 400) * 0.3
    return np.tanh((body + click) * 1.6)

def clap_hit():
    n = int(0.3 * SR); t = t_of(n)
    noise = bp(rng.standard_normal(n), 900, 5000)
    env = np.exp(-t * 22)
    for d in (0.0, 0.011, 0.022):  # flam
        k = int(d * SR); env[k:] += np.exp(-t[:n - k] * 90) * 0.6
    return noise * env * 0.5

def hat_hit(open_=False):
    n = int((0.22 if open_ else 0.05) * SR); t = t_of(n)
    noise = hp(rng.standard_normal(n), 7500, 4)
    return noise * np.exp(-t * (14 if open_ else 80)) * (0.35 if open_ else 0.18)

K, C = kick_hit(), clap_hit()
for bar in range(BARS + TAIL_BARS):
    b0 = bar * BAR
    e = energy(bar % BARS) if bar < BARS else energy(bar - BARS)
    for beat in range(4):
        add(kick, K, b0 + beat * BEAT)
        if beat in (1, 3) and e > 0.9: add(clap, C, b0 + beat * BEAT)
        add(hats, hat_hit(True) * (0.8 if e > 0.9 else 0.5), b0 + beat * BEAT + BEAT / 2)
        if e > 0.9:
            for s in (1, 3):
                add(hats, hat_hit(False) * (0.6 + 0.3 * rng.random()), b0 + beat * BEAT + s * BEAT / 4)

# sidechain envelope from kick timing
duck = np.ones(N)
dn = int(0.32 * SR); dt = t_of(dn)
dshape = 1 - 0.75 * np.exp(-dt * 11)
for bar in range(BARS + TAIL_BARS):
    for beat in range(4):
        s = int((bar * BAR + beat * BEAT) * SR)
        e = min(N, s + dn); duck[s:e] = np.minimum(duck[s:e], dshape[:e - s])

# ---------- bass: offbeat 8ths + rolling 16th pickup ----------
def bass_note(midi, dur):
    n = int(dur * SR); t = t_of(n)
    f = mtof(midi)
    x = saw(f, n) * 0.6 + saw(f * 1.004, n) * 0.4 + np.sin(2 * np.pi * f / 2 * t) * 0.6
    cut = 180 + 900 * np.exp(-t * 18)
    y = np.zeros(n)
    # block-wise envelope filter
    blk = 256
    for i in range(0, n, blk):
        y[i:i + blk] = lp(x[i:i + blk + 512], cut[i])[:len(y[i:i + blk])]
    env = np.minimum(1, t * 400) * np.exp(-t * 4)
    return np.tanh(y * env * 1.4)

for bar in range(BARS + TAIL_BARS):
    root = chord_at(bar % BARS)[0]
    b0 = bar * BAR
    for beat in range(4):
        add(bass, bass_note(root, BEAT * 0.45), b0 + beat * BEAT + BEAT / 2)
    add(bass, bass_note(root + 12, BEAT * 0.2), b0 + 3 * BEAT + 3 * BEAT / 4)

# ---------- pads: detuned saw chords, stereo ----------
for bar in range(0, BARS + TAIL_BARS, 2):
    notes = chord_at(bar % BARS)[1]
    dur = 2 * BAR; n = int(dur * SR); t = t_of(n)
    L = np.zeros(n); R = np.zeros(n)
    for m in notes:
        f = mtof(m)
        for k, det in enumerate((-0.12, -0.05, 0.05, 0.12)):
            s = saw(f * 2 ** (det / 12), n, phase=rng.random())
            (L if k % 2 == 0 else R)[:] += s
            (R if k % 2 == 0 else L)[:] += s * 0.4
    env = np.minimum(1, t / 0.25) * np.minimum(1, (dur - t) / 0.3 + 0.0).clip(0, 1)
    add(padL, L * env * 0.035, bar * BAR)
    add(padR, R * env * 0.035, bar * BAR)
padL, padR = lp(padL, 2200), lp(padR, 2200)

# ---------- arp: 16th plucks over chord tones ----------
def pluck(midi, dur=0.18):
    n = int(dur * SR); t = t_of(n)
    f = mtof(midi)
    x = saw(f, n) * 0.5 + np.sign(np.sin(2 * np.pi * f * 2 * t)) * 0.2
    return lp(x, 3500) * np.exp(-t * 22)

ARP_ORDER = [0, 2, 1, 3, 2, 1, 3, 2]
step = 0
for bar in range(BARS + TAIL_BARS):
    tones = chord_at(bar % BARS)[1]
    for s in range(16):
        m = tones[ARP_ORDER[step % 8]] + 12
        acc = 1.0 if s % 4 == 0 else 0.7
        add(arp, pluck(m) * 0.22 * acc, bar * BAR + s * BEAT / 4)
        step += 1

# ---------- lead hook in last 8 bars (and its tail) ----------
HOOK = [  # (beat offset within 2 bars, scale step semitone above A4, length beats)
    (0, 7, 1.5), (1.5, 5, 0.5), (2, 3, 1), (3, 0, 1), (4, 3, 1.5), (5.5, 5, 0.5), (6, 7, 2),
]
def lead_note(midi, dur):
    n = int(dur * SR); t = t_of(n)
    f = mtof(midi) * (1 + 0.004 * np.sin(2 * np.pi * 5.2 * t) * np.minimum(1, t * 3))
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = np.sin(ph) + 0.35 * np.sin(2 * ph) + 0.15 * np.sign(np.sin(ph))
    env = np.minimum(1, t / 0.02) * np.exp(-t * 1.2) * np.minimum(1, (dur - t) / 0.05).clip(0, 1)
    return lp(x * env, 4200)

for bar in list(range(24, BARS, 2)):
    for off, semi, ln in HOOK:
        add(lead, lead_note(69 + semi, ln * BEAT) * 0.13, bar * BAR + off * BEAT)

# ---------- intro filter sweep on pads/arp (bars 0-8 dark, opening up) ----------
def sweep(x):
    dark = lp(x, 600)
    mix = np.ones(N)
    i8 = int(8 * BAR * SR)
    mix[:i8] = np.linspace(0.15, 1.0, i8) ** 2
    return dark * (1 - mix) + x * mix
arp, padL, padR = sweep(arp), sweep(padL), sweep(padR)

# ---------- delay + reverb ----------
def delay(x, time, fb=0.35, mix=0.3):
    d = int(time * SR); y = x.copy()
    for k in range(1, 5):
        sh = np.zeros_like(x); sh[d * k:] = x[:-d * k] if d * k < len(x) else 0
        y += sh * (fb ** k) * mix / fb
    return y

def ir(sec=2.4):
    n = int(sec * SR); t = t_of(n)
    return lp(rng.standard_normal(n), 6000) * np.exp(-t * 3.0) * 0.012

arpL = delay(arp, BEAT * 0.75, mix=0.35)
arpR = delay(arp, BEAT * 0.5, mix=0.3)
leadD = delay(lead, BEAT * 0.75, mix=0.4)

IR_L, IR_R = ir(), ir()
send = arp * 0.6 + lead * 0.8 + clap * 0.5 + (padL + padR) * 0.3
revL = fftconvolve(send, IR_L)[:N]
revR = fftconvolve(send, IR_R)[:N]

# ---------- mix ----------
L = (kick * 0.75 + clap * 0.55 + hats * 0.8 + bass * 0.55 * duck
     + padL * duck * 1.35 + arpL * duck * 1.2 + leadD * 0.9 + revL)
R = (kick * 0.75 + clap * 0.55 + hats * 0.7 + bass * 0.55 * duck
     + padR * duck * 1.35 + arpR * duck * 1.2 + leadD * 0.9 + revR)
# slight hat pan
L += hats * 0.1; R -= hats * 0.1

# ---------- seamless loop: fold tail onto start ----------
def fold(x):
    y = x[:LOOP_N].copy()
    tail = x[LOOP_N:]
    # only the decaying part of the tail (drums restart anyway) -> fold reverb/delay residue
    y[:len(tail)] += tail * 0.0  # drums/bass restart at bar 0; keep clean to avoid doubling
    return y
# Fold just the effect returns so reverb/delay don't cut off at the loop point
fxL = revL + leadD * 0.9 + (arpL - arp) * 0.9
fxR = revR + leadD * 0.9 + (arpR - arp) * 0.9
L, R = L[:LOOP_N].copy(), R[:LOOP_N].copy()
tl = len(fxL) - LOOP_N
L[:tl] += fxL[LOOP_N:]; R[:tl] += fxR[LOOP_N:]

# ---------- master: HP, glue, limit ----------
L, R = hp(L, 28), hp(R, 28)
peak = max(np.abs(L).max(), np.abs(R).max())
L, R = L / peak * 1.25, R / peak * 1.25
L, R = np.tanh(L) * 0.92, np.tanh(R) * 0.92

out = np.stack([L, R], 1)
from scipy.io import wavfile
wavfile.write(sys.argv[1], SR, (out * 32767).astype(np.int16))
print('wrote', sys.argv[1], f'{LOOP_N / SR:.2f}s')
