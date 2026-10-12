"""Synthesize the soundtrack: 72 BPM, 19 bars of 4/4 (63.33 s), D major.

Scene cuts in the video land on bar lines, so the music and the picture breathe together.
"""
import numpy as np, wave, sys

SR = 48000
BPM = 72
BEAT = 60 / BPM
BAR = 4 * BEAT
NBARS = 19
TOTAL = NBARS * BAR + 4.0  # reverb tail
N = int(TOTAL * SR)
rng = np.random.default_rng(7)

L = np.zeros(N); R = np.zeros(N)

NOTE = {'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4, 'F': 5, 'F#': 6, 'Gb': 6,
        'G': 7, 'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11}

def hz(name):
    n, o = name[:-1], int(name[-1])
    midi = 12 * (o + 1) + NOTE[n]
    return 440 * 2 ** ((midi - 69) / 12)

def add(sig, t0, pan=0.0, gain=1.0):
    i = max(0, int(t0 * SR))
    if i >= N: return
    sig = sig[:N - i] * gain
    l = np.cos((pan + 1) * np.pi / 4); r = np.sin((pan + 1) * np.pi / 4)
    L[i:i + len(sig)] += sig * l
    R[i:i + len(sig)] += sig * r

def piano(f, dur, vel=0.6):
    """Additive piano-like tone: inharmonic partials, per-partial decay, soft hammer."""
    length = min(dur + 2.5, 7.0)
    t = np.arange(int(length * SR)) / SR
    out = np.zeros_like(t)
    B = 0.00035
    bright = 0.6 + 0.6 * vel
    for n in range(1, 13):
        fn = f * n * np.sqrt(1 + B * n * n)
        if fn > 16000: break
        amp = (1 / n ** 1.45) * (bright ** (n - 1)) * (1.0 if n < 4 else 0.7)
        t60 = 5.5 * (220 / f) ** 0.35 / (1 + 0.45 * (n - 1))
        env = np.exp(-6.9 * t / t60)
        # two slightly detuned strings -> gentle beating
        out += amp * env * (np.sin(2 * np.pi * fn * t + rng.uniform(0, 6.28)) +
                            0.6 * np.sin(2 * np.pi * fn * 1.0007 * t + rng.uniform(0, 6.28)))
    att = np.minimum(1, t / 0.004)
    # damper release after note duration
    rel = np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.35))
    hammer = rng.normal(0, 1, len(t)) * np.exp(-t / 0.006) * 0.04
    hammer = np.convolve(hammer, np.ones(12) / 12, 'same')
    return (out * att * rel + hammer) * vel * 0.18

def pad(freqs, dur, gain=0.05):
    """Warm string-ish pad: detuned soft sawtooth (few harmonics), slow swell."""
    t = np.arange(int((dur + 1.5) * SR)) / SR
    out = np.zeros_like(t)
    for f in freqs:
        for det in (-0.004, 0.0, 0.0045):
            ff = f * (1 + det)
            ph = rng.uniform(0, 6.28)
            for n in range(1, 7):
                out += np.sin(2 * np.pi * ff * n * t + ph * n) / (n ** 1.7)
    swell = np.minimum(1, t / 1.6) ** 2
    rel = np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.6))
    lfo = 1 + 0.06 * np.sin(2 * np.pi * 0.23 * t)
    return out * swell * rel * lfo * gain / max(1, len(freqs))

def sub(f, dur, gain=0.055):
    t = np.arange(int((dur + 0.8) * SR)) / SR
    env = np.minimum(1, t / 0.05) * np.where(t < dur, np.exp(-t / 6), np.exp(-dur / 6) * np.exp(-(t - dur) / 0.3))
    return (np.sin(2 * np.pi * f * t) + 0.25 * np.sin(4 * np.pi * f * t)) * env * gain

# --- harmony, one chord per bar -------------------------------------------------
# (bass, voicing for pad/arpeggio)
CH = [
    ('D2',  ['D3', 'A3', 'C#4', 'E4', 'F#4']),   # 1  Dmaj9
    ('B1',  ['B2', 'F#3', 'A3', 'D4', 'E4']),    # 2  Bm(add11)
    ('G1',  ['G2', 'D3', 'F#3', 'B3', 'D4']),    # 3  Gmaj7
    ('A1',  ['A2', 'E3', 'G3', 'D4', 'E4']),     # 4  A7sus4
    ('D2',  ['D3', 'A3', 'C#4', 'E4', 'F#4']),   # 5  Dmaj9
    ('F#1', ['F#2', 'C#3', 'E3', 'A3', 'C#4']),  # 6  F#m7
    ('G1',  ['G2', 'D3', 'F#3', 'B3', 'D4']),    # 7  Gmaj7
    ('A1',  ['A2', 'E3', 'G3', 'C#4', 'E4']),    # 8  A7
    ('B1',  ['B2', 'F#3', 'A3', 'D4', 'F#4']),   # 9  Bm7
    ('G1',  ['G2', 'D3', 'F#3', 'B3', 'E4']),    # 10 Gmaj7(6)
    ('D2',  ['D3', 'A3', 'C#4', 'E4', 'F#4']),   # 11 Dmaj9
    ('A1',  ['A2', 'E3', 'G3', 'C#4', 'E4']),    # 12 A7
    ('E1',  ['E2', 'B2', 'D3', 'G3', 'B3']),     # 13 Em7   (cost)
    ('A1',  ['A2', 'E3', 'G3', 'C#4', 'E4']),    # 14 A7
    ('G1',  ['G2', 'D3', 'F#3', 'B3', 'D4']),    # 15 Gmaj7 (honesty)
    ('G1',  ['G2', 'D3', 'E3', 'Bb3', 'D4']),    # 16 Gm6   borrowed: the honest beat
    ('A1',  ['A2', 'D3', 'E3', 'G3', 'D4']),     # 17 A7sus4
    ('D2',  ['D3', 'A3', 'C#4', 'E4', 'F#4']),   # 18 Dmaj9
    ('D2',  ['D3', 'A3', 'E4', 'F#4', 'A4']),    # 19 Dadd9 ring
]

# melody: (bar, beat offset, note, length in beats, velocity)
MEL = [
    (1, 0, 'F#5', 1.5, .45), (1, 1.5, 'E5', .5, .35), (1, 2, 'A4', 2, .4),
    (2, 0, 'D5', 1.5, .42), (2, 1.5, 'C#5', .5, .32), (2, 2, 'F#4', 2, .36),
    (3, 0, 'B4', 1, .42), (3, 1, 'D5', 1, .4), (3, 2, 'F#5', 2, .5),
    (4, 0, 'E5', 2, .45), (4, 2.5, 'A4', 1.5, .34),
    (5, 0, 'F#5', 1.5, .5), (5, 1.5, 'A5', .5, .42), (5, 2, 'F#5', 1, .44), (5, 3, 'E5', 1, .4),
    (6, 0, 'C#5', 2, .44), (6, 2.5, 'E5', .5, .36), (6, 3, 'A4', 1, .34),
    (7, 0, 'B4', 1.5, .44), (7, 1.5, 'D5', .5, .38), (7, 2, 'G5', 2, .5),
    (8, 0, 'E5', 1.5, .46), (8, 1.5, 'C#5', .5, .36), (8, 2, 'A4', 2, .4),
    (9, 0, 'D5', 1.5, .48), (9, 1.5, 'F#5', .5, .42), (9, 2, 'B5', 2, .52),
    (10, 0, 'A5', 1.5, .5), (10, 1.5, 'G5', .5, .42), (10, 2, 'E5', 2, .44),
    (11, 0, 'F#5', 2, .5), (11, 2, 'A5', 1, .46), (11, 3, 'F#5', 1, .42),
    (12, 0, 'E5', 3, .46),
    (13, 0, 'B4', 1.5, .4), (13, 1.5, 'D5', .5, .34), (13, 2, 'G5', 2, .42),
    (14, 0, 'E5', 1.5, .42), (14, 1.5, 'C#5', .5, .34), (14, 2, 'A4', 2, .36),
    (15, 0, 'D5', 1.5, .4), (15, 1.5, 'B4', .5, .32), (15, 2, 'D5', 2, .38),
    (16, 0, 'D5', 1.5, .38), (16, 1.5, 'Bb4', .5, .3), (16, 2, 'G4', 2, .34),
    (17, 0, 'D5', 2, .42), (17, 2, 'E5', 2, .4),
    (18, 0, 'F#5', 4, .5),
    (19, 0, 'A5', 4, .34),
]

for b, (bass, voic) in enumerate(CH, start=1):
    t0 = (b - 1) * BAR
    last = b == NBARS
    # pad from bar 3 (title) onward, thinner in the honest section
    if b >= 3:
        g = 0.05 if b < 13 else (0.04 if b < 17 else 0.052)
        add(pad([hz(n) for n in voic], BAR + (2 if last else 0.15), g), t0, 0)
    # sub bass from bar 5
    if b >= 5 and not (15 <= b <= 16):
        add(sub(hz(bass), BAR * (1.0 if not last else 1.6)), t0, 0)
    # left hand: low bass note on beat 1 (piano)
    add(piano(hz(bass) * 2, BAR * 0.95, 0.3), t0 + 0.01, -0.25)
    # rolled chord on beat 1 (bars 1-2 sparse; full later)
    tones = voic[1:] if b > 2 else voic[1:4]
    for k, n in enumerate(tones):
        add(piano(hz(n), BAR * 0.9, 0.24 if b > 2 else 0.2), t0 + 0.02 + 0.035 * k, -0.15 + 0.08 * k)
    # gentle eighth-note arpeggio pulse in the "journey" section (bars 5-12) and the finale (15-18)
    if 5 <= b <= 12 or 17 <= b <= 18:
        pattern = [voic[1], voic[2], voic[3], voic[2], voic[4], voic[3], voic[2], voic[3]]
        for k, n in enumerate(pattern):
            v = 0.13 + (0.03 if k % 2 == 0 else 0)
            add(piano(hz(n) * 2, BEAT * 0.45, v), t0 + k * BEAT / 2 + 0.008 * rng.normal(), 0.35 if k % 2 else -0.05)

for (b, beat, n, ln, v) in MEL:
    t0 = (b - 1) * BAR + beat * BEAT + 0.006 * rng.normal()
    add(piano(hz(n), ln * BEAT, v * 1.15), t0, 0.08)

# tiny shimmer on the final chord
for k, n in enumerate(['A5', 'D6', 'E6', 'F#6']):
    add(piano(hz(n), 3, 0.12), 18 * BAR + 0.9 + k * 0.18, 0.4 - 0.25 * k)

# --- reverb: decorrelated decaying noise IR, convolved via FFT -------------------
def ir(seconds=3.2, seed=0):
    r = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    x = r.normal(0, 1, len(t)) * np.exp(-6.9 * t / seconds)
    # darken over time: simple one-pole lowpass with growing coefficient
    y = np.zeros_like(x); a = 0.0; prev = 0.0
    coef = 0.35 + 0.55 * (t / seconds)
    for i in range(0, len(x), 256):
        c = coef[i]
        seg = x[i:i + 256]
        out = np.empty_like(seg)
        for j, s in enumerate(seg):
            prev = prev * c + s * (1 - c); out[j] = prev
        y[i:i + 256] = out
    pre = int(0.018 * SR)
    return np.concatenate([np.zeros(pre), y]) / np.sqrt(np.sum(y ** 2))

def conv(x, h):
    n = len(x) + len(h) - 1
    nf = 1 << (n - 1).bit_length()
    f = np.fft.rfftfreq(nf, 1 / SR)
    hp = 1 / (1 + (280 / np.maximum(f, 1)) ** 4)  # keep the low end dry and centred
    return np.fft.irfft(np.fft.rfft(x, nf) * np.fft.rfft(h, nf) * hp, nf)[:len(x)]

irL, irR = ir(seed=1), ir(seed=2)
wetL, wetR = conv(L, irL), conv(R, irR)
mixL = L * 0.8 + wetL * 0.5
mixR = R * 0.8 + wetR * 0.5

# fade out the tail
t = np.arange(N) / SR
end = NBARS * BAR + 1.5
fade = np.clip((end + 2.2 - t) / 2.2, 0, 1)
fade = np.where(t < end, 1, fade)
fadein = np.clip(t / 0.05, 0, 1)
mixL *= fade * fadein; mixR *= fade * fadein

# gentle bus compression-ish + normalize to -1 dBFS peak
st = np.stack([mixL, mixR], 1)
st = np.tanh(st / np.max(np.abs(st)) * 1.25) / np.tanh(1.25)
st *= 10 ** (-1.5 / 20)
pcm = (st * 32767).astype('<i2')
with wave.open(sys.argv[1] if len(sys.argv) > 1 else 'music.wav', 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
print('duration', TOTAL)
