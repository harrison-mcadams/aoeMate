import numpy as np
import wave
import os
import random

def save_wav(filename, data, sample_rate=44100):
    # Normalize
    max_val = np.max(np.abs(data))
    if max_val > 0:
        data = data / max_val * 0.8
    
    data = (data * 32767).astype(np.int16)
    with wave.open(filename, 'w') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(data.tobytes())
    print(f"Generated {filename}")

def apply_envelope(audio, attack, release, sample_rate=44100):
    total = len(audio)
    att_samp = int(attack * sample_rate)
    rel_samp = int(release * sample_rate)
    env = np.ones(total)
    if att_samp > 0:
        env[:att_samp] = np.linspace(0, 1, att_samp)
    if rel_samp > 0:
        start_rel = max(0, total - rel_samp)
        env[start_rel:] = np.linspace(1, 0, total - start_rel)
    return audio * env

def generate_tones():
    if not os.path.exists('sounds'):
        os.makedirs('sounds')
    
    sr = 44100
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # 1. Soft Sine (Classic)
    tone = np.sin(2 * np.pi * 440 * t)
    save_wav('sounds/tone_01_soft_sine.wav', apply_envelope(tone, 0.5, 0.5, sr), sr)
    
    # 2. Warm Hum (Rich Lows)
    tone = np.zeros_like(t)
    for n in range(1, 6):
        tone += (1/n) * np.sin(2 * np.pi * (220 * n) * t)
    save_wav('sounds/tone_02_warm_hum.wav', apply_envelope(tone, 0.5, 0.5, sr), sr)
    
    # 3. Glassy Drone (Ethereal)
    tone = np.sin(2 * np.pi * 698.46 * t) + 0.3 * np.sin(2 * np.pi * 1396.92 * t)
    mod = 0.05 * np.sin(2 * np.pi * 0.2 * t)
    save_wav('sounds/tone_03_glassy_drone.wav', apply_envelope(tone * (1+mod), 0.5, 0.5, sr), sr)
    
    # 4. Pure Ether (Minimal)
    tone = np.sin(2 * np.pi * 523.25 * t) + 0.1 * np.sin(2 * np.pi * 784.88 * t)
    save_wav('sounds/tone_04_pure_ether.wav', apply_envelope(tone, 0.5, 0.5, sr), sr)
    
    # 5. Low Rumble (Subtle tension)
    # 110Hz + noise
    noise = np.random.normal(0, 0.1, len(t))
    # Low pass filter noise (simple moving average)
    window_size = 50
    noise_smooth = np.convolve(noise, np.ones(window_size)/window_size, mode='same')
    tone = np.sin(2 * np.pi * 110 * t) + 0.5 * noise_smooth
    save_wav('sounds/tone_05_low_rumble.wav', apply_envelope(tone, 1.0, 1.0, sr), sr)
    
    # 6. Shimmer Pad (Bright)
    # Major chord high up
    freqs = [523.25, 659.25, 783.99, 1046.50] # C5, E5, G5, C6
    tone = np.zeros_like(t)
    for f in freqs:
        tone += np.sin(2 * np.pi * f * t)
    # Fast tremolo
    trem = 0.1 * np.sin(2 * np.pi * 6.0 * t)
    save_wav('sounds/tone_06_shimmer_pad.wav', apply_envelope(tone * (1+trem), 0.5, 0.5, sr), sr)
    
    # 7. Organ Swell (Church-like)
    # Additive synthesis with many harmonics
    base = 261.63 # C4
    tone = np.zeros_like(t)
    harmonics = [1, 2, 3, 4, 6, 8]
    weights = [1.0, 0.5, 0.3, 0.2, 0.1, 0.05]
    for h, w in zip(harmonics, weights):
        tone += w * np.sin(2 * np.pi * (base * h) * t)
    save_wav('sounds/tone_07_organ_swell.wav', apply_envelope(tone, 1.5, 1.5, sr), sr)
    
    # 8. Bamboo Flute (Breathy)
    # Sine + filtered noise
    base = 440.0
    sine = np.sin(2 * np.pi * base * t)
    breath = np.random.normal(0, 0.3, len(t))
    breath = np.convolve(breath, np.ones(20)/20, mode='same')
    tone = sine + breath
    # Tremolo
    tone = tone * (1 + 0.1 * np.sin(2 * np.pi * 5 * t))
    save_wav('sounds/tone_08_bamboo_flute.wav', apply_envelope(tone, 0.2, 0.2, sr), sr)
    
    # 9. Crystal Glass (Inharmonic)
    freqs = [880, 880*1.5, 880*2.1, 880*2.9]
    tone = np.zeros_like(t)
    for f in freqs:
        tone += np.sin(2 * np.pi * f * t) * np.exp(-0.5 * t) # Decay even though sustained file
    # Make it sustain a bit more
    tone += 0.2 * np.sin(2 * np.pi * 880 * t)
    save_wav('sounds/tone_09_crystal_glass.wav', apply_envelope(tone, 0.1, 1.0, sr), sr)
    
    # 10. Analog Drift (Detuned Saws)
    # Approx saw with sines
    def approx_saw(f, t, n=10):
        s = np.zeros_like(t)
        for i in range(1, n+1):
            s += (1/i) * np.sin(2 * np.pi * f * i * t)
        return s
    
    tone = approx_saw(220, t) + approx_saw(221, t) # Detuned
    # Low pass filter effect (simulate by reducing higher harmonics over time? Hard in simple script)
    # Just apply envelope
    save_wav('sounds/tone_10_analog_drift.wav', apply_envelope(tone, 0.5, 0.5, sr), sr)
    
    # 11. Soft Pulse (Rhythmic)
    tone = np.sin(2 * np.pi * 329.63 * t) # E4
    pulse = 0.5 * (1 + np.sin(2 * np.pi * 2.0 * t)) # 2Hz pulse
    save_wav('sounds/tone_11_soft_pulse.wav', apply_envelope(tone * pulse, 0.1, 0.1, sr), sr)
    
    # 12. Underwater (Muffled)
    noise = np.random.normal(0, 1, len(t))
    # Strong LP filter
    window = 200
    noise_lp = np.convolve(noise, np.ones(window)/window, mode='same')
    tone = noise_lp + 0.5 * np.sin(2 * np.pi * 110 * t)
    save_wav('sounds/tone_12_underwater.wav', apply_envelope(tone, 1.0, 1.0, sr), sr)
    
    # 13. Space Ambience (Sci-fi)
    tone = np.sin(2 * np.pi * 55 * t) # Deep bass
    tone += 0.1 * np.sin(2 * np.pi * 880 * t + 5 * np.sin(2 * np.pi * 0.5 * t)) # FM mod high pitch
    save_wav('sounds/tone_13_space_ambience.wav', apply_envelope(tone, 2.0, 2.0, sr), sr)

if __name__ == "__main__":
    generate_tones()
