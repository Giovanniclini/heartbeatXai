from scipy.signal import medfilt
import dtcwt
import numpy as np

def dtcwt_denoise(signal):
    transform = dtcwt.Transform1d()

    vecs_t = transform.forward(signal, nlevels=11)

    highpasses_modifiable = list(vecs_t.highpasses)
    for i in range(len(highpasses_modifiable)):
        if i < 1 or i > 8:
            highpasses_modifiable[i] = np.zeros_like(highpasses_modifiable[i])

    modified_pyramid = dtcwt.Pyramid(vecs_t.lowpass, tuple(highpasses_modifiable))

    reconstructed_signal = transform.inverse(modified_pyramid)

    return reconstructed_signal

def ensure_odd(number):
    """Ensure the number is odd."""
    return number if number % 2 != 0 else number + 1

def remove_baseline_wander(signal, fs):
    width_p_qrs = ensure_odd(int(0.2 * fs))  # 200 ms width for P wave and QRS complex
    width_t_wave = ensure_odd(int(0.6 * fs))  # 600 ms width for T waves

    baseline_p_qrs = medfilt(signal, kernel_size=width_p_qrs)
    baseline_t_wave = medfilt(signal, kernel_size=width_t_wave)

    baseline = baseline_t_wave

    # Subtract the baseline
    signal_denoised = signal - baseline
    return signal_denoised

def denoise_ecg(signal, fs):
    denoised_signal = dtcwt_denoise(signal)

    final_signal = remove_baseline_wander(denoised_signal, fs)

    return final_signal