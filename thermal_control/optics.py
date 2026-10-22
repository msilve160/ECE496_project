"""Ring optical response shared by the plant and by the controller's internal model."""
import numpy as np


def lorentz(delta, hwhm):
    """Normalised Lorentzian drop response (1 at resonance)."""
    x = delta / hwhm
    return 1.0 / (1.0 + x * x)


def drop_avg(delta, hwhm, mod_shift, frac_ones=0.5):
    """Drop response averaged over the NRZ data: the drive shifts the resonance by +/- mod_shift.

    A '1' shifts the resonance by +mod_shift, a '0' by -mod_shift. The monitor PD is far slower
    than 53 GBd, so it sees the average weighted by the fraction of ones in its sample period.
    With mod_shift = 0 (squelch) this is the bare Lorentzian.
    """
    return frac_ones * lorentz(delta + mod_shift, hwhm) + (1.0 - frac_ones) * lorentz(delta - mod_shift, hwhm)


_NQ = 24
_THETA = 2 * np.pi * (np.arange(_NQ) + 0.5) / _NQ
_COS1 = np.cos(_THETA)
_COS2 = np.cos(2 * _THETA)


def lockin_ratios(delta, a, hwhm, mod_shift):
    """Ideal lock-in outputs normalised by the DC level.

    For a detuning wobble delta + a*cos(theta), returns
      z1 = y1 / y0  (first harmonic, ~ a T'/T: slope with sign)
      z2 = y2 / y0  (second harmonic, ~ a^2 T''/(4T): curvature)
    exactly (by quadrature over theta), with y_k = 2 <L cos(k theta)>, y0 = <L>.
    Broadcasts over delta / a / hwhm. Dividing by y0 removes laser power and PD gain.
    """
    d = np.asarray(delta)[..., None] + np.asarray(a)[..., None] * _COS1
    h = np.asarray(hwhm)[..., None]
    m = np.asarray(mod_shift, float)
    lv = drop_avg(d, h, m[..., None] if m.ndim else m)
    y0 = lv.mean(-1)
    y1 = 2 * (lv * _COS1).mean(-1)
    y2 = 2 * (lv * _COS2).mean(-1)
    return y1 / y0, y2 / y0


def through_levels(delta, hwhm, depth, mod_shift):
    """Through-port transmission for a transmitted '1' and '0'."""
    t_a = 1 - depth * lorentz(delta + mod_shift, hwhm)
    t_b = 1 - depth * lorentz(delta - mod_shift, hwhm)
    return np.maximum(t_a, t_b), np.minimum(t_a, t_b)


def er_db(delta, hwhm, depth, mod_shift):
    hi, lo = through_levels(delta, hwhm, depth, mod_shift)
    return 10 * np.log10(hi / lo)


def oma_dbm(delta, hwhm, depth, mod_shift, p_mw):
    hi, lo = through_levels(delta, hwhm, depth, mod_shift)
    return 10 * np.log10(p_mw * (hi - lo))


def inflection(hwhm, mod_shift, side=+1):
    """Zero of the second derivative of the data-averaged drop curve (maximum slope) on one side."""
    x = np.linspace(0.05, 3.0, 20000) * hwhm * side
    f = drop_avg(x, hwhm, mod_shift)
    d2 = np.gradient(np.gradient(f, x), x)
    s = np.where(np.diff(np.sign(d2)) != 0)[0]
    i = s[0] if side > 0 else s[0]
    # linear interpolation of the sign change
    x0, x1, y0, y1 = x[i], x[i + 1], d2[i], d2[i + 1]
    return x0 - y0 * (x1 - x0) / (y1 - y0)

