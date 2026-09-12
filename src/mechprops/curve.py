import numpy as np
from numpy.typing import ArrayLike
from scipy import integrate, stats

from mechprops.plotter import (
    BreakDetectGraph,
    Iso527Graph,
    RmsPropAnimation,
    UltimatePointGraph,
)

# Analysis Thresholds
TINY_MODULUS = 1.0e-5
CONVERGENCE_EPS = 1e-9
BREAK_THRESHOLD = 0.1

# RMSProp Hyperparameters
RMS_PROP_BASE_LR = 0.01
RMS_PROP_ALPHA = 0.9
RMS_PROP_EPS = 1e-8

# ISO-527
STRAIN_LOWER = 0.0005
STRAIN_UPPER = 0.0025

# Modulus Fitting Defaults
DEFAULT_TOL = 2.0e-5
DEFAULT_MAX_ITER = 20_000


class StressStrainCurve:
    """Fits Young's modulus from a stress-strain curve, via ISO 527 linear
    regression or a custom RMSProp-based enclosed-area minimization."""

    def __init__(
        self,
        strain: ArrayLike,
        stress: ArrayLike,
    ) -> None:
        self.strain_raw = np.asarray(strain, dtype=float)
        self.stress_raw = np.asarray(stress, dtype=float)

        idx = int(np.argmax(self.stress_raw))
        self.ultimate_x = float(self.strain_raw[idx])
        self.ultimate_y = float(self.stress_raw[idx])

        self.strain_norm = self.strain_raw / self.ultimate_x
        self.stress_norm = self.stress_raw / self.ultimate_y

    def find_ultimate_point(
        self, show: bool = True, save_path: str | None = None
    ) -> tuple[float, float]:
        """Returns the (strain, stress) at maximum stress, the ultimate
        point of the curve.

        Args:
            show: Plots the ultimate point over the raw input curve.
            save_path: If given (and show is True), saves the plot as a PNG
                to this path.

        Returns:
            The (strain, stress) of the ultimate point in raw units.
        """
        ultimate_point = (self.ultimate_x, self.ultimate_y)
        if show:
            graph = UltimatePointGraph(self.strain_raw, self.stress_raw, ultimate_point)
            graph.show(save_path)

        return ultimate_point

    def fit_modulus_iso527(self, show: bool = True, save_path: str | None = None) -> float:
        """Fits modulus via ISO 527 linear regression over the
        [STRAIN_LOWER, STRAIN_UPPER] strain range.

        Args:
            show: Plots the fitted secant line over the raw input curve.
            save_path: If given (and show is True), saves the plot as a PNG
                to this path.

        Raises:
            ValueError: Fewer than 2 raw strain samples fall in the range.
        """
        xs = self.strain_raw
        ys = self.stress_raw

        target = (xs >= STRAIN_LOWER) & (xs <= STRAIN_UPPER)
        if target.sum() < 2:
            raise ValueError("Not enough raw strain data in the range for regression.")

        res = stats.linregress(xs[target], ys[target], "greater")
        m = res.slope
        shift = res.intercept

        if show:
            secant = m * xs + shift
            graph = Iso527Graph(xs, ys, secant, m)
            graph.show(save_path)

        return m

    def fit_modulus_rmsprop(
        self,
        tol: float = DEFAULT_TOL,
        max_iter: int = DEFAULT_MAX_ITER,
        show: bool = True,
        save_path: str | None = None,
    ) -> float:
        """Fits modulus by using RMSProp to minimize the area enclosed between
        the normalized curve and its secant line.

        Args:
            tol: Stops iterating once the relative change in modulus is below this.
            max_iter: Maximum number of RMSProp iterations to run.
            show: Plays an animation of the fitting process.
            save_path: If given (and show is True), saves the final animation
                frame as a PNG to this path.

        Returns:
            The fitted modulus, rescaled back to raw strain/stress units.
        """
        m = 1.0
        v = 0.0

        xs = self.strain_norm
        ys = self.stress_norm

        x0 = float(self.strain_norm[0])
        y0 = float(self.stress_norm[0])

        m_records = []
        loss_records = []
        lr_records = []

        for _ in range(max_iter):
            area = self._compute_enclosed_area(m, x0, y0, xs, ys)
            area_eps = self._compute_enclosed_area(m + TINY_MODULUS, x0, y0, xs, ys)
            gradient = (area_eps - area) / TINY_MODULUS
            v, lr = self._update_rmsprop(v, gradient)

            m_records.append(m)
            loss_records.append(area)
            lr_records.append(lr)

            m_new = m - lr * gradient
            if abs(m_new - m) / (abs(m) + CONVERGENCE_EPS) < tol:
                break
            m = m_new

        m_scale = m * self.ultimate_y / self.ultimate_x

        if show:
            ani = RmsPropAnimation(xs, ys, m_records, lr_records, loss_records)
            ani.play(save_path)

        return m_scale

    def find_break_point(
        self,
        threshold: float = BREAK_THRESHOLD,
        show: bool = True,
        save_path: str | None = None,
    ) -> tuple[float, float]:
        """Locates the first point where the normalized stress curve's slope
        drops below -threshold, signaling the specimen break.

        Args:
            threshold: Slope drop (in normalized stress per sample) that
                signals a break.
            show: Plots the detected break point over the normalized curve.
            save_path: If given (and show is True), saves the plot as a PNG
                to this path.

        Returns:
            The (strain, stress) of the break point in raw units, or the last
            sample's coordinates if no break is detected.
        """
        xs = self.strain_norm
        ys = self.stress_norm

        gradient = -np.gradient(ys)
        positions = np.where(gradient > threshold)[0]
        position = int(positions[0]) if positions.size > 0 else -1

        if show:
            break_pt_norm = float(xs[position]), float(ys[position])
            graph = BreakDetectGraph(xs, ys, gradient, threshold, break_pt_norm)
            graph.show(save_path)

        return float(self.strain_raw[position]), float(self.stress_raw[position])

    @staticmethod
    def _compute_enclosed_area(
        m: float, x0: float, y0: float, xs: np.ndarray, ys: np.ndarray
    ) -> float:
        shift = y0 - m * x0
        secant = m * xs + shift
        clipped = np.clip(ys - secant, 0, None)
        return float(integrate.trapezoid(clipped, xs))

    @staticmethod
    def _update_rmsprop(v: float, gradient: float) -> tuple[float, float]:
        v_new = RMS_PROP_ALPHA * v + (1 - RMS_PROP_ALPHA) * (gradient**2)
        lr = RMS_PROP_BASE_LR / (v_new**0.5 + RMS_PROP_EPS)
        return v_new, lr
