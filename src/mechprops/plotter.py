import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation
from matplotlib.collections import PathCollection, PolyCollection
from matplotlib.lines import Line2D
from matplotlib.text import Annotation

# Shared constants
MARKER_SIZE = 49
MARKER_ZORDER = 5
LINE_WIDTH = 1.5
SECANT_POSITION_DIVISOR = 50
ANNOTATION_OFFSET_LEFT = (5, 0)
ANNOTATION_OFFSET_RIGHT = (-10, 0)
COLOR_BLUE = "tab:blue"
COLOR_ORANGE = "tab:orange"

# RmsPropAnimation constants
FIG_SIZE_RMSPROP = (4.8, 6)
HEIGHT_RATIOS_RMSPROP = [2.4, 0.8, 0.8]
ANI_INTERVAL = 50
FRAME_SIZE_REQUEST = 50
ALPHA_FILL = 0.3
COLOR_FILL = "lightsteelblue"
PROGRESS_MARKER_EDGECOLOR = "white"

# Iso527Graph constants
FIG_SIZE_ISO527 = (4.8, 3.6)

# UltimatePointGraph constants
FIG_SIZE_ULTIMATE = (4.8, 3.6)

# BreakDetectGraph constants
FIG_SIZE_BREAK = (4.8, 4.8)
HEIGHT_RATIOS_BREAK = [2.4, 0.8]

Point = tuple[float, float]


def _get_secant_line(modulus: float, strain: pd.Series, stress: pd.Series) -> pd.Series:
    """Calculates a secant line pinned directly to the initial data point."""
    intercept = stress.values[0] - modulus * strain.values[0]
    return modulus * strain + intercept


def _get_annotation_pos(strain: pd.Series, secant: pd.Series) -> Point:
    """Determines a stable, indexed coordinate for placing text annotations."""
    idx = max(1, len(strain) // SECANT_POSITION_DIVISOR)
    return float(strain.iloc[idx]), float(secant.iloc[idx])


def _show_figure(fig: plt.Figure) -> None:
    """Displays only the given figure, leaving any other open figures untouched."""
    fig.canvas.manager.show()
    fig.canvas.mpl_connect("close_event", lambda _: fig.canvas.stop_event_loop())
    fig.canvas.start_event_loop()


class RmsPropAnimation:
    """Animates the RMSProp fit: a moving secant line over the stress-strain
    curve alongside the learning-rate and loss traces. Call play() to start."""

    _fig: plt.Figure
    _ax_norm: plt.Axes
    _ax_lr: plt.Axes
    _ax_loss: plt.Axes
    _line_secant: Line2D
    _ann_secant: Annotation
    _lr_marker: PathCollection
    _loss_marker: PathCollection
    _fill: PolyCollection | None
    _ani: animation.FuncAnimation | None
    _rids: list[int]

    def __init__(
        self,
        strain: pd.Series,
        stress: pd.Series,
        m_records: list[float],
        lr_records: list[float],
        loss_records: list[float],
    ):
        self.strain = strain
        self.stress = stress
        self.m_records = m_records
        self.lr_records = lr_records
        self.loss_records = loss_records

        self._fig, axs = plt.subplots(
            nrows=3,
            ncols=1,
            figsize=FIG_SIZE_RMSPROP,
            tight_layout=True,
            height_ratios=HEIGHT_RATIOS_RMSPROP,
        )
        self._ax_norm, self._ax_lr, self._ax_loss = axs

        self._setup_norm_axis()
        self._setup_lr_axis()
        self._setup_loss_axis()
        self._fill = None
        self._ani = None
        self._rids = []

    def _setup_norm_axis(self) -> None:
        self._ax_norm.plot(
            self.strain,
            self.stress,
            label="Stress-Strain Curve",
            linewidth=LINE_WIDTH,
            color=COLOR_BLUE,
        )

        # Placeholder y=x line, overwritten with the real secant on the first animation frame
        (self._line_secant,) = self._ax_norm.plot(
            self.strain,
            self.strain,
            label="RMSProp Fit",
            linewidth=LINE_WIDTH,
            linestyle="--",
            color=COLOR_ORANGE,
        )
        self._ann_secant = self._ax_norm.annotate(
            text="",
            xy=_get_annotation_pos(self.strain, self.strain),
            xytext=ANNOTATION_OFFSET_LEFT,
            textcoords="offset points",
            color=COLOR_ORANGE,
        )

        self._ax_norm.legend()
        self._ax_norm.set(
            xlabel="Normalized Strain",
            ylabel="Normalized Stress",
            title="RMSProp Optimization",
        )
        self._ax_norm.grid(True)
        self.initial_ybound = self._ax_norm.get_ybound()

    def _setup_lr_axis(self) -> None:
        self._ax_lr.semilogy(self.lr_records, linewidth=LINE_WIDTH, color=COLOR_BLUE)
        self._lr_marker = self._ax_lr.scatter(
            0,
            self.lr_records[0],
            facecolors=COLOR_ORANGE,
            edgecolors=PROGRESS_MARKER_EDGECOLOR,
            linewidth=LINE_WIDTH,
            zorder=MARKER_ZORDER,
            s=MARKER_SIZE,
        )
        self._ax_lr.set(xlabel="Iteration", ylabel="LR")
        self._ax_lr.grid(True)

    def _setup_loss_axis(self) -> None:
        self._ax_loss.plot(self.loss_records, linewidth=LINE_WIDTH, color=COLOR_BLUE)
        self._loss_marker = self._ax_loss.scatter(
            0,
            self.loss_records[0],
            facecolors=COLOR_ORANGE,
            edgecolors=PROGRESS_MARKER_EDGECOLOR,
            linewidth=LINE_WIDTH,
            zorder=MARKER_ZORDER,
            s=MARKER_SIZE,
        )
        self._ax_loss.set(xlabel="Iteration", ylabel="Loss")
        self._ax_loss.grid(True)

    def _get_record_indices(self) -> list[int]:
        total = len(self.m_records)
        step = max(1, total // FRAME_SIZE_REQUEST)
        return list(range((total - 1) % step, total, step))

    def _update_frame(self, fid: int) -> None:
        rid = self._rids[fid]
        m = self.m_records[rid]
        secant = _get_secant_line(m, self.strain, self.stress)

        # Redraw secant profile line adjustments
        self._line_secant.set_ydata(secant)
        self._ann_secant.set_text(f"E = {m:.2f}")
        self._ann_secant.xy = _get_annotation_pos(self.strain, secant)

        # Update shaded regional error tracking
        if self._fill:
            self._fill.remove()

        self._fill = self._ax_norm.fill_between(
            x=self.strain,
            y1=self.stress,
            y2=secant,
            where=(self.stress > secant),
            alpha=ALPHA_FILL,
            color=COLOR_FILL,
        )

        # Update dynamic progress indicator markers
        self._lr_marker.set_offsets(np.array([[rid, self.lr_records[rid]]]))
        self._loss_marker.set_offsets(np.array([[rid, self.loss_records[rid]]]))
        self._ax_norm.set_ybound(*self.initial_ybound)

    def play(self) -> None:
        self._rids = self._get_record_indices()
        self._ani = animation.FuncAnimation(
            fig=self._fig,
            func=self._update_frame,
            frames=len(self._rids),
            interval=ANI_INTERVAL,
            blit=False,
            repeat=False,
        )
        _show_figure(self._fig)


class Iso527Graph:
    """Renders the static ISO 527 stress-strain curve with the fitted secant
    line. Call show() to display it."""

    _fig: plt.Figure

    def __init__(
        self,
        strain: pd.Series,
        stress: pd.Series,
        secant: pd.Series,
        m: float,
    ):
        self._fig, ax = plt.subplots(figsize=FIG_SIZE_ISO527, tight_layout=True)

        ax.plot(
            strain,
            stress,
            label="Stress-Strain Curve",
            linewidth=LINE_WIDTH,
            color=COLOR_BLUE,
        )
        ybound = ax.get_ybound()

        ax.plot(
            strain,
            secant,
            label="ISO 527 Fit",
            linewidth=LINE_WIDTH,
            linestyle="--",
            color=COLOR_ORANGE,
        )

        ax.annotate(
            text=f"E = {m:.1f}",
            xy=_get_annotation_pos(strain, secant),
            xytext=ANNOTATION_OFFSET_LEFT,
            textcoords="offset points",
            color=COLOR_ORANGE,
        )

        ax.set(xlabel="Strain", ylabel="Stress", title="ISO 527 Regression")
        ax.grid(True)
        ax.set_ybound(*ybound)
        ax.legend()

    def show(self) -> None:
        _show_figure(self._fig)


class UltimatePointGraph:
    """Renders the stress-strain curve with the ultimate (peak stress) point
    marked. Call show() to display it."""

    _fig: plt.Figure

    def __init__(
        self,
        strain: pd.Series,
        stress: pd.Series,
        ultimate_point: Point,
    ):
        self._fig, ax = plt.subplots(figsize=FIG_SIZE_ULTIMATE, tight_layout=True)

        ax.plot(
            strain,
            stress,
            label="Stress-Strain Curve",
            linewidth=LINE_WIDTH,
            color=COLOR_BLUE,
        )

        ux, uy = ultimate_point
        ax.scatter(
            ux,
            uy,
            marker="^",
            s=MARKER_SIZE,
            c=COLOR_ORANGE,
            label="Ultimate Point",
            zorder=MARKER_ZORDER,
        )
        ax.annotate(
            f"({ux:.2f}, {uy:.2f})",
            xy=(ux, uy),
            xytext=ANNOTATION_OFFSET_RIGHT,
            textcoords="offset points",
            ha="right",
            va="center",
            color=COLOR_ORANGE,
        )

        ax.set(xlabel="Strain", ylabel="Stress", title="Ultimate Point Detection")
        ax.grid(True)
        ax.legend()

    def show(self) -> None:
        _show_figure(self._fig)


class BreakDetectGraph:
    """Renders the stress-strain curve with the detected break point and its
    underlying gradient trace. Call show() to display it."""

    _fig: plt.Figure
    _ax_norm: plt.Axes
    _ax_break: plt.Axes

    def __init__(
        self,
        strain: pd.Series,
        stress: pd.Series,
        bindex: np.ndarray,
        threshold: float,
        break_point: tuple[float, float],
    ):
        self._fig, axs = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=FIG_SIZE_BREAK,
            tight_layout=True,
            height_ratios=HEIGHT_RATIOS_BREAK,
        )

        self._ax_norm, self._ax_break = axs

        self._ax_norm.plot(
            strain,
            stress,
            label="Stress-Strain Curve",
            linewidth=LINE_WIDTH,
            color=COLOR_BLUE,
        )

        self._ax_norm.set(ylabel="Normalized Stress", title="Break Point Detection")
        self._ax_norm.grid(True)

        bx, by = break_point
        self._ax_norm.scatter(
            bx,
            by,
            marker="x",
            s=MARKER_SIZE,
            c=COLOR_ORANGE,
            label="Break Point",
            zorder=MARKER_ZORDER,
        )
        self._ax_norm.annotate(
            f"({bx:.2f}, {by:.2f})",
            xy=(bx, by),
            xytext=ANNOTATION_OFFSET_RIGHT,
            textcoords="offset points",
            ha="right",
            va="center",
            color=COLOR_ORANGE,
        )
        self._ax_norm.legend()

        self._ax_break.plot(strain, bindex)
        self._ax_break.axhline(threshold, color=COLOR_ORANGE, linestyle="--")
        self._ax_break.grid(True)
        self._ax_break.set(xlabel="Normalized Strain", ylabel="Gradient")

    def show(self) -> None:
        _show_figure(self._fig)
