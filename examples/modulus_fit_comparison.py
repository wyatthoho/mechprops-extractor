"""Compares three Young's modulus fits on the README example data (strain/
stress with a deliberately distorted initial toe region): the naive initial
slope, ISO 527 windowed regression, and the RMSProp enclosed-area fit.
Renders a single overlay chart of all three secant lines against the raw
curve.
"""

import textwrap

import matplotlib.pyplot as plt
import numpy as np

from mechprops import StressStrainCurve

strain = [
    0.00000,
    0.00011,
    0.00020,
    0.00030,
    0.00040,
    0.00050,
    0.00075,
    0.00100,
    0.00125,
    0.00150,
    0.00175,
    0.00200,
    0.00225,
    0.00250,
    0.00275,
    0.00300,
    0.00325,
    0.00350,
    0.00353,
]
stress = [
    0.00,
    0.96,
    3.00,
    6.00,
    9.30,
    12.50,
    18.91,
    24.90,
    31.22,
    36.90,
    43.77,
    48.86,
    52.43,
    54.90,
    55.86,
    56.80,
    55.11,
    53.15,
    25.00,
]

xs = np.asarray(strain)
ys = np.asarray(stress)

curve = StressStrainCurve(strain, stress)

m_naive = (ys[1] - ys[0]) / (xs[1] - xs[0])
m_iso527 = curve.fit_modulus_iso527(show=False)
m_rmsprop = curve.fit_modulus_rmsprop(show=False)

print(f"Naive (first two points): {m_naive:.1f}")
print(f"ISO 527 (fixed window):   {m_iso527:.1f}")
print(f"RMSProp (enclosed area):  {m_rmsprop:.1f}")


def secant_of(m: float) -> np.ndarray:
    return m * xs + (ys[0] - m * xs[0])


secant_naive = secant_of(m_naive)
secant_iso527 = secant_of(m_iso527)
secant_rmsprop = secant_of(m_rmsprop)

y_bound = (ys.min() - 3, ys.max() + 3)


def new_ax(title: str):
    fig, ax = plt.subplots(figsize=(6, 4.5), tight_layout=True)
    ax.plot(
        xs,
        ys,
        "o-",
        color="black",
        linewidth=2.0,
        markersize=5,
        label="Stress-Strain Curve",
    )
    ax.set(xlabel="Strain", ylabel="Stress, MPa", title=title, ylim=y_bound)
    ax.grid(True)
    return fig, ax


fig, ax = new_ax("Comparison of Young's Modulus From Different Methods")
ax.plot(
    xs,
    secant_naive,
    ":",
    color="gray",
    linewidth=3.5,
    alpha=0.55,
    label=f"Initial Slope, E={m_naive:.0f} MPa",
)
ax.plot(
    xs,
    secant_iso527,
    "-.",
    color="gray",
    linewidth=3.5,
    alpha=0.55,
    label=f"ISO 527, E={m_iso527:.0f} MPa",
)
ax.plot(
    xs,
    secant_rmsprop,
    "-",
    color="fuchsia",
    linewidth=3.5,
    alpha=0.6,
    label=f"RMSProp, E={m_rmsprop:.0f} MPa",
)
ax.legend(fontsize=8, loc="best", handlelength=4)


def annotate_line(x: float, y: float, color: str, text: str, wrap: int = 25) -> None:
    """Labels a secant line directly on the plot, near the given x."""
    ax.annotate(
        textwrap.fill(text, width=wrap),
        xy=(x, y),
        xytext=(6, 6),
        textcoords="offset points",
        color=color,
        fontsize=9,
        fontweight="normal",
    )


annotate_line(0.0015, 6.4, "gray", "Slope calculated from the first 2 points.")
annotate_line(0.0018, 30.0, "gray", "Slope regressed over the standard-defined strain range, e.g., ISO 527 (0.0005-0.0025).", wrap=32)
annotate_line(0.0000, 26.0, "fuchsia", "Slope fitted using the RMSProp optimization algorithm.")

fig.savefig("examples/modulus_fit_comparison.png", dpi=150)


print("Saved: examples/modulus_fit_comparison.png")
