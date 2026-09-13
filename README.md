# Mechanical Properties Extractor

Extracts Young's modulus, ultimate point, and break-point properties from
stress-strain curve data, via ISO 527 linear regression, a custom
RMSProp-based curve fit, or break-point detection.

## Features

- Young's modulus via ISO 527 secant-line linear regression
- Young's modulus via a custom RMSProp-based enclosed-area fit
- Ultimate (peak stress) point detection on the stress-strain curve
- Break-point (failure) detection on the stress-strain curve
- Built-in plots and an animation of the RMSProp fitting process

## Installation

```bash
pip install mechprops-extractor
```

## Usage

Pass your strain and stress data (any array-like: list, tuple, NumPy array,
pandas Series, ...) into a `StressStrainCurve`, then call any combination of
its analysis methods:

```python
from mechprops import StressStrainCurve

strain = [0.0000, 0.0005, 0.0010, 0.0015, 0.0020, 0.0025, 0.0030, 0.0035]
stress = [0.1, 12.5, 24.9, 36.9, 46.8, 54.9, 58.1, 51.1]

curve = StressStrainCurve(strain, stress)

modulus_iso527 = curve.fit_modulus_iso527()
modulus_rmsprop = curve.fit_modulus_rmsprop()
ultimate_strain, ultimate_stress = curve.find_ultimate_point()
break_strain, break_stress = curve.find_break_point()

print(f"Modulus (ISO 527): {modulus_iso527:.2f}")
print(f"Modulus (RMSProp): {modulus_rmsprop:.2f}")
print(f"Ultimate point: strain={ultimate_strain:.4f}, stress={ultimate_stress:.2f}")
print(f"Break point: strain={break_strain:.4f}, stress={break_stress:.2f}")
```

Output:

```
Modulus (ISO 527): 21340.00
Modulus (RMSProp): 24868.47
Ultimate point: strain=0.0030, stress=58.10
Break point: strain=0.0035, stress=51.10
```

Each method returns its computed value and, by default, opens a plot window
illustrating the result — close it to continue. Pass `show=False` to skip the
plot and get the value only. Pass `save_path="path/to/file.png"` to save the
plot as a PNG (in addition to displaying it).
