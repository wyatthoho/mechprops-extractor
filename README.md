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

For development (editable install):

```bash
pip install -e .
```

## Usage

Load a stress-strain CSV (`strain`, `stress` columns) into a `StressStrainCurve`,
then call any combination of its analysis methods. The example below uses one
of the sample datasets bundled in [`data/`](data):

```python
import pandas as pd

from mechprops import StressStrainCurve

df = pd.read_csv("data/polyimide-stress-strain.csv")
curve = StressStrainCurve(df["strain"], df["stress"])

curve.fit_modulus_iso527()      # Young's modulus via ISO 527 linear regression
curve.fit_modulus_rmsprop()     # Young's modulus via RMSProp area minimization
curve.find_ultimate_point()     # (strain, stress) at the ultimate (peak stress) point
curve.find_break_point()        # (strain, stress) at the detected break point
```

Each method returns its computed value and, by default, opens a plot window
illustrating the result — close it to continue. Pass `show=False` to skip the
plot and get the value only.
