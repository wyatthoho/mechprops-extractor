import pytest

from mechprops import StressStrainCurve

strain = [0.0000, 0.0005, 0.0010, 0.0015, 0.0020, 0.0025, 0.0030, 0.0035]
stress = [0.1, 12.5, 24.9, 36.9, 46.8, 54.9, 58.1, 51.1]


@pytest.fixture
def curve() -> StressStrainCurve:
    return StressStrainCurve(strain, stress)


def test_fit_modulus_iso527(curve: StressStrainCurve) -> None:
    assert curve.fit_modulus_iso527(show=False) == pytest.approx(21340.00, abs=0.01)


def test_fit_modulus_rmsprop(curve: StressStrainCurve) -> None:
    assert curve.fit_modulus_rmsprop(show=False) == pytest.approx(24868.47, abs=0.01)


def test_find_ultimate_point(curve: StressStrainCurve) -> None:
    ultimate_strain, ultimate_stress = curve.find_ultimate_point(show=False)
    assert ultimate_strain == pytest.approx(0.0030)
    assert ultimate_stress == pytest.approx(58.10)


def test_find_break_point(curve: StressStrainCurve) -> None:
    break_strain, break_stress = curve.find_break_point(show=False)
    assert break_strain == pytest.approx(0.0035)
    assert break_stress == pytest.approx(51.10)
