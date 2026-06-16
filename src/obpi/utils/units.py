"""Unit conversions and normalizers for OBPI."""


def meters_per_second_to_kmh(v: float) -> float:
    """Convert metres per second to kilometres per hour."""
    return v * 3.6
