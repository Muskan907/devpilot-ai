"""Core financial & scientific calculator module for the sample repository."""

import math
from typing import List, Union


class AdvancedCalculator:
    """Provides high precision arithmetic and statistics."""

    def add(self, a: float, b: float) -> float:
        return a + b

    def subtract(self, a: float, b: float) -> float:
        return a - b

    def multiply(self, a: float, b: float) -> float:
        return a * b

    def divide(self, a: float, b: float) -> float:
        if b == 0:
            raise ZeroDivisionError("Cannot divide by zero")
        return a / b

    def power(self, base: float, exponent: float) -> float:
        return math.pow(base, exponent)

    def mean(self, numbers: List[Union[int, float]]) -> float:
        if not numbers:
            raise ValueError("Input list cannot be empty")
        return sum(numbers) / len(numbers)

    # Note: A deliberate missing feature or bug can be requested by developer requirements:
    # e.g., 'standard_deviation' or 'compound_interest' or 'fibonacci' or 'safe_percentage'.

