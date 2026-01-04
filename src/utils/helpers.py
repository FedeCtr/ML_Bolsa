"""
funciones auxiliares generales
"""
import pandas as pd
from datetime import datetime
from typing import Optional


def format_currency(value: float) -> str:
    """formatea un valor como moneda"""
    return f"${value:,.2f}"


def format_percentage(value: float) -> str:
    """formatea un valor como porcentaje"""
    return f"{value:.2%}"


def get_date_str(date: datetime = None) -> str:
    """obtiene fecha en formato string"""
    if date is None:
        date = datetime.now()
    return date.strftime("%Y%m%d")


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """division segura evitando division por cero"""
    try:
        return a / b if b != 0 else default
    except:
        return default


def calculate_change_pct(current: float, previous: float) -> Optional[float]:
    """calcula cambio porcentual"""
    if previous == 0:
        return None
    return ((current - previous) / previous) * 100
