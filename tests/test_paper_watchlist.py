"""tests de paper trading y watchlists"""
import numpy as np
import pandas as pd
import pytest

from src.trading.paper import PaperTrader
from src.ml.watchlist import WatchlistStore


@pytest.fixture
def trader(tmp_path):
    return PaperTrader(db_path=str(tmp_path / 'paper.db'))


def _signal(ticker, date, side='largo', entry=100.0, sl=98.0, tp1=103.0, tp2=105.0):
    return {
        'ticker': ticker, 'as_of': date, 'signal': 'COMPRA' if side == 'largo' else 'VENTA',
        'side': side, 'entry': entry, 'stop_loss': sl,
        'take_profits': [{'name': 'TP1', 'price': tp1}, {'name': 'TP2', 'price': tp2}],
        'confidence_pct': 58.0, 'conviction_pct': 12.0, 'prob_up': 0.58,
        'price': entry, 'position_size_pct': 5.0, 'regime': {'status': 'normal'},
    }


def _provider_hit_tp(ticker):
    """dia siguiente toca TP1"""
    idx = pd.bdate_range('2024-01-02', periods=3)
    return pd.DataFrame({
        'Open': [100, 100, 102], 'High': [101, 104, 104], 'Low': [99, 99.5, 101],
        'Close': [100, 103.5, 104], 'Volume': [1e6] * 3,
    }, index=idx)


def _provider_hit_sl(ticker):
    idx = pd.bdate_range('2024-01-02', periods=3)
    return pd.DataFrame({
        'Open': [100, 100, 97], 'High': [101, 100.5, 98], 'Low': [99, 97.5, 96],
        'Close': [100, 98, 96.5], 'Volume': [1e6] * 3,
    }, index=idx)


def test_record_and_resolve_hit_tp1(trader):
    trader.record_signal(_signal('AAA', '2024-01-02'))
    n = trader.resolve_pending(lambda t: _provider_hit_tp(t))
    assert n == 1
    p = trader.performance()
    assert p['available'] and p['n_resolved'] == 1
    assert p['hit_tp1'] == 1 and p['hit_sl'] == 0
    assert p['directional_accuracy'] == 1.0


def test_resolve_hit_sl_counts_loss(trader):
    trader.record_signal(_signal('BBB', '2024-01-02'))
    trader.resolve_pending(lambda t: _provider_hit_sl(t))
    p = trader.performance()
    assert p['hit_sl'] == 1 and p['hit_tp1'] == 0
    assert p['directional_accuracy'] == 0.0
    assert p['win_rate'] == 0.0


def test_pending_not_resolved_without_future_data(trader):
    trader.record_signal(_signal('CCC', '2024-01-04'))  # ultimo dia del provider
    def provider(t):
        df = _provider_hit_tp(t)
        return df.iloc[:2]  # sin barra posterior a la señal
    assert trader.resolve_pending(provider) == 0
    assert trader.performance()['available'] is False


def test_unique_per_day_updates(trader):
    trader.record_signal(_signal('DDD', '2024-01-02'))
    trader.record_signal(_signal('DDD', '2024-01-02', entry=101.0))  # mismo dia
    sigs = trader.list_signals()
    assert len(sigs) == 1
    assert sigs[0]['entry'] == 101.0


# ---------------- watchlists ----------------

@pytest.fixture
def store(tmp_path):
    return WatchlistStore(path=str(tmp_path / 'wl.json'))


def test_watchlist_crud(store):
    wl = store.create('tech', ['MSFT', 'AAPL'])
    assert wl['tickers'] == ['AAPL', 'MSFT']
    store.add_tickers('tech', ['nvda'])
    assert store.get('tech')['tickers'] == ['AAPL', 'MSFT', 'NVDA']
    store.remove_tickers('tech', ['AAPL'])
    assert 'AAPL' not in store.get('tech')['tickers']
    assert store.delete('tech') is True
    assert store.get('tech') is None


def test_watchlist_duplicates_and_errors(store):
    store.create('a', ['AAPL'])
    with pytest.raises(ValueError):
        store.create('a')
    with pytest.raises(ValueError):
        store.add_tickers('noexiste', ['AAPL'])
