"""Tests for qf.broker.OKXSpotBroker -- request shape and fail-closed
response parsing, against fake python-okx clients injected in place of
the lazily-built real ones. No network, no OKX account.

What matters most here is that each order goes out with exactly the
fields OKX needs to do the right thing:
  * market BUY sized in quote (`tgtCcy=quote_ccy`) -- the fixed-dollar lane;
  * market SELL sized in base (`tgtCcy=base_ccy`) -- the exact recorded size;
  * one OCO carrying both legs as market-on-trigger (`-1`) with the tag.

Run with: pytest qf/tests -m unit
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from qf.broker import BrokerError, OKXSpotBroker, _fmt

pytestmark = pytest.mark.unit

OK = {"code": "0", "msg": ""}


class FakeApi:
    """Records every call; returns the canned response for that method."""

    def __init__(self, **responses):
        self.responses = responses
        self.calls: list[tuple[str, tuple, dict]] = []

    def __getattr__(self, name):
        def call(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return self.responses[name]

        return call


def _broker(trade=None, account=None, public=None) -> OKXSpotBroker:
    b = OKXSpotBroker(SimpleNamespace(is_demo=True), credentials_source="test")
    b._trade, b._account, b._public = trade, account, public
    return b


def test_market_buy_is_sized_in_quote_and_tagged():
    api = FakeApi(place_order={**OK, "data": [{"ordId": "111", "sCode": "0"}]})
    assert _broker(trade=api).market_buy_usd("SOL-USDT", 1000.0, "qfSOL1e") == "111"
    _, _, kw = api.calls[0]
    assert kw == {
        "instId": "SOL-USDT", "tdMode": "cash", "side": "buy", "ordType": "market",
        "sz": "1000", "tgtCcy": "quote_ccy", "clOrdId": "qfSOL1e",
    }


def test_market_sell_is_sized_in_base():
    api = FakeApi(place_order={**OK, "data": [{"ordId": "222", "sCode": "0"}]})
    _broker(trade=api).market_sell("SOL-USDT", 9.965, "qfSOL1t")
    kw = api.calls[0][2]
    assert (kw["side"], kw["sz"], kw["tgtCcy"]) == ("sell", "9.965", "base_ccy")


def test_order_rejection_surfaces_okx_reason():
    api = FakeApi(place_order={"code": "1", "msg": "All operations failed",
                               "data": [{"sCode": "51008", "sMsg": "Insufficient balance"}]})
    with pytest.raises(BrokerError, match="Insufficient balance.*51008"):
        _broker(trade=api).market_sell("SOL-USDT", 1.0, "x")


def test_row_level_rejection_fails_even_with_ok_envelope():
    api = FakeApi(place_order={**OK, "data": [{"sCode": "51155", "sMsg": "restricted"}]})
    with pytest.raises(BrokerError, match="restricted"):
        _broker(trade=api).market_buy_usd("XRP-USDT", 1000.0, "x")


def test_oco_carries_both_legs_as_market_on_trigger():
    api = FakeApi(place_algo_order={**OK, "data": [{"algoId": "A1", "sCode": "0"}]})
    assert _broker(trade=api).place_oco("SOL-USDT", 9.965, 110.0, 95.0, "qfSOL1p") == "A1"
    kw = api.calls[0][2]
    assert kw["ordType"] == "oco" and kw["side"] == "sell" and kw["tdMode"] == "cash"
    assert (kw["tpTriggerPx"], kw["tpOrdPx"], kw["slTriggerPx"], kw["slOrdPx"]) == ("110", "-1", "95", "-1")
    assert kw["algoClOrdId"] == "qfSOL1p"


def test_amend_stop_only_touches_the_stop_leg():
    api = FakeApi(amend_algo_order={**OK, "data": [{"algoId": "A1", "sCode": "0"}]})
    _broker(trade=api).amend_stop("SOL-USDT", "A1", 100.71)
    kw = api.calls[0][2]
    assert kw == {"instId": "SOL-USDT", "algoId": "A1", "newSlTriggerPx": "100.71",
                  "newSlOrdPx": "-1", "newSlTriggerPxType": "last"}


def test_cancel_algo_sends_inst_and_id():
    api = FakeApi(cancel_algo_order={**OK, "data": [{"algoId": "A1", "sCode": "0"}]})
    _broker(trade=api).cancel_algo("SOL-USDT", "A1")
    assert api.calls[0][1] == ([{"instId": "SOL-USDT", "algoId": "A1"}],)


def test_order_fill_reports_fee_as_positive_charge():
    api = FakeApi(get_order={**OK, "data": [{"state": "filled", "avgPx": "100.5", "accFillSz": "9.95",
                                             "fee": "-0.034825", "feeCcy": "SOL"}]})
    fill = _broker(trade=api).order_fill("SOL-USDT", "111")
    assert (fill.state, fill.avg_px, fill.filled_sz, fill.fee, fill.fee_ccy) == ("filled", 100.5, 9.95, 0.034825, "SOL")
    assert fill.done


def test_order_fill_tolerates_blank_numbers_on_a_live_order():
    api = FakeApi(get_order={**OK, "data": [{"state": "live", "avgPx": "", "accFillSz": "0", "fee": "", "feeCcy": ""}]})
    fill = _broker(trade=api).order_fill("SOL-USDT", "111")
    assert (fill.avg_px, fill.filled_sz, fill.done) == (0.0, 0.0, False)


@pytest.mark.parametrize(
    "row, expected",
    [
        ({"state": "live"}, ("live", None, None, True, False)),
        ({"state": "effective", "ordId": "9", "actualSide": "tp"}, ("effective", "9", "tp", False, True)),
        ({"state": "effective", "ordId": "", "ordIdList": ["7"], "actualSide": "sl"}, ("effective", "7", "sl", False, True)),
    ],
)
def test_algo_status_parsing(row, expected):
    api = FakeApi(get_algo_order_details={**OK, "data": [row]})
    st = _broker(trade=api).algo_status("A1")
    assert (st.state, st.ord_id, st.side, st.live, st.triggered) == expected


def test_algo_failed_and_cancelled_states():
    for state, attr in (("order_failed", "failed"), ("canceled", "cancelled")):
        api = FakeApi(get_algo_order_details={**OK, "data": [{"state": state}]})
        assert getattr(_broker(trade=api).algo_status("A1"), attr)


def test_holding_reads_cash_balance_of_the_right_currency():
    api = FakeApi(get_account_balance={**OK, "data": [{"details": [
        {"ccy": "USDT", "cashBal": "5000"}, {"ccy": "SOL", "cashBal": "9.965", "availBal": "0"},
    ]}]})
    assert _broker(account=api).holding("SOL") == 9.965
    assert _broker(account=api).holding("BTC") == 0.0


def test_taker_rate_is_a_positive_rate():
    api = FakeApi(get_fee_rates={**OK, "data": [{"maker": "-0.002", "taker": "-0.0035"}]})
    assert _broker(account=api).taker_rate("SOL-USDT") == 0.0035


def test_instrument_is_cached_and_must_be_live():
    api = FakeApi(get_instruments={**OK, "data": [{"lotSz": "0.000001", "minSz": "0.001", "tickSz": "0.01", "state": "live"}]})
    b = _broker(public=api)
    inst = b.instrument("SOL-USDT")
    b.instrument("SOL-USDT")
    assert (inst.lot_sz, inst.min_sz, inst.tick_sz) == (0.000001, 0.001, 0.01)
    assert len(api.calls) == 1

    suspended = FakeApi(get_instruments={**OK, "data": [{"lotSz": "1", "minSz": "1", "tickSz": "1", "state": "suspend"}]})
    with pytest.raises(BrokerError, match="not live"):
        _broker(public=suspended).instrument("XYZ-USDT")


def test_tagged_live_algos_filters_by_prefix():
    api = FakeApi(order_algos_list={**OK, "data": [
        {"instId": "SOL-USDT", "algoId": "A1", "algoClOrdId": "qfSOL1p"},
        {"instId": "BTC-USDT", "algoId": "A2", "algoClOrdId": "vtBTC"},
        {"instId": "ETH-USDT", "algoId": "A3", "algoClOrdId": ""},
    ]})
    assert _broker(trade=api).tagged_live_algos("qf") == [("SOL-USDT", "A1")]


def test_envelope_error_on_reads_raises():
    api = FakeApi(get_algo_order_details={"code": "51603", "msg": "Order does not exist", "data": []})
    with pytest.raises(BrokerError, match="does not exist"):
        _broker(trade=api).algo_status("nope")


@pytest.mark.parametrize("value, text", [(1e-06, "0.000001"), (100.0, "100"), (110.0, "110"), (9.965, "9.965")])
def test_fmt_is_plain_decimal(value, text):
    assert _fmt(value) == text


def test_cancel_order_sends_inst_and_order_id():
    api = FakeApi(cancel_order={**OK, "data": [{"ordId": "111", "sCode": "0"}]})
    _broker(trade=api).cancel_order("SOL-USDT", "111")
    assert api.calls[0][2] == {"instId": "SOL-USDT", "ordId": "111"}


def test_ok_envelope_with_no_rows_is_a_clean_broker_error():
    api = FakeApi(place_order={**OK, "data": []})
    with pytest.raises(BrokerError, match="market_buy"):
        _broker(trade=api).market_buy_usd("SOL-USDT", 1000.0, "x")


def test_no_pending_algos_is_an_empty_list_not_an_error():
    # found live 2026-09-25: OKX answers code "0" with data [] when nothing is pending
    api = FakeApi(order_algos_list={**OK, "data": []})
    assert _broker(trade=api).tagged_live_algos("qf") == []
