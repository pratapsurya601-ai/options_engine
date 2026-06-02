"""
OI-based support / resistance zone detection from option chain.

Logic:
  - Above-spot CALL strike with the highest OI = nearest CALL WALL (resistance)
  - Below-spot PUT strike with the highest OI = nearest PUT WALL (support)
  - Max-pain: strike where total OI×|distance| is maximized (where sellers profit)
  - "Inside zone": spot is between put wall and call wall (typical state)

For NIFTY current-week expiry, the call wall and put wall are usually within
1-2% of spot and act as magnets/repulsors intraday.

Limitations:
  - Single-snapshot OI; for live trading, refresh every 5-15 min
  - Open interest CHANGE is often more informative than absolute OI
  - On expiry day, OI evaporates as positions close — zones become noisy
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OIZones:
    spot: float
    put_wall: int | None        # strike of highest-OI PE below spot
    put_wall_oi: int | None
    call_wall: int | None       # strike of highest-OI CE above spot
    call_wall_oi: int | None
    max_pain: int | None
    zone_width: float           # call_wall - put_wall
    distance_to_put_wall: float | None
    distance_to_call_wall: float | None
    position_in_zone: float | None   # 0=at put wall, 1=at call wall, 0.5=middle

    def in_lower_third(self) -> bool:
        return self.position_in_zone is not None and self.position_in_zone <= 0.33

    def in_upper_third(self) -> bool:
        return self.position_in_zone is not None and self.position_in_zone >= 0.67

    def in_middle(self) -> bool:
        return self.position_in_zone is not None and 0.33 < self.position_in_zone < 0.67


def compute_zones(chain, min_oi: int = 1000) -> OIZones:
    """Find call wall, put wall, max pain from a Chain object."""
    spot = chain.spot
    by_strike = chain.by_strike()
    ce_above: list[tuple[int, int]] = []   # (strike, oi)
    pe_below: list[tuple[int, int]] = []

    for strike, legs in by_strike.items():
        ce = legs.get("CE")
        pe = legs.get("PE")
        if ce and ce.oi and ce.oi >= min_oi and strike > spot:
            ce_above.append((strike, ce.oi))
        if pe and pe.oi and pe.oi >= min_oi and strike < spot:
            pe_below.append((strike, pe.oi))

    call_wall = max(ce_above, key=lambda x: x[1]) if ce_above else (None, None)
    put_wall = max(pe_below, key=lambda x: x[1]) if pe_below else (None, None)

    # Max pain: strike where the sum of |distance| weighted by total OI on the
    # losing-side options is highest. Classic computation: at expiry, max pain
    # is where ITM option intrinsic value paid by writers is minimized.
    # Approximation: argmin over strikes of sum(|K-S|*OI) where ITM
    max_pain_strike = None
    if by_strike:
        min_payout = float("inf")
        for k in by_strike:
            payout = 0.0
            for k2, legs in by_strike.items():
                ce = legs.get("CE"); pe = legs.get("PE")
                if ce and ce.oi and k > k2:
                    payout += (k - k2) * ce.oi
                if pe and pe.oi and k < k2:
                    payout += (k2 - k) * pe.oi
            if payout < min_payout:
                min_payout = payout
                max_pain_strike = k

    z_width = (call_wall[0] - put_wall[0]) if (call_wall[0] and put_wall[0]) else 0.0
    pos = None
    if z_width > 0:
        pos = (spot - put_wall[0]) / z_width
        pos = max(0.0, min(1.0, pos))

    return OIZones(
        spot=spot,
        put_wall=put_wall[0], put_wall_oi=put_wall[1],
        call_wall=call_wall[0], call_wall_oi=call_wall[1],
        max_pain=max_pain_strike,
        zone_width=z_width,
        distance_to_put_wall=(spot - put_wall[0]) if put_wall[0] else None,
        distance_to_call_wall=(call_wall[0] - spot) if call_wall[0] else None,
        position_in_zone=pos,
    )
