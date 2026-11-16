"""Session state for paired MM."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from bot2_btc.models import MarketInfo, Side
from bot6_mm.models import ActiveOrder, SideInventory


@dataclass
class PairArmState:
    started_at: float = 0.0
    up_order: Optional[ActiveOrder] = None
    dn_order: Optional[ActiveOrder] = None
    up_filled: bool = False
    dn_filled: bool = False


@dataclass
class PairedSession:
    market_info: MarketInfo
    market_index: int
    started_at: float = field(default_factory=time.time)

    up_inv: SideInventory = field(default_factory=lambda: SideInventory(Side.UP))
    dn_inv: SideInventory = field(default_factory=lambda: SideInventory(Side.DOWN))

    arm: Optional[PairArmState] = None
    quoting_stopped: bool = False
    session_closed: bool = False
    last_requote_at: float = 0.0
    last_trigger_at: float = 0.0
    n_triggers: int = 0
    pairs_opened: int = 0
    osl_blocks: int = 0
    orphan_sellbacks: int = 0
    rebalances: int = 0

    def paired_shares(self) -> float:
        return min(self.up_inv.shares, self.dn_inv.shares)
