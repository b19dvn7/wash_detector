"""Configuration management for wash_detector.

Supports loading thresholds from JSON config file with sensible defaults.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
import json


@dataclass
class MirrorReversalConfig:
    """Thresholds for mirror reversal detection."""
    window_ms: int = 120_000          # Max time between trades (2 min)
    lookback_trades: int = 5          # Check this many previous trades, not just adjacent
    amount_gap_ratio: float = 0.20    # Max allowed difference in trade amounts
    price_diff_bps: float = 8.0       # Max price difference in basis points
    low_volatility_bps: float = 6.0   # Volatility threshold for risk bonus
    base_risk_points: int = 12        # Base risk score per alert
    low_vol_bonus: int = 3            # Extra points when volatility is low
    tight_spread_bonus: int = 2       # Extra points when spread is tight
    spread_threshold_bps: float = 5.0 # Spread threshold for bonus


@dataclass
class LayeringClusterConfig:
    """Thresholds for layering cluster detection."""
    window_ms: int = 60_000           # 60 second window
    min_trades: int = 8               # Minimum trades in window
    price_range_bps: float = 6.0      # Max price range in window
    amount_ratio: float = 0.6         # Max avg amount vs global median
    cooldown_ms: int = 60_000         # Cooldown between alerts
    base_risk_points: int = 9         # Base risk score per alert
    high_imbalance_bonus: int = 2     # Extra points when orderbook imbalanced
    imbalance_threshold: float = 0.3  # Imbalance threshold for bonus


@dataclass
class SpoofingConfig:
    """Thresholds for spoofing detection."""
    window_ms: int = 30_000                      # 30 second window
    min_depth_ratio_change: float = 0.4          # 40% asymmetry swing
    min_orderbook_pressure: float = 0.5          # Minimum pressure for directional spoofing
    spread_compression_threshold: float = 0.5    # 50% compression spike
    cooldown_ms: int = 30_000                    # Cooldown between alerts
    base_risk_points: int = 12                   # Base risk score per alert
    compression_bonus: int = 3                   # Extra points for spread compression


@dataclass
class QuoteStuffingConfig:
    """Thresholds for quote stuffing detection."""
    window_ms: int = 5_000                       # 5 second window
    min_trade_intensity: float = 100.0           # Minimum trades per second (peak normal ~166)
    max_realized_volatility: float = 0.0002      # Maximum price movement (stuffing = noise)
    min_liquidity_drop: float = 0.3              # 30% liquidity quality drop
    cooldown_ms: int = 10_000                    # Cooldown between alerts
    base_risk_points: int = 10                   # Base risk score per alert
    low_volatility_bonus: int = 2                # Extra points for very low volatility


@dataclass
class BalancedChurnConfig:
    """Thresholds for balanced churn detection."""
    window_ms: int = 300_000          # 5 minute window
    min_trades: int = 20              # Minimum trades in window
    balance_ratio: float = 0.20       # Max side imbalance ratio
    price_move_bps: float = 10.0      # Max price movement in window
    notional_multiplier: float = 20.0 # Min window notional vs median
    min_notional_usd: float = 0.0     # Min absolute window notional (0 = disabled)
    cooldown_ms: int = 300_000        # Cooldown between alerts
    base_risk_points: int = 14        # Base risk score per alert


@dataclass
class RiskScoringConfig:
    """Risk score calculation weights and thresholds."""
    mirror_rate_weight: float = 0.60
    mirror_rate_cap: float = 45.0
    layering_rate_weight: float = 6.00
    layering_rate_cap: float = 20.0
    churn_rate_weight: float = 8.00
    churn_rate_cap: float = 35.0
    # Escalation thresholds
    escalation_tier1_points: int = 2_000
    escalation_tier1_bonus: float = 5.0
    escalation_tier2_points: int = 10_000
    escalation_tier2_bonus: float = 10.0
    escalation_tier3_points: int = 50_000
    escalation_tier3_bonus: float = 15.0


@dataclass
class DetectorConfig:
    """Complete detector configuration."""
    mirror_reversal: MirrorReversalConfig = field(default_factory=MirrorReversalConfig)
    layering_cluster: LayeringClusterConfig = field(default_factory=LayeringClusterConfig)
    spoofing: SpoofingConfig = field(default_factory=SpoofingConfig)
    quote_stuffing: QuoteStuffingConfig = field(default_factory=QuoteStuffingConfig)
    balanced_churn: BalancedChurnConfig = field(default_factory=BalancedChurnConfig)
    risk_scoring: RiskScoringConfig = field(default_factory=RiskScoringConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DetectorConfig":
        """Load config from dictionary, using defaults for missing values."""
        mirror_data = data.get("mirror_reversal", {})
        layering_data = data.get("layering_cluster", {})
        spoofing_data = data.get("spoofing", {})
        quote_stuffing_data = data.get("quote_stuffing", {})
        churn_data = data.get("balanced_churn", {})
        scoring_data = data.get("risk_scoring", {})

        return cls(
            mirror_reversal=MirrorReversalConfig(**{
                k: v for k, v in mirror_data.items()
                if k in MirrorReversalConfig.__dataclass_fields__
            }),
            layering_cluster=LayeringClusterConfig(**{
                k: v for k, v in layering_data.items()
                if k in LayeringClusterConfig.__dataclass_fields__
            }),
            spoofing=SpoofingConfig(**{
                k: v for k, v in spoofing_data.items()
                if k in SpoofingConfig.__dataclass_fields__
            }),
            quote_stuffing=QuoteStuffingConfig(**{
                k: v for k, v in quote_stuffing_data.items()
                if k in QuoteStuffingConfig.__dataclass_fields__
            }),
            balanced_churn=BalancedChurnConfig(**{
                k: v for k, v in churn_data.items()
                if k in BalancedChurnConfig.__dataclass_fields__
            }),
            risk_scoring=RiskScoringConfig(**{
                k: v for k, v in scoring_data.items()
                if k in RiskScoringConfig.__dataclass_fields__
            }),
        )

    @classmethod
    def load(cls, config_path: Optional[str | Path] = None) -> "DetectorConfig":
        """Load config from JSON file, or return defaults if no path given."""
        if config_path is None:
            return cls()

        path = Path(config_path)
        if not path.exists():
            return cls()

        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        """Export config as dictionary."""
        from dataclasses import asdict
        return asdict(self)

    def save(self, config_path: str | Path) -> None:
        """Save config to JSON file."""
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


# Default global config instance
_default_config = DetectorConfig()


def get_default_config() -> DetectorConfig:
    """Get the default configuration."""
    return _default_config
