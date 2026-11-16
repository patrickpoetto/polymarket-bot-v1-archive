from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChainlinkPricePoint:
    timestamp_unix: int
    price: float


class ChainlinkDataStreamsClient:
    """
    Thin wrapper around Chainlink Data Streams SDK.

    Notes:
    - We intentionally keep SDK integration localized here.
    - If SDK package name/API changes, only this file should be updated.
    """

    def __init__(self, api_base_url: str, api_key: str, api_secret: str, feed_id: str):
        self.api_base_url = api_base_url
        self.api_key = api_key
        self.api_secret = api_secret
        self.feed_id = feed_id
        self._client = self._build_sdk_client()

    def _build_sdk_client(self):
        """
        Build SDK client lazily.
        Replace this implementation with the exact official SDK object once installed.
        """
        try:
            # Placeholder import name; adjust to the official Python SDK package.
            # Keeping this dynamic avoids breaking the rest of the project.
            import chainlink_data_streams_sdk as ds_sdk  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Chainlink Data Streams SDK not installed. "
                "Install the official SDK and update bot4_chainlink/chainlink_client.py import."
            ) from exc

        # Placeholder constructor call; adapt to official SDK.
        return ds_sdk.Client(
            base_url=self.api_base_url,
            api_key=self.api_key,
            api_secret=self.api_secret,
        )

    def get_price_at_or_before(self, timestamp_unix: int) -> ChainlinkPricePoint:
        """
        Return latest available report at or before timestamp.

        This method must be mapped to the SDK historical reports API.
        """
        # Placeholder SDK call; adapt to official SDK response format.
        data = self._client.get_report_at_or_before(  # type: ignore[attr-defined]
            feed_id=self.feed_id,
            timestamp=timestamp_unix,
        )
        return ChainlinkPricePoint(
            timestamp_unix=int(data["timestamp"]),
            price=float(data["price"]),
        )
