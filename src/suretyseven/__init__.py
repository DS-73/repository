"""SuretySeven underwriting take-home service.

A small, production-minded service that accepts surety bond applications,
enriches them from an external Applicant API, scores them with a configurable
rule set, produces an underwriting decision and reliably notifies a downstream
system exactly-once-ish (at-least-once delivery + consumer-side de-duplication).
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
