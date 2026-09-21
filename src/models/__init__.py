"""V6 local model stack: registry + resource-aware manager + adapters.

Models live under models/<task>/ after explicit acquisition
(scripts/acquire_models.py). Runtime is offline-capable: with
FLYBRAIN_OFFLINE=1 every download/refetch raises instead of touching
the network.
"""
