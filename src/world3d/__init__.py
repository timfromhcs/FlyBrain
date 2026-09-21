"""V6 embodied world: persistent 3D environment + real physics + embodiment.

Authoritative simulation is server-side (MuJoCo). The browser renders state;
it never simulates. TRUE_WORLD vs ORGANISM_KNOWLEDGE is strictly separated:
cognition receives only sensor observations + memories, never hidden state.
"""
