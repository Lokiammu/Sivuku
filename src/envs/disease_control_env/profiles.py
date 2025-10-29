DISEASE_PROFILES = {
    "covid":   dict(mu0=14.0, sigma=0.025, delta=0.006, nu=0.0015, phi=0.12),
    "flu":     dict(mu0=12.0, sigma=0.015, delta=0.004, nu=0.0002, phi=0.18),
    "measles": dict(mu0=18.0, sigma=0.070, delta=0.000, nu=0.0001, phi=0.08),
}

MUTATION_VARIANTS = [
    dict(name="high_transmissibility", mu0=1.10, sigma=1.35, delta=1.10, nu=1.00, phi=0.90),
    dict(name="immune_escape",         mu0=1.15, sigma=1.25, delta=1.20, nu=1.05, phi=0.85),
    dict(name="deadlier",              mu0=1.00, sigma=1.05, delta=1.00, nu=1.50, phi=0.95),
]
