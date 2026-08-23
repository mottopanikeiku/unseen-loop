"""Unseen Loop: certificate-guided policies for encrypted closed-loop inference."""

from unseen_loop.certificate import ActionCertificate, certify_actions
from unseen_loop.policy import PolynomialPolicy, fit_polynomial_policy
from unseen_loop.specs import CandidateMetrics, PolicySpec, QuantizerSpec

__all__ = [
    "ActionCertificate",
    "CandidateMetrics",
    "PolicySpec",
    "PolynomialPolicy",
    "QuantizerSpec",
    "certify_actions",
    "fit_polynomial_policy",
]

__version__ = "0.1.0"
