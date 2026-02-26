"""Voice Frontend Edge Auth — pluggable authentication providers."""

from edge_auth.base import AuthProvider, AuthResult
from edge_auth.middleware import auth_middleware, auth_dependency
from edge_auth.composite import CompositeProvider

__all__ = [
    "AuthProvider",
    "AuthResult",
    "auth_middleware",
    "auth_dependency",
    "CompositeProvider",
]
