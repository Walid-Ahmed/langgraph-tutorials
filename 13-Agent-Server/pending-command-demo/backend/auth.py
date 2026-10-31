# Who is calling? A stand-in for your real login check.
#
# Option B in Chapter 16 puts your own backend between the browser and the
# graph. Its first job is to turn a request into a user id. This file does
# that with two fixed demo tokens, so the example runs with no login system.
#
# Before you run this:
#    python -m pip install -e ".[backend]"
#    from 13-Agent-Server/pending-command-demo/
#
# It is not run on its own. backend/routes.py depends on current_user, and
# uvicorn backend.app:app serves both (see backend/app.py).
#
# Expected: a request with "Authorization: Bearer token-amira" is user
# amira. A missing or unknown token is rejected before the graph runs.

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

# Demo tokens only. A real service checks a session cookie or a signed token
# issued by its own login system, and never ships a table like this.
DEMO_TOKENS = {
    "token-amira": "amira",
    "token-omar": "omar",
}

bearer = HTTPBearer()


def current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer),
) -> str:
    """Return the user id behind the bearer token, or reject the request."""
    user_id = DEMO_TOKENS.get(credentials.credentials)
    if user_id is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Unknown token")
    return user_id
