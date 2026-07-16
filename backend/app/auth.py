import os
from fastapi import HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security_scheme = HTTPBearer(auto_error=False)

def verify_admin(credentials: HTTPAuthorizationCredentials = Security(security_scheme)):
    """
    Validates that the incoming request contains a valid Admin token 
    in the Authorization header (e.g., 'Bearer your-admin-token').
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication credentials.",
        )
    
    expected_token = os.environ.get("ADMIN_API_KEY", "admin-debug-secret-key")
    
    if credentials.credentials != expected_token:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied. Admin permissions required.",
        )
    return credentials.credentials