import os 
from langgraph_sdk import Auth

auth = Auth()

@auth.authenticate
async def authenticate(headers: dict) -> Auth.types.MinimalUserDict:

    expected_key = os.environ.get("LANGGRAPH_SECRET")
    if not expected_key:
        return {"identity": "local_dev"}

    auth_header = headers.get(b"authorization")
    
    if not auth_header:
        raise Auth.exceptions.HTTPException(
            status_code=401, 
            detail="Missing Authorization header"
        )

    try:
        scheme, token = auth_header.decode("utf-8").split()
        if scheme.lower() != "bearer" or token != expected_key:
            raise Auth.exceptions.HTTPException(
                status_code=401, 
                detail="Unauthorized: Invalid API Key"
            )
    except ValueError:
        raise Auth.exceptions.HTTPException(
            status_code=401, 
            detail="Invalid Authorization header format"
        )
    return {"identity": "basecase_nextjs_backend"}
