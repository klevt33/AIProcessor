"""
## Overview
This module provides middleware for FastAPI applications, including:
1. **LoggingMiddleware**: Logs incoming requests, their payloads, and response times.
2. **JWTAuthMiddleware**: Validates JSON Web Tokens (JWT) for authentication and authorization.

The middleware enhances observability and security for FastAPI applications
by logging request details and enforcing token-based authentication.

"""

import json
import time
from typing import Awaitable, Callable

import jwt
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response

from api_schemas import Responses
from cdb_utils import CDB_Utils
from config import Config
from logger import logger, request_id_var
from utils import validate_token


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to log details of incoming HTTP requests and their responses.

    Logs the request method, URL, body, and processing time. Extracts the request ID
    from the request body (if available) and stores it in a context variable for tracking.

    Attributes:
        None
    """

    async def dispatch(self, request: Request, call_next):
        """
        Intercepts HTTP requests to log their details before and after processing.

        Args:
            request (Request): The incoming HTTP request.
            call_next (Callable): The next middleware or endpoint to call.

        Returns:
            Response: The HTTP response after processing the request.
        """
        # Record the start time for calculating processing time
        start_time = time.time()

        # Attempt to read the request body
        request_body = await request.body()
        try:
            # Parse the request body as JSON
            request_json = json.loads(request_body.decode("utf-8"))
        except json.JSONDecodeError:
            # Handle cases where the request body is not valid JSON
            request_json = None

        try:
            # Extract request ID from the request JSON (if available)
            request_id = request_json.get("id") if isinstance(request_json, dict) else None if request_json is not None else None

            if request_id:
                request_id_var.set(request_id)  # Store the request ID in the context variable
        except Exception as e:
            # Log any errors encountered while extracting the request ID
            logger.error(str(e))

        # Log the incoming request details
        logger.debug("=" * 80)
        logger.info(f"Incoming Request: {request.method} {request.url}")
        # logger.info(f"Headers: {dict(request.headers)}")
        if request_json:
            logger.info(f"Request Body: {json.dumps(request_json, indent=2)}")

        # Process the request and record the response
        response = await call_next(request)

        # Calculate the time taken to process the request
        process_time = time.time() - start_time

        logger.info(f"{request.method} {request.url} completed in {process_time:.2f}s with status {response.status_code}")
        return response


class JWTAuthMiddleware(BaseHTTPMiddleware):
    """
    Middleware to validate JSON Web Tokens (JWT) for authentication and authorization.

    Ensures that incoming requests include a valid `Bearer` token in the `Authorization` header.
    Validates the token and checks its payload for required attributes.

    Attributes:
        config (Config): Configuration object containing token validation settings.
    """

    def __init__(self, app, config: Config):
        """
        Initializes the middleware with the application and configuration.

        Args:
            app: The FastAPI application instance.
            config (Config): Configuration object containing token validation settings.
        """
        super().__init__(app)
        self.config = config  # Store config object

    async def dispatch(self, request: Request, call_next):
        """
        Intercepts HTTP requests to validate their JWT tokens.

        Args:
            request (Request): The incoming HTTP request.
            call_next (Callable): The next middleware or endpoint to call.

        Returns:
            Response: The HTTP response after processing the request.

        Raises:
            HTTPException: If the token is missing or invalid.
        """
        # Extract the Authorization header from the request
        authorization: str = request.headers.get("Authorization")
        if not authorization or not authorization.startswith("Bearer "):
            # Raise an exception if the Authorization header is missing or invalid
            # raise HTTPException(status_code=401, detail="Missing or invalid token")
            logger.error(f"Missing or invalid header token. URL:{str(request.url)} Body:{str(request.body)}")
            return JSONResponse({"error": "Missing or invalid header token"}, status_code=401)

        # Extract the token from the Authorization header
        token = authorization.split("Bearer ")[1]

        try:
            # Validate the token using the utility function
            payload = validate_token(self.config, token)
            # logger.debug(f"Token is valid. Payload: {payload}")
            logger.debug(f"Token is valid. Payload-oid: {payload['oid']}")

            # Extract the Object ID (OID) from the token payload
            payload.get("oid")

            # Uncomment the following lines to enforce OID-based authorization
            # if not oid or oid not in self.config.allowed_clients.get(self.config.environment, []):
            #     return JSONResponse({"error": "Unauthorized access"}, status_code=403)

        except jwt.ExpiredSignatureError:
            # Handle expired tokens
            return JSONResponse({"error": "Token expired"}, status_code=401)
        except jwt.InvalidTokenError as e:
            # Handle invalid tokens
            return JSONResponse({"error": f"Invalid token: {str(e)}"}, status_code=401)

        # Proceed to the next middleware or endpoint
        return await call_next(request)


class CosmosLoggingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, cdb_utils: CDB_Utils):
        """
        Initializes the middleware with the application and configuration.

        Args:
            app: The FastAPI application instance.
            cdb_utils (CDB_Utils): CDB_Utils object to update cosmos api container.
        """
        super().__init__(app)
        self.cdb_utils = cdb_utils

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        """
        Intercepts HTTP requests to log to Cosmos DB

        Args:
            request (Request): The incoming HTTP request.
            call_next (Callable): The next middleware or endpoint to call.

        Returns:
            Response: The HTTP response after processing the request.
        """
        # Get the request path
        path = request.url.path
        uuid = None

        # Read the body safely
        try:
            body_dict = await request.json()
        except Exception:
            body_dict = None

        try:
            _, uuid = await self.cdb_utils.add_api_request_in_cosmos(request_body=body_dict, path=path)

        except Exception as e:
            logger.error(f"Error logging api request to CDB. {str(e)}", exc_info=True)
            return JSONResponse(
                Responses.ErrorResponse(
                    status="failed", message="Unable to add API request to Cosmos.", error_detail=str(e)
                ).__dict__,
                status_code=500,
            )

        request.state.api_request_uuid = uuid

        # Continue processing the request
        return await call_next(request)
