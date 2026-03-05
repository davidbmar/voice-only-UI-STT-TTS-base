"""Fetch ephemeral TURN credentials from Twilio NTS.

Ported from engine-repo/gateway/turn.py for use in the scheduling app.
The Twilio Network Traversal Service provides short-lived TURN/STUN
credentials so browser clients can traverse NATs and firewalls.

Environment variables:
  TWILIO_ACCOUNT_SID  — Twilio account SID
  TWILIO_AUTH_TOKEN    — Twilio auth token

Falls back to an empty list if credentials are not configured, allowing
the app to run without Twilio (STUN-only fallback is set in config).
"""

import logging
import os

import aiohttp

log = logging.getLogger("turn")


async def fetch_twilio_turn_credentials() -> list:
    """Call Twilio's Network Traversal Service to get temporary TURN/STUN creds.

    Returns a list of ICE server dicts in the format::

        [
            {"urls": "stun:global.stun.twilio.com:3478"},
            {"urls": "turn:global.turn.twilio.com:3478?transport=udp",
             "username": "...", "credential": "..."},
            ...
        ]

    Returns empty list if Twilio is not configured or the request fails.
    """
    account_sid = os.getenv("TWILIO_ACCOUNT_SID", "")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN", "")

    if not account_sid or not auth_token:
        log.warning("TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN not set — no TURN servers")
        return []

    url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Tokens.json"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                auth=aiohttp.BasicAuth(account_sid, auth_token),
            ) as resp:
                if resp.status != 201:
                    body = await resp.text()
                    log.error("Twilio token request failed (%d): %s", resp.status, body)
                    return []

                data = await resp.json()

        ice_servers = data.get("ice_servers", [])
        log.info(
            "Got %d ICE servers from Twilio (TTL: %ss)",
            len(ice_servers),
            data.get("ttl", "?"),
        )
        return ice_servers

    except Exception as e:
        log.error("Failed to fetch Twilio TURN credentials: %s", e)
        return []
