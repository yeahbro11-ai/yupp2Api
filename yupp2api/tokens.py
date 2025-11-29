"""Token rotation and Yupp account management."""

from __future__ import annotations

import time
from typing import Optional

from .config import Settings
from .models import YuppAccount
from .state import RuntimeState
from .utils import create_requests_session, log_debug


def initialize_accounts(state: RuntimeState, settings: Settings) -> None:
    """Initialize Yupp accounts from configured tokens."""
    state.accounts.clear()
    for token in settings.yupp_tokens:
        state.accounts.append(
            {
                "token": token,
                "is_valid": True,
                "last_used": 0.0,
                "error_count": 0,
            }
        )
    print(f"Successfully loaded {len(state.accounts)} Yupp accounts from configuration.")


def get_best_yupp_account(state: RuntimeState, settings: Settings) -> Optional[YuppAccount]:
    """
    Get the best available Yupp account using a smart selection algorithm.

    Selects valid accounts with error_count < max_error_count or
    accounts whose cooldown period has expired.
    """
    with state.account_rotation_lock:
        now = time.time()
        valid_accounts = [
            acc
            for acc in state.accounts
            if acc["is_valid"]
            and (
                acc["error_count"] < settings.max_error_count
                or now - acc["last_used"] > settings.error_cooldown
            )
        ]

        if not valid_accounts:
            return None

        for acc in valid_accounts:
            if (
                acc["error_count"] >= settings.max_error_count
                and now - acc["last_used"] > settings.error_cooldown
            ):
                acc["error_count"] = 0

        valid_accounts.sort(key=lambda x: (x["last_used"], x["error_count"]))
        account = valid_accounts[0]
        account["last_used"] = now
        return account


def claim_yupp_reward(account: YuppAccount, reward_id: str, settings: Settings) -> Optional[int]:
    """
    Claim a Yupp reward synchronously.

    Returns the new credit balance if successful, None otherwise.
    """
    try:
        log_debug(settings, f"Claiming reward {reward_id}...")
        url = "https://yupp.ai/api/trpc/reward.claim?batch=1"
        payload = {"0": {"json": {"rewardId": reward_id}}}
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36 Edg/137.0.0.0",
            "Content-Type": "application/json",
            "sec-fetch-site": "same-origin",
            "Cookie": f"__Secure-yupp.session-token={account['token']}",
        }
        session = create_requests_session(settings)
        response = session.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        balance = data[0]["result"]["data"]["json"]["currentCreditBalance"]
        print(f"Reward claimed successfully. New balance: {balance}")
        return balance
    except Exception as e:
        print(f"Failed to claim reward {reward_id}. Error: {e}")
        return None
