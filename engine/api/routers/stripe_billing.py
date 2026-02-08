"""Stripe billing integration — checkout sessions + webhooks.

POST /billing/checkout   — Create a Stripe Checkout session (requires auth)
POST /billing/webhook    — Stripe webhook handler (no auth — signature verified)
GET  /billing/portal     — Create a Stripe Customer Portal session (requires auth)
"""

import logging
from typing import Optional

import stripe
from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from pydantic import BaseModel

from api.dependencies import require_auth
from engine.config.settings import settings
from engine.db.session import async_session_factory
from engine.db import eval_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/billing", tags=["billing"])


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class CheckoutRequest(BaseModel):
    price_id: Optional[str] = None  # Override STRIPE_PRO_PRICE_ID if needed


class CheckoutResponse(BaseModel):
    checkout_url: str


class PortalResponse(BaseModel):
    portal_url: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stripe_client() -> None:
    """Configure the stripe module with the API key."""
    if not settings.stripe_secret_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Stripe is not configured.",
        )
    stripe.api_key = settings.stripe_secret_key


async def _get_or_create_customer(tenant_id: str) -> str:
    """Return existing Stripe customer ID, or create one."""
    async with async_session_factory() as session:
        tier_row = await eval_service.get_tenant_tier(session, tenant_id)
        if tier_row and tier_row.stripe_customer_id:
            return tier_row.stripe_customer_id

    # Create customer in Stripe
    customer = stripe.Customer.create(
        email=tenant_id,  # tenant_id is the user's email (from JWT sub)
        metadata={"tenant_id": tenant_id},
    )

    # Persist the mapping
    async with async_session_factory() as session:
        await eval_service.upsert_tenant_tier(
            session,
            tenant_id=tenant_id,
            tier="community",
            stripe_customer_id=customer.id,
        )
        await session.commit()

    return customer.id


# ---------------------------------------------------------------------------
# POST /billing/checkout
# ---------------------------------------------------------------------------


@router.post("/checkout", response_model=CheckoutResponse)
async def create_checkout_session(
    body: CheckoutRequest = CheckoutRequest(),
    actor: str = Depends(require_auth),
):
    """Create a Stripe Checkout session for Pro upgrade."""
    _stripe_client()
    price_id = body.price_id or settings.stripe_pro_price_id
    if not price_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No price_id specified and STRIPE_PRO_PRICE_ID not configured.",
        )

    customer_id = await _get_or_create_customer(actor)

    session = stripe.checkout.Session.create(
        customer=customer_id,
        mode="subscription",
        line_items=[{"price": price_id, "quantity": 1}],
        success_url=settings.stripe_success_url,
        cancel_url=settings.stripe_cancel_url,
        subscription_data={
            "metadata": {"tenant_id": actor},
        },
        metadata={"tenant_id": actor},
    )

    return CheckoutResponse(checkout_url=session.url)


# ---------------------------------------------------------------------------
# GET /billing/portal
# ---------------------------------------------------------------------------


@router.get("/portal", response_model=PortalResponse)
async def create_portal_session(
    actor: str = Depends(require_auth),
):
    """Create a Stripe Customer Portal session for subscription management."""
    _stripe_client()
    customer_id = await _get_or_create_customer(actor)

    portal = stripe.billing_portal.Session.create(
        customer=customer_id,
        return_url=settings.stripe_success_url,
    )

    return PortalResponse(portal_url=portal.url)


# ---------------------------------------------------------------------------
# POST /billing/webhook
# ---------------------------------------------------------------------------

# Tier mapping from Stripe product metadata or price lookup
_TIER_BY_EVENT = {
    "customer.subscription.created": "pro",
    "customer.subscription.updated": "pro",  # may check status
    "customer.subscription.deleted": "community",
}


@router.post("/webhook")
async def stripe_webhook(request: Request):
    """Handle Stripe webhook events.

    Validates the signature, then updates tenant_tiers based on
    subscription lifecycle events.
    """
    if not settings.stripe_secret_key:
        raise HTTPException(status_code=503, detail="Stripe not configured")

    stripe.api_key = settings.stripe_secret_key
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature", "")

    if settings.stripe_webhook_secret:
        try:
            event = stripe.Webhook.construct_event(
                payload, sig_header, settings.stripe_webhook_secret
            )
        except stripe.error.SignatureVerificationError:
            logger.warning("Stripe webhook signature verification failed")
            raise HTTPException(status_code=400, detail="Invalid signature")
        except Exception as e:
            logger.error("Stripe webhook error: %s", e)
            raise HTTPException(status_code=400, detail=str(e))
    else:
        # No webhook secret configured — parse without verification (dev only)
        import json
        event = stripe.Event.construct_from(json.loads(payload), stripe.api_key)
        logger.warning("Stripe webhook received WITHOUT signature verification")

    event_type = event["type"]
    logger.info("Stripe webhook: %s", event_type)

    if event_type in (
        "customer.subscription.created",
        "customer.subscription.updated",
        "customer.subscription.deleted",
    ):
        subscription = event["data"]["object"]
        customer_id = subscription.get("customer")
        sub_id = subscription.get("id")
        sub_status = subscription.get("status")

        # Determine tier from event
        if event_type == "customer.subscription.deleted" or sub_status in (
            "canceled", "unpaid", "past_due",
        ):
            new_tier = "community"
        else:
            new_tier = "pro"

        # Find tenant by Stripe customer ID
        async with async_session_factory() as session:
            tier_row = await eval_service.get_tenant_tier_by_stripe_customer(
                session, customer_id
            )
            if tier_row:
                await eval_service.upsert_tenant_tier(
                    session,
                    tenant_id=tier_row.tenant_id,
                    tier=new_tier,
                    stripe_customer_id=customer_id,
                    stripe_subscription_id=sub_id,
                )
                await session.commit()
                logger.info(
                    "Updated tenant %s to tier=%s (sub=%s, status=%s)",
                    tier_row.tenant_id, new_tier, sub_id, sub_status,
                )
            else:
                # Try metadata fallback
                tenant_id = subscription.get("metadata", {}).get("tenant_id")
                if tenant_id:
                    await eval_service.upsert_tenant_tier(
                        session,
                        tenant_id=tenant_id,
                        tier=new_tier,
                        stripe_customer_id=customer_id,
                        stripe_subscription_id=sub_id,
                    )
                    await session.commit()
                    logger.info(
                        "Created tier for tenant %s = %s (from metadata)",
                        tenant_id, new_tier,
                    )
                else:
                    logger.warning(
                        "Stripe webhook: no tenant found for customer %s",
                        customer_id,
                    )

    elif event_type == "checkout.session.completed":
        # Log successful checkout — subscription events handle the tier change
        cs = event["data"]["object"]
        logger.info(
            "Checkout completed: customer=%s, subscription=%s",
            cs.get("customer"), cs.get("subscription"),
        )

    return {"status": "ok"}
