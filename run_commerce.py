import argparse
import asyncio
import contextlib
import io
import os
import sys

from app.logger import define_log_level, logger


@contextlib.contextmanager
def _suppress_process_stdout():
    try:
        stdout_fd = sys.stdout.fileno()
    except (AttributeError, OSError, io.UnsupportedOperation):
        with contextlib.redirect_stdout(io.StringIO()):
            yield
        return

    saved_fd = os.dup(stdout_fd)
    try:
        with open(os.devnull, "w") as devnull:
            sys.stdout.flush()
            os.dup2(devnull.fileno(), stdout_fd)
            with contextlib.redirect_stdout(io.StringIO()):
                yield
    finally:
        try:
            os.dup2(saved_fd, stdout_fd)
        finally:
            os.close(saved_fd)


async def main():
    with _suppress_process_stdout():
        from app.config import config
        from app.flow.commerce import (
            DEFAULT_STABLE_PUBLIC_WEB_PROMPT,
            CommerceDecisionFlow,
        )

    define_log_level(name="commerce")
    parser = argparse.ArgumentParser(
        description="Run the cross-platform general product research and decision agent"
    )
    parser.add_argument("--prompt", type=str, required=False, help="User request")
    parser.add_argument(
        "--demo-profile",
        type=str,
        choices=["default", "stable_public_web"],
        default="default",
        help="Optional execution profile for a more stable public-web product-research demo.",
    )
    parser.add_argument(
        "--execution-profile",
        type=str,
        choices=["default", "stable_public_web", "product_compare_v2"],
        default=None,
        help="Explicit product-research execution profile. Overrides --demo-profile when provided.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=900,
        help="Overall timeout for the commerce flow",
    )
    parser.add_argument(
        "--browser-session-mode",
        type=str,
        choices=["auto", "public_only", "local_cdp"],
        default=None,
        help="Override browser session strategy for commerce browsing.",
    )
    args = parser.parse_args()

    execution_profile = args.execution_profile or args.demo_profile

    if args.prompt:
        prompt = args.prompt
    elif execution_profile == "stable_public_web":
        prompt = DEFAULT_STABLE_PUBLIC_WEB_PROMPT
        logger.warning(
            "No prompt provided. Using the stable public web demo prompt."
        )
    else:
        prompt = input("Enter your commerce request: ")
    if not prompt.strip():
        logger.warning("Empty prompt provided.")
        return

    if args.browser_session_mode:
        config.set_browser_session_mode(args.browser_session_mode)

    try:
        with _suppress_process_stdout():
            flow = CommerceDecisionFlow(agents={}, execution_profile=execution_profile)
    except ValueError as exc:
        logger.error(str(exc))
        return
    logger.warning("Processing commerce decision request...")
    try:
        with _suppress_process_stdout():
            result = await asyncio.wait_for(
                flow.execute(prompt), timeout=args.timeout_seconds
            )
    except asyncio.TimeoutError:
        logger.error(
            f"Commerce flow timed out after {args.timeout_seconds} seconds."
        )
        return

    print(result)


if __name__ == "__main__":
    asyncio.run(main())
