"""Streamlit dashboard for the Emergency Information Extractor."""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, Optional
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure project root is importable when Streamlit runs from /app.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import run_pipeline_from_text
from src.notifications import send_emergency_email


def _inject_styles() -> None:
    """Inject custom CSS to create a polished dashboard look."""
    st.markdown(
        """
        <style>
        .stApp {
            background: radial-gradient(1200px 800px at 10% 10%, #16335b 0%, #0b132b 45%, #070b1c 100%);
            color: #f3f6ff;
        }

        .main > div {
            padding-top: 1.5rem;
        }

        .hero {
            border-radius: 22px;
            padding: 1.8rem 1.5rem;
            background: linear-gradient(130deg, rgba(255, 89, 94, 0.85), rgba(255, 146, 69, 0.85), rgba(0, 163, 255, 0.85));
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.35);
            position: relative;
            overflow: hidden;
            animation: pulseGlow 7s ease-in-out infinite;
        }

        .hero::after {
            content: "";
            position: absolute;
            width: 240px;
            height: 240px;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.15);
            right: -70px;
            top: -70px;
        }

        .hero-title {
            font-size: 2rem;
            font-weight: 800;
            letter-spacing: 0.4px;
            margin-bottom: 0.3rem;
        }

        .hero-subtitle {
            font-size: 1rem;
            opacity: 0.96;
            line-height: 1.5;
            margin-bottom: 0;
        }

        .glass-card {
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.16);
            border-radius: 18px;
            padding: 1rem 1rem 0.7rem;
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
            backdrop-filter: blur(8px);
        }

        .metric-label {
            font-size: 0.85rem;
            color: #d4deff;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 0.15rem;
        }

        .metric-value {
            font-size: 1.65rem;
            font-weight: 700;
            color: #ffffff;
        }

        .result-card {
            background: rgba(17, 25, 48, 0.75);
            border: 1px solid rgba(133, 170, 255, 0.35);
            border-radius: 16px;
            padding: 1rem;
            margin-top: 0.7rem;
        }

        @keyframes pulseGlow {
            0%, 100% { box-shadow: 0 20px 40px rgba(0, 0, 0, 0.35); }
            50% { box-shadow: 0 25px 55px rgba(255, 132, 76, 0.45); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _severity_color(severity: str) -> str:
    """Return a color for severity level text."""
    palette = {
        "low": "#74d680",
        "medium": "#ffd166",
        "high": "#ff7f50",
        "critical": "#ff4d6d",
    }
    return palette.get(str(severity).lower(), "#9eb4ff")


def _safe_float(value: Any) -> Optional[float]:
    """Convert numeric-like values to float safely."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _render_kpi(label: str, value: str) -> None:
    """Render a custom KPI card."""
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_map(result: Dict[str, Any]) -> None:
    """Render map for incident coordinates when available."""
    lat = _safe_float(result.get("latitude"))
    lon = _safe_float(result.get("longitude"))

    if lat is None or lon is None:
        st.info("Map unavailable because valid coordinates were not detected.")
        return

    st.markdown("#### Incident Location Map")
    map_df = pd.DataFrame([{"lat": lat, "lon": lon}])
    st.map(map_df, zoom=11)


def _render_result(result: Dict[str, Any]) -> None:
    """Render extracted structured output and visuals."""
    incident = str(result.get("incident", "Unknown"))
    severity = str(result.get("severity", "Unknown"))
    priority = result.get("priority_score", "N/A")
    language = str(result.get("language", "Unknown"))
    location = result.get("location") or "Not detected"
    confidence = result.get("confidence", "N/A")
    duplicate = result.get("duplicate", "N/A")
    victims = result.get("victims", "N/A")

    left, middle, right = st.columns(3)
    with left:
        _render_kpi("Incident Type", incident)
    with middle:
        _render_kpi("Priority Score", str(priority))
    with right:
        _render_kpi("Language", language.upper())

    st.markdown(
        f"""
        <div class="result-card">
            <h4 style="margin:0 0 0.6rem 0; color:#f5f7ff;">Structured Emergency Output</h4>
            <p style="margin:0.25rem 0;"><b>Location:</b> {location}</p>
            <p style="margin:0.25rem 0;"><b>Severity:</b>
              <span style="color:{_severity_color(severity)}; font-weight:700;">{severity}</span>
            </p>
            <p style="margin:0.25rem 0;"><b>Confidence:</b> {confidence}</p>
            <p style="margin:0.25rem 0;"><b>Victims:</b> {victims}</p>
            <p style="margin:0.25rem 0;"><b>Duplicate Check:</b> {duplicate}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _render_map(result)

    with st.expander("View Raw JSON Output", expanded=False):
        st.code(json.dumps(result, indent=2), language="json")


def main() -> None:
    """Run Streamlit UI for emergency extraction."""
    st.set_page_config(
        page_title="Emergency Monitoring Dashboard",
        page_icon="🚨",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _inject_styles()

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">🚨 Emergency Information Extractor</div>
            <p class="hero-subtitle">
                Enter an emergency report, run AI extraction, and monitor location + severity in a live dashboard.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("## Control Panel")
    st.sidebar.write("Use the controls below to simulate incoming emergency text in real time.")
    st.sidebar.markdown("### Email Alerts")
    enable_email = st.sidebar.toggle("Send email alert after each analysis", value=True)
    recipient_email = st.sidebar.text_input(
        "Recipient Email",
        value="baalebailmane@gmail.com",
        placeholder="you@example.com",
        help="Emergency alert will be sent to this email address.",
    )
    show_tips = st.sidebar.toggle("Show input examples", value=True)
    if show_tips:
        st.sidebar.caption("Example: Fire near MG Road Gurugram, people injured and multiple vehicles burned.")

    user_input = st.text_area(
        "Emergency Input",
        value="Fire near MG Road Gurugram, multiple vehicles burnt, people injured.",
        height=140,
        help="Type or paste a distress report message.",
    )

    run_clicked = st.button("Analyze Emergency Report", type="primary", use_container_width=True)

    if run_clicked:
        if not user_input.strip():
            st.warning("Please enter an emergency report before running the pipeline.")
            return

        with st.spinner("Analyzing emergency report..."):
            try:
                result = run_pipeline_from_text(user_input.strip())
            except Exception as exc:  # noqa: BLE001
                st.error(f"Pipeline execution failed: {exc}")
                return

        st.success("Analysis complete. Dashboard updated with extracted emergency intelligence.")
        _render_result(result)

        if enable_email:
            if not recipient_email.strip():
                st.warning("Email alert is enabled, but recipient email is empty.")
            else:
                try:
                    send_emergency_email(
                        result=result,
                        recipient_email=recipient_email.strip(),
                        raw_text=user_input.strip(),
                    )
                    st.success(f"Emergency alert email sent to {recipient_email.strip()}.")
                except Exception as exc:  # noqa: BLE001
                    st.error(
                        "Failed to send email alert. "
                        "Check SMTP_SENDER_EMAIL and SMTP_SENDER_PASSWORD environment variables. "
                        f"Details: {exc}"
                    )
    else:
        st.info("Enter emergency text and click **Analyze Emergency Report** to run the pipeline.")


if __name__ == "__main__":
    main()

