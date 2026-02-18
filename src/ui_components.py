
import streamlit as st

import textwrap

def card(title: str, content: str, icon: str = None):
    """
    Renders a styled card component.
    """
    # Check if icon is SVG
    if icon and icon.strip().startswith("<svg"):
        icon_html = f'<div style="width: 32px; height: 32px; margin-bottom: 0.5rem; color: var(--accent-color);">{icon}</div>'
    elif icon:
        icon_html = f'<div style="font-size: 2.5rem; margin-bottom: 0.5rem;">{icon}</div>'
    else:
        icon_html = ""
    
    # Use st.markdown directly with a single line string to avoid any indentation issues
    st.markdown(f'<div class="apple-card">{icon_html}<h3>{title}</h3><div style="color: var(--text-sub); font-size: 16px; line-height: 1.5;">{content}</div></div>', unsafe_allow_html=True)

def metric_card(label: str, value: str):
    """
    Renders a simple metric card.
    """
    st.markdown(f'<div class="apple-card" style="text-align: center; padding: 1.5rem;"><div style="font-size: 0.9rem; color: var(--text-sub); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;">{label}</div><div style="font-size: 2rem; font-weight: 700; color: var(--text-main);">{value}</div></div>', unsafe_allow_html=True)

def section_header(title: str, subtitle: str = None):
    """
    Renders a section header with optional subtitle.
    """
    st.markdown(f"<h2>{title}</h2>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<p style='margin-bottom: 24px;'>{subtitle}</p>", unsafe_allow_html=True)

def feature_list(title: str, items: list):
    """
    Renders a list of features inside a card.
    """
    items_html = "".join([f"<li><strong>{item['name']}</strong> - {item['desc']}</li>" for item in items])
    
    st.markdown(f'<div class="apple-card"><h4>{title}</h4><ul style="padding-left: 1.2rem; color: var(--text-sub); margin-top: 12px;">{items_html}</ul></div>', unsafe_allow_html=True)
