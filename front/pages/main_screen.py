
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from front.pages.animated_bg import animated_background
from back.select_models import select_models

def main_screen():
    st.set_page_config(layout="wide")
    
    animated_background()
    
    st.markdown(
    """
    <style>
    html, body {
    background: transparent;
    color: #00eaff;
    font-family: "Orbitron", system-ui, sans-serif;
    }

    /* ===== FULLSCREEN ===== */
    section.main {
    position: fixed;
    inset: 0;
    }

    /* ===== sub small ===== */
    .sub-small {
    position: fixed;       /* 画面全体で固定 */
    top: 72%;              /* CONNECT ボタン top:62% + ボタン高さ 約10% */
    left: 50%;             /* 横中央 */
    transform: translateX(-50%);
    font-size: 14px;       /* 小さめ */
    letter-spacing: 0.3em;
    color: #ff00ff;
    text-shadow:
        0 0 8px #ff00ff,
        0 0 20px #ff00ff;
    opacity: 0.85;
    z-index: 10;
    }

    section.main > div {
    padding: 0 !important;
    margin: 0 !important;
    max-width: none !important;
    }

    /* ===== HUD LINES (MOVING) ===== */
    .hud-line {
    position: fixed;
    left: 0;
    right: 0;
    height: 3px;
    pointer-events: none;
    overflow: hidden;
    }

    /* 線の本体 */
    .hud-line::before {
    content: "";
    position: absolute;
    left: -30%;
    top: 0;
    width: 70%;
    height: 100%;
    background: linear-gradient(
        90deg,
        transparent,
        rgba(0,234,255,0.9),
        rgba(255,0,200,0.8),
        transparent
    );
    filter: blur(0.5px);
    animation:
        sweep 6s ease-in-out infinite,
        hue 8s linear infinite;
    opacity: 0;
    }

    /* 上・中・下 */
    .hud-top    { top: 12%; }
    .hud-mid    { top: 50%; }
    .hud-bot    { bottom: 12%; }

    /* 時間ずらし */
    .hud-top::before { animation-delay: 0s, 0s; }
    .hud-mid::before { animation-delay: 0s, 0s; }
    .hud-bot::before { animation-delay: 0s, 0s; }

    /* ===== 横切る動き ===== */
    @keyframes sweep {
    0% {
        left: -30%;
        width: 0%;
        opacity: 0;
    }
    10% {
        opacity: 1;
        width: 20%;
    }
    50% {
        opacity: 1;
        width: 70%;
    }
    80% {
        opacity: 1;
        width: 20%;
    }
    100% {
        left: 100%;
        width: 0%;
        opacity: 0;
    }
    }

    /* ===== 色相変化 ===== */
    @keyframes hue {
    from { filter: hue-rotate(0deg); }
    to   { filter: hue-rotate(360deg); }
    }

    /* ===== CENTER CONTENT ===== */
    .center {
    position: fixed;
    inset: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    }

    /* ===== TITLE ===== */
    .title {
    font-size: 84px;
    letter-spacing: 0.35em;
    animation: hueShift 6s linear infinite;
    text-shadow:
        0 0 25px currentColor,
        0 0 70px currentColor;
    }

    /* 色相変化 */
    @keyframes hueShift {
    0%   { color: #00eaff; }
    33%  { color: #7cff00; }
    66%  { color: #ff2aff; }
    100% { color: #00eaff; }
    }

    /* ===== BROKEN S ===== */
    .broken {
    display: inline-block;
    animation: flicker 2.3s infinite;
    }

    @keyframes flicker {
    0%, 17%, 22%, 100% {
        opacity: 1;
        text-shadow: 0 0 20px currentColor, 0 0 60px currentColor;
    }
    18%, 21% {
        opacity: 0.15;
        text-shadow: none;
    }
    60% { opacity: 0.4; }
    }

    /* ===== SUB ===== */
    .subtitle {
    margin-top: 26px;
    font-size: 18px;
    letter-spacing: 0.3em;
    color: #9bdcff;
    opacity: 0.85;
    }

    /* small sub */
    .sub-small {
    margin-top: 18px;
    font-size: 18px;
    letter-spacing: 0.3em;
    color: #9bdcff;
    opacity: 0.7;
    }

    /* ===== CONNECT BUTTON (ABSOLUTE CENTER) ===== */
    div[data-testid="stButton"] {
    position: fixed;
    top: 62%;
    left: 50%;
    transform: translateX(-50%);
    width: 360px;
    z-index: 10;
    }

    div[data-testid="stButton"] > button {
    width: 100%;
    padding: 20px;
    background: transparent;
    border: 2px solid #00eaff;
    color: #00eaff;
    font-size: 16px;
    letter-spacing: 0.45em;
    box-shadow:
        0 0 35px rgba(0,234,255,0.9),
        inset 0 0 20px rgba(0,234,255,0.6);
    transition: all .25s ease;
    }

    div[data-testid="stButton"] > button:hover {
    background: #00eaff;
    color: #020409;
    box-shadow:
        0 0 90px rgba(0,234,255,1),
        0 0 140px rgba(0,234,255,0.9);
    transform: scale(1.08);
    }
    </style>

    <div class="hud-line hud-top"></div>
    <div class="hud-line hud-mid"></div>
    <div class="hud-line hud-bot"></div>

    """,
        unsafe_allow_html=True,
    )
    
    # CSSでサイズと配置を調整
    st.markdown("""
        <style>
        html, body {
        background: transparent;
        color: #00eaff;
        font-family: "Orbitron", system-ui, sans-serif;
        }
        
        <style>
        /* 1. ボタン全体の配置を中央にする */
        div.stButton {
            display: flex;
            justify-content: center;
        }

        /* 2. ボタンそのものの箱（サイズや形）を調整 */
        div.stButton > button {
            height: 3em;
            width: 5em;
            border-radius: 10px;
        }

        /* 3. 【重要】ボタンの中の文字（pタグ）に直接干渉する */
        div.stButton > button p {
            font-family: "BIZ UDPGothic", sans-serif !important;
            font-size: 24px !important;  /* ここでサイズを自由に変更 */
            font-weight: regular !important;
            line-height: 1.5;            /* 行間を調整して中央に寄せる */
        }
        </style>
        """, unsafe_allow_html=True)
        
    empty_left, col1, col2, col3, empty_right = st.columns([1,2,2,2,1])
    with empty_left:
        st.write("") # 左側の余白（何もしない）
    with empty_right:
        st.write("") # 右側の余白（何もしない）
    with col1:
        selected_stock = st.selectbox("select ticker", ["-","NTT", "Nvidia", "AMD", "Google", "Amazon", "Vodafone", "ベルーナ", "オルカン", "S&P500"], key="stock")
    with col2:
        selected_period = st.selectbox("forecast dates", ["-", "1day", "1week", "1month", "1year"], key="period")
    with col3:
        selected_model = st.selectbox("select model", ["-", "LR", "prophet-f", "prophet-w"], key="model")
        
    empty_l, content, empty_r = st.columns([1, 5, 1])
    with empty_l:
        st.write("") # 左側の余白（何もしない）
    with empty_r:
        st.write("") # 右側の余白（何もしない）
    with content:
        if not (selected_stock == "-" or selected_period == "-" or selected_model == "-"):
            st.cache_data.clear()
            st.cache_resource.clear()
            act_date, act_data, pred_date, pred_data = select_models(selected_stock, selected_period, selected_model)

            # 1. グラフオブジェクトの作成
            fig = px.line()
            fig.add_scatter(x=act_date, y=act_data, mode='lines+markers', name='Actual', line=dict(color='blue'))
            fig.add_scatter(x=pred_date, y=pred_data, mode='lines+markers', name='Predicted', line=dict(color='red'))
            fig.update_layout(margin=dict(l=10, r=10, t=50, b=20)) # 左・右・上・下の余白
            fig.update_layout(height=650)
            fig.update_layout(font=dict(family="\"BIZ UDPGothic\", \"BIZ UDPゴシック\", Meiryo, sans-serif", size=18, color="white"))
            fig.update_layout(xaxis=dict(title=dict(text="Time（dates）",font=dict(size=18)),tickfont=dict(size=18)))
            fig.update_layout(yaxis=dict(title=dict(text="Stock Price （￥）",font=dict(size=24)),tickfont=dict(size=24)))
            # 2. Streamlitで表示
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("") # 何もしない
            pass
    
    print(selected_stock, selected_period, selected_model)
    
