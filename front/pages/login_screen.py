import streamlit as st
from front.pages.main_screen import main_screen
from front.pages.animated_bg import animated_background



def login():
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
  font-size: 25px;
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

<div class="center">
  <div class="title">
    <span class="broken">S</span>TOCK FORECAST
  </div>
  <div class="subtitle">CYBER NETWORK SYSTEM</div>
</div>

<div class="sub-small">BOYS BE AMBITIOUS</div>
<style>

""",
        unsafe_allow_html=True,
    )

    if st.button("CONNECT"):
        st.session_state.page = "main_screen"
        st.rerun()