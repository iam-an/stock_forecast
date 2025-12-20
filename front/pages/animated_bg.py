import streamlit.components.v1 as components

def animated_background():
    components.html(
        """
<style>
/* 背景全体 */
body {
  margin: 0;
  padding: 0;
  overflow: hidden;
}

.full-bg {
  position: fixed;
  inset: 0;
  background:
    radial-gradient(circle at 50% 30%, rgba(0,255,255,0.2), transparent 70%),
    linear-gradient(135deg, #0d0015, #000108 80%);
  z-index: -10;
}

/* ライングリッド */
.full-bg::before {
  content:"";
  position:absolute;
  inset:0;
  background:
    repeating-linear-gradient(90deg, rgba(0,234,255,0.05), rgba(0,234,255,0.05) 1px, transparent 1px, transparent 100px),
    repeating-linear-gradient(0deg, rgba(0,234,255,0.05), rgba(0,234,255,0.05) 1px, transparent 1px, transparent 100px);
  opacity:0.8;
}
</style>

<div class="full-bg"></div>
""",
        height=0
    )
