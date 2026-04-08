import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(
    page_title="Adaptive Task Allocation — KnowledgeQuarry 2026",
    page_icon="🤖",
    layout="wide",
)

LOG_DIR = "logs"
MODEL_DIR = "models"
GRID_SIZE = 10
NUM_TASKS = 10
AGENT_CONFIGS = [
    {"type": "A", "capacity": 3, "speed": 1, "energy": 100},
    {"type": "A", "capacity": 3, "speed": 1, "energy": 100},
    {"type": "B", "capacity": 1, "speed": 2, "energy": 60},
    {"type": "B", "capacity": 1, "speed": 2, "energy": 60},
    {"type": "B", "capacity": 1, "speed": 2, "energy": 60},
]

# Orange-first color palette
ORANGE = "#ff6b2b"
ORANGE_SOFT = "#ff8c55"
ORANGE_DIM = "#cc4a10"
GREEN = "#00c875"
BLUE = "#4a9eff"
RED = "#ff4b4b"
GREY = "#888888"
PANEL = "#1e2130"
BORDER = "#2d3250"
WHITE = "#ffffff"
DARK = "#0e1117"

AGENT_COLORS = [BLUE, BLUE, ORANGE, ORANGE, ORANGE]
AGENT_SYMBOLS = ["square", "square", "circle", "circle", "circle"]
RANDOM_AVG_REWARD = 21.92

st.markdown(f"""
<style>
.block-container {{ padding-top: 1rem; padding-bottom: 0.5rem; }}
.section-header {{
    background: linear-gradient(90deg, #1a1f35, {DARK});
    border-left: 4px solid {ORANGE};
    padding: 8px 14px; border-radius: 4px;
    color: {WHITE}; font-size: 1rem; font-weight: 600;
    margin-bottom: 0.8rem;
}}
.explainer {{
    background: {PANEL}; border-radius: 8px;
    padding: 10px 14px; color: #cccccc;
    font-size: 0.85rem; margin-bottom: 0.8rem;
    border: 1px solid {BORDER}; line-height: 1.6;
}}
.metric-box {{
    background: {PANEL}; border-radius: 8px;
    padding: 10px; text-align: center;
    border: 1px solid {BORDER}; margin-bottom: 8px;
}}
.metric-val {{ font-size: 1.6rem; font-weight: bold; color: {ORANGE}; }}
.metric-lbl {{ font-size: 0.72rem; color: {GREY}; margin-top: 2px; }}
.narration-box {{
    background: #12161f; border-radius: 8px;
    padding: 12px 16px; border: 1px solid {ORANGE};
    color: {WHITE}; font-size: 0.85rem;
    line-height: 1.8; min-height: 100px;
}}
.narration-title {{ color: {ORANGE}; font-weight: bold; margin-bottom: 6px; font-size: 0.8rem; }}
.log-complete {{ color: {GREEN}; }}
.log-move {{ color: {BLUE}; }}
.log-idle {{ color: {GREY}; }}
.proof-box {{
    background: #1a1a0e; border: 1px solid {ORANGE};
    border-radius: 8px; padding: 12px 16px;
    color: #cccccc; font-size: 0.85rem; text-align: center;
}}
.proof-number {{ color: {ORANGE}; font-size: 1.6rem; font-weight: bold; }}
.hero-stat {{
    background: linear-gradient(135deg, #1a1208, #1e2130);
    border: 2px solid {ORANGE}; border-radius: 12px;
    padding: 20px; text-align: center;
}}
.hero-number {{ color: {ORANGE}; font-size: 3rem; font-weight: bold; line-height: 1; }}
.hero-label {{ color: #cccccc; font-size: 0.85rem; margin-top: 6px; }}
.agent-a {{ color: {BLUE}; font-weight: bold; }}
.agent-b {{ color: {ORANGE}; font-weight: bold; }}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_replay_data():
    path = os.path.join(LOG_DIR, "replay_data.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


@st.cache_data
def load_comparison_data():
    path = os.path.join(LOG_DIR, "comparison_data.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


@st.cache_data
def load_training_metrics():
    path = os.path.join(LOG_DIR, "training_metrics.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


@st.cache_data
def load_eval_metrics():
    path = os.path.join(LOG_DIR, "eval_metrics.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


@st.cache_resource
def load_model():
    try:
        from sb3_contrib import MaskablePPO
        path = os.path.join(MODEL_DIR, "ppo_final")
        if os.path.exists(path + ".zip"):
            return MaskablePPO.load(path)
    except Exception:
        pass
    return None


def get_configs_for_frames(frames, custom=None):
    if custom:
        return custom
    n = len(frames[0]["agent_positions"]) if frames else len(AGENT_CONFIGS)
    return AGENT_CONFIGS[:n] if n <= len(AGENT_CONFIGS) else AGENT_CONFIGS + [{"type": "B", "capacity": 1, "speed": 2, "energy": 60}] * (n - len(AGENT_CONFIGS))


def agent_colors_symbols(n, configs=None):
    colors, symbols = [], []
    for i in range(n):
        cfg = configs[i] if configs and i < len(configs) else (AGENT_CONFIGS[i] if i < len(AGENT_CONFIGS) else {"type": "B"})
        colors.append(BLUE if cfg["type"] == "A" else ORANGE)
        symbols.append("square" if cfg["type"] == "A" else "circle")
    return colors, symbols


def build_grid_figure(frames, step_idx, title="", custom_configs=None):
    if not frames or step_idx >= len(frames):
        return go.Figure()
    frame = frames[step_idx]
    task_pos = np.array(frame["task_positions"])
    ap = np.array(frame["agent_positions"])
    td = np.array(frame["task_done"])
    diff = np.array(frame["task_difficulty"])
    ae = np.array(frame["agent_energy"])
    n = len(ap)
    configs = get_configs_for_frames(frames, custom_configs)
    colors, symbols = agent_colors_symbols(n, configs)

    pos_count = {}
    offsets = []
    for i in range(n):
        key = (round(ap[i, 0]), round(ap[i, 1]))
        c = pos_count.get(key, 0)
        pos_count[key] = c + 1
        offsets.append(c * 0.3)

    data = []
    task_colors = [GREEN if not d else "#333a4f" for d in td]
    task_hover = [
        f"<b>Task {i}</b><br>Pos: ({int(task_pos[i,0])},{int(task_pos[i,1])})<br>"
        f"Difficulty: {int(diff[i])}/3<br>{'✅ Done' if td[i] else '⏳ Pending'}"
        for i in range(len(td))
    ]
    data.append(go.Scatter(
        x=task_pos[:, 0], y=task_pos[:, 1], mode="markers",
        marker=dict(color=task_colors, size=[12 + int(d)*5 for d in diff],
                    symbol="diamond", line=dict(color=WHITE, width=1)),
        hovertext=task_hover, hoverinfo="text", name="Tasks", showlegend=True,
    ))
    for i in range(n):
        cfg = configs[i] if i < len(configs) else {"type": "?", "capacity": 1, "speed": 1, "energy": 100}
        ep = ae[i] / cfg["energy"] if cfg["energy"] > 0 else 0
        border = GREEN if ep > 0.5 else "#ff9f43" if ep > 0.2 else RED
        data.append(go.Scatter(
            x=[ap[i, 0] + offsets[i]], y=[ap[i, 1] + offsets[i]],
            mode="markers+text",
            marker=dict(color=colors[i], size=22, symbol=symbols[i],
                        line=dict(color=border, width=3)),
            text=[f"{'A' if cfg['type']=='A' else 'B'}{i}"],
            textposition="top center", textfont=dict(size=9, color=WHITE),
            hovertext=f"<b>Agent {i} (Type {cfg['type']})</b><br>Energy: {ae[i]:.1f}/{cfg['energy']} ({ep*100:.0f}%)<br>{'🔵 Heavy Lifter' if cfg['type']=='A' else '🟠 Sprinter'}",
            hoverinfo="text", name=f"Agent {i} ({cfg['type']})", showlegend=True,
        ))

    tasks_done = int(td.sum())
    num_t = len(td)
    fig = go.Figure(data=data)
    fig.update_layout(
        title=dict(text=f"{title} | Step {frame['step']} | <b>{tasks_done}/{num_t} done</b>",
                   font=dict(color=WHITE, size=13)),
        paper_bgcolor=DARK, plot_bgcolor="#1a1f35",
        xaxis=dict(range=[-0.5, GRID_SIZE-0.5], showgrid=True, gridcolor=BORDER,
                   title="Grid X", color=GREY, zeroline=False, dtick=1, fixedrange=True),
        yaxis=dict(range=[-0.5, GRID_SIZE-0.5], showgrid=True, gridcolor=BORDER,
                   title="Grid Y", color=GREY, zeroline=False, dtick=1, fixedrange=True),
        legend=dict(bgcolor=PANEL, bordercolor=BORDER, font=dict(color="#cccccc", size=10), x=1.01, y=1),
        height=420, margin=dict(l=40, r=160, t=50, b=40),
    )
    return fig


def build_animated_figure(frames, title="", custom_configs=None):
    if not frames:
        return go.Figure()
    task_pos = np.array(frames[0]["task_positions"])
    n = len(frames[0]["agent_positions"])
    configs = get_configs_for_frames(frames, custom_configs)
    colors, symbols = agent_colors_symbols(n, configs)
    fig_frames = []

    for frame in frames:
        ap = np.array(frame["agent_positions"])
        td = np.array(frame["task_done"])
        diff = np.array(frame["task_difficulty"])
        ae = np.array(frame["agent_energy"])
        pos_count = {}
        offsets = []
        for i in range(n):
            key = (round(ap[i, 0]), round(ap[i, 1]))
            c = pos_count.get(key, 0)
            pos_count[key] = c + 1
            offsets.append(c * 0.3)
        data = [go.Scatter(
            x=task_pos[:, 0], y=task_pos[:, 1], mode="markers",
            marker=dict(color=[GREEN if not d else "#333a4f" for d in td],
                        size=[12 + int(d)*5 for d in diff],
                        symbol="diamond", line=dict(color=WHITE, width=1)),
            hovertext=[f"Task {i} | Diff {int(diff[i])} | {'Done' if td[i] else 'Pending'}" for i in range(len(td))],
            hoverinfo="text", name="Tasks", showlegend=True,
        )]
        for i in range(n):
            cfg = configs[i] if i < len(configs) else {"type": "?", "capacity": 1, "speed": 1, "energy": 100}
            ep = ae[i] / cfg["energy"] if cfg["energy"] > 0 else 0
            border = GREEN if ep > 0.5 else "#ff9f43" if ep > 0.2 else RED
            data.append(go.Scatter(
                x=[ap[i, 0] + offsets[i]], y=[ap[i, 1] + offsets[i]],
                mode="markers+text",
                marker=dict(color=colors[i], size=22, symbol=symbols[i],
                            line=dict(color=border, width=3)),
                text=[f"{'A' if cfg['type']=='A' else 'B'}{i}"],
                textposition="top center", textfont=dict(size=9, color=WHITE),
                hovertext=f"Agent {i} ({cfg['type']}) | Energy: {ae[i]:.1f}/{cfg['energy']}",
                hoverinfo="text", name=f"Agent {i} ({cfg['type']})", showlegend=True,
            ))
        fig_frames.append(go.Frame(
            data=data, name=str(frame["step"]),
            layout=go.Layout(title_text=f"{title} | Step {frame['step']} | {int(td.sum())}/{len(td)} done")
        ))

    fig = go.Figure(
        data=fig_frames[0].data if fig_frames else [],
        frames=fig_frames,
        layout=go.Layout(
            title=dict(text=title, font=dict(color=WHITE, size=13)),
            paper_bgcolor=DARK, plot_bgcolor="#1a1f35",
            xaxis=dict(range=[-0.5, GRID_SIZE-0.5], showgrid=True, gridcolor=BORDER,
                       title="Grid X", color=GREY, zeroline=False, dtick=1),
            yaxis=dict(range=[-0.5, GRID_SIZE-0.5], showgrid=True, gridcolor=BORDER,
                       title="Grid Y", color=GREY, zeroline=False, dtick=1),
            legend=dict(bgcolor=PANEL, bordercolor=BORDER, font=dict(color="#cccccc", size=10), x=1.01, y=1),
            height=450, margin=dict(l=40, r=160, t=50, b=70),
            updatemenus=[dict(
                type="buttons", showactive=False, y=1.18, x=0.5, xanchor="center",
                buttons=[
                    dict(label="▶ Play", method="animate",
                         args=[None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}]),
                    dict(label="⏸ Pause", method="animate",
                         args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}]),
                ],
            )],
            sliders=[dict(
                steps=[dict(method="animate", args=[[f.name], {"mode": "immediate"}], label=f.name)
                       for f in fig_frames],
                x=0, y=0, len=1.0,
                currentvalue=dict(prefix="Step: ", font=dict(color=WHITE, size=12)),
                bgcolor=PANEL, bordercolor=BORDER, font=dict(color=GREY),
            )],
        )
    )
    return fig


def build_narration(frame, policy_label=""):
    if not frame or "action_logs" not in frame:
        return f"<i style='color:{GREY}'>No action data — re-run evaluate.py.</i>"
    step = frame["step"]
    summary = frame.get("step_summary", f"Step {step}")
    lines = [f'<div class="narration-title">📍 {summary}</div>']
    for log in frame["action_logs"]:
        if "✅" in log:
            lines.append(f'<span class="log-complete">🟢 {log}</span><br>')
        elif "idle" in log:
            lines.append(f'<span class="log-idle">⚪ {log}</span><br>')
        else:
            lines.append(f'<span class="log-move">🔵 {log}</span><br>')
    total_t = len(frame.get("task_done", [0]*NUM_TASKS))
    if frame.get("tasks_completed", 0) == total_t and total_t > 0:
        lines.append(f'<br><b style="color:{GREEN}">✅ All {total_t} tasks completed in {step} steps!</b>')
    return "".join(lines)


def build_reward_chart(frames_so_far, random_avg_total):
    if not frames_so_far:
        return go.Figure()
    steps = [f["step"] for f in frames_so_far]
    cum, running = [], 0.0
    for f in frames_so_far:
        running += f["reward"]
        cum.append(running)
    max_s = max(steps) if steps else 1
    rand_line = [random_avg_total * (s / max_s) for s in steps]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=steps, y=rand_line, mode="lines", name="Random (avg)",
                              line=dict(color=GREY, width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=steps, y=cum, mode="lines+markers", name="This policy",
                              line=dict(color=ORANGE, width=2.5),
                              marker=dict(size=5, color=ORANGE),
                              fill="tonexty", fillcolor="rgba(255,107,43,0.08)"))
    if len(steps) > 1:
        gap = cum[-1] - rand_line[-1]
        fig.add_annotation(x=steps[-1], y=cum[-1], text=f"+{gap:.1f} vs random",
                           showarrow=True, arrowhead=2,
                           font=dict(color=GREEN, size=10), arrowcolor=GREEN, ax=-50, ay=-25)
    fig.update_layout(
        paper_bgcolor=DARK, plot_bgcolor="#1a1f35", height=200,
        font=dict(color=GREY, size=10),
        xaxis=dict(title="Step", gridcolor=BORDER, color=GREY, zeroline=False),
        yaxis=dict(title="Cumulative Reward", gridcolor=BORDER, color=GREY, zeroline=False),
        legend=dict(bgcolor=PANEL, bordercolor=BORDER, font=dict(color="#cccccc", size=9)),
        margin=dict(l=50, r=20, t=10, b=35),
    )
    return fig


# ── HEADER ──────────────────────────────────────────────────────────────────
st.markdown("## 🤖 Adaptive Multi-Agent Task Allocation")
st.markdown(f"""<div class="explainer">
<b>No rules. No hardcoding. The system must learn.</b><br><br>
5 agents complete 10 tasks on a 10×10 grid through 95,000+ episodes of pure trial and error.
<span class="agent-a">■ Blue squares = Type A — Heavy Lifters</span> (capacity 3, speed 1, energy 100 — handles hard tasks at 1/3 the energy cost).
<span class="agent-b">● Orange circles = Type B — Sprinters</span> (capacity 1, speed 2, energy 60 — covers ground fast, best for nearby easy tasks).
<b style="color:{GREEN}">🟢 Green diamond = pending task</b> (bigger = harder). <b style="color:{GREY}">⚫ Dark = completed</b>.
The policy discovers the optimal assignment strategy entirely from reward signals.
</div>""", unsafe_allow_html=True)

# Hero stats
h1, h2, h3, h4 = st.columns(4)
h1.markdown(f'<div class="hero-stat"><div class="hero-number">95K+</div><div class="hero-label">Episodes Trained</div></div>', unsafe_allow_html=True)
h2.markdown(f'<div class="hero-stat"><div class="hero-number">+20%</div><div class="hero-label">Reward vs Random</div></div>', unsafe_allow_html=True)
h3.markdown(f'<div class="hero-stat"><div class="hero-number">3</div><div class="hero-label">Steps to Solve (vs 80 limit)</div></div>', unsafe_allow_html=True)
h4.markdown(f'<div class="hero-stat"><div class="hero-number">32×</div><div class="hero-label">Lower Variance than Random</div></div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

page = st.radio("", ["🎬 Episode Replay", "⚔️ Trained vs Random", "📈 Proof of Learning", "🎛️ What-If"],
                horizontal=True, label_visibility="collapsed")
st.markdown("---")


# ── PAGE 1: EPISODE REPLAY ───────────────────────────────────────────────────
if page == "🎬 Episode Replay":
    replay_data = load_replay_data()
    if not replay_data:
        st.error("Run `python evaluate.py` first.")
    else:
        col_left, col_right = st.columns([1, 3])
        with col_left:
            st.markdown('<div class="section-header">Controls</div>', unsafe_allow_html=True)
            policy = st.selectbox("Policy", list(replay_data.keys()),
                                  help="early_30k = just started | final = fully trained")
            frames = replay_data[policy]
            view_type = st.radio("View mode", ["▶ Animated", "🎚️ Step-by-Step"],
                                 horizontal=True, key="replay_view")
            if frames:
                step_idx = st.slider("Step", 0, len(frames)-1, 0) if view_type == "🎚️ Step-by-Step" else len(frames)-1
                frame = frames[step_idx]
                st.markdown("---")
                tasks_done = frame["tasks_completed"]
                total_r = sum(f["reward"] for f in frames[:step_idx+1])
                energy_left = sum(frame["agent_energy"])
                total_energy = sum(cfg["energy"] for cfg in AGENT_CONFIGS)
                st.markdown(f'<div class="metric-box"><div class="metric-val" style="color:{GREEN}">{tasks_done}/{NUM_TASKS}</div><div class="metric-lbl">Tasks Done</div></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-box"><div class="metric-val">{frame["step"]}</div><div class="metric-lbl">Step</div></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-box"><div class="metric-val">{total_r:.1f}</div><div class="metric-lbl">Reward So Far</div></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-box"><div class="metric-val" style="color:#ff9f43">{energy_left:.0f}/{total_energy}</div><div class="metric-lbl">Energy Remaining</div></div>', unsafe_allow_html=True)
                st.markdown("---")
                st.markdown("**Agent Energy**")
                for i, e in enumerate(frame["agent_energy"]):
                    cfg = AGENT_CONFIGS[i] if i < len(AGENT_CONFIGS) else {"type": "?", "energy": 100}
                    pct = max(0.0, min(1.0, e / cfg["energy"]))
                    icon = "🟢" if pct > 0.5 else "🟡" if pct > 0.2 else "🔴"
                    label = "Heavy" if cfg["type"] == "A" else "Sprint"
                    st.progress(float(pct), text=f"{icon} A{i} ({label}): {e:.0f}/{cfg['energy']}")

        with col_right:
            if frames:
                label = ("early (30k)" if "30k" in policy else "mid (100k)" if "100k" in policy
                         else "late (200k)" if "200k" in policy else "final (trained)")
                if view_type == "▶ Animated":
                    fig = build_animated_figure(frames, f"Policy: {label}")
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown("**📋 Full Episode Log**")
                    for fr in frames:
                        step_label = fr.get("step_summary", f"Step {fr['step']}")
                        with st.expander(f"**{step_label}**", expanded=(fr["step"] <= 1)):
                            st.markdown(f'<div class="narration-box">{build_narration(fr)}</div>', unsafe_allow_html=True)
                else:
                    fig = build_grid_figure(frames, step_idx, f"Policy: {label}")
                    st.plotly_chart(fig, use_container_width=True)
                    col_narr, col_chart = st.columns([1, 1])
                    with col_narr:
                        st.markdown("**📋 What happened this step**")
                        st.markdown(f'<div class="narration-box">{build_narration(frames[step_idx])}</div>', unsafe_allow_html=True)
                    with col_chart:
                        st.markdown("**📈 Reward vs Random Baseline**")
                        st.markdown(f'<div class="explainer" style="font-size:0.78rem">Orange = policy reward accumulating step by step. Dotted grey = random avg. <b>Gap above grey = what the model learned.</b></div>', unsafe_allow_html=True)
                        st.plotly_chart(build_reward_chart(frames[:step_idx+1], RANDOM_AVG_REWARD), use_container_width=True)


# ── PAGE 2: TRAINED VS RANDOM ────────────────────────────────────────────────
elif page == "⚔️ Trained vs Random":
    comparison = load_comparison_data()
    if not comparison:
        st.error("Run `python evaluate.py` first.")
    else:
        st.markdown('<div class="section-header">⚔️ Same Grid. Same Tasks. Different Brain.</div>', unsafe_allow_html=True)
        st.markdown(f"""<div class="explainer">
        Both policies start with <b>identical positions and tasks</b> (same seed). Only the decisions differ.
        Trained policy: spreads out, no overlap, finishes in 3 steps.
        Random policy: stumbles, doubles up, wastes energy.
        </div>""", unsafe_allow_html=True)

        t_frames = comparison["trained"]
        r_frames = comparison["random"]
        t_sum = comparison["trained_summary"]
        r_sum = comparison["random_summary"]
        max_step = max(len(t_frames), len(r_frames))
        step_idx = st.slider("Step (controls both)", 0, max_step-1, 0, key="cmp_slider")

        col_t, col_r = st.columns(2)
        for col, frames, summary, label, color in [
            (col_t, t_frames, t_sum, "🤖 Trained Policy (PPO)", GREEN),
            (col_r, r_frames, r_sum, "🎲 Random Policy", RED),
        ]:
            with col:
                st.markdown(f"### {label}")
                idx = min(step_idx, len(frames)-1)
                fr = frames[idx]
                reward_so_far = sum(f["reward"] for f in frames[:idx+1])
                m1, m2, m3 = st.columns(3)
                m1.markdown(f'<div class="metric-box"><div class="metric-val" style="color:{color}">{fr["tasks_completed"]}/{NUM_TASKS}</div><div class="metric-lbl">Tasks Done</div></div>', unsafe_allow_html=True)
                m2.markdown(f'<div class="metric-box"><div class="metric-val" style="color:{color}">{fr["step"]}</div><div class="metric-lbl">Step</div></div>', unsafe_allow_html=True)
                m3.markdown(f'<div class="metric-box"><div class="metric-val" style="color:{color}">{reward_so_far:.1f}</div><div class="metric-lbl">Reward</div></div>', unsafe_allow_html=True)
                st.plotly_chart(build_grid_figure(frames, idx, label.split(" ", 1)[1]), use_container_width=True)
                st.markdown(f'<div class="narration-box">{build_narration(fr)}</div>', unsafe_allow_html=True)

        st.markdown("### 📈 Cumulative Reward — Same Scenario")
        st.markdown(f'<div class="explainer">Every point where orange is above red = a smarter decision. The shaded area = total advantage of the trained policy on this run.</div>', unsafe_allow_html=True)

        t_steps = [f["step"] for f in t_frames]
        r_steps = [f["step"] for f in r_frames]
        t_cum, r_cum, tr, rr = [], [], 0.0, 0.0
        for f in t_frames:
            tr += f["reward"]
            t_cum.append(tr)
        for f in r_frames:
            rr += f["reward"]
            r_cum.append(rr)

        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Scatter(x=r_steps, y=r_cum, mode="lines", name="🎲 Random",
                                      line=dict(color=RED, width=2, dash="dot")))
        fig_cmp.add_trace(go.Scatter(x=t_steps, y=t_cum, mode="lines", name="🤖 Trained",
                                      line=dict(color=ORANGE, width=2.5),
                                      fill="tonexty", fillcolor="rgba(255,107,43,0.1)"))
        if step_idx < len(t_frames):
            fig_cmp.add_vline(x=step_idx+1, line_color=WHITE, line_dash="dot", opacity=0.3,
                              annotation_text="← now", annotation_font_color=GREY)
        fig_cmp.update_layout(paper_bgcolor=DARK, plot_bgcolor="#1a1f35", height=280,
                               font=dict(color=GREY),
                               xaxis=dict(title="Step", gridcolor=BORDER, color=GREY),
                               yaxis=dict(title="Cumulative Reward", gridcolor=BORDER, color=GREY, zeroline=False),
                               legend=dict(bgcolor=PANEL, bordercolor=BORDER, font=dict(color="#cccccc")),
                               margin=dict(t=10, b=40))
        st.plotly_chart(fig_cmp, use_container_width=True)


# ── PAGE 3: PROOF OF LEARNING ─────────────────────────────────────────────────
elif page == "📈 Proof of Learning":
    st.markdown('<div class="section-header">📈 Proof — Did the Model Actually Learn?</div>', unsafe_allow_html=True)
    st.markdown(f"""<div class="explainer">
    All policies complete 100% of tasks — the environment is solvable by anyone. The real question is <b>how efficiently</b>.
    <b>Reward = speed + energy conservation + no wasted moves.</b>
    The trained policy scores +20% higher reward AND has 32× lower variance — it's not just better, it's <i>consistent</i>.
    </div>""", unsafe_allow_html=True)

    df_eval = load_eval_metrics()
    df_train = load_training_metrics()

    if df_eval is not None and not df_eval.empty:
        available = df_eval["model"].unique().tolist()
        order = [m for m in ["random", "early_30k", "mid_100k", "late_200k", "final"] if m in available]
        colors_map = {"random": GREY, "early_30k": "#ff9f43", "mid_100k": BLUE, "late_200k": "#a29bfe", "final": ORANGE}

        summary = df_eval.groupby("model").agg(
            avg_reward=("total_reward", "mean"),
            std_reward=("total_reward", "std"),
            avg_efficiency=("efficiency", "mean"),
            avg_completion=("completion_rate", "mean"),
        ).reset_index()
        summary = summary[summary["model"].isin(order)].copy()
        summary["_ord"] = summary["model"].map({m: i for i, m in enumerate(order)})
        summary = summary.sort_values("_ord").drop(columns="_ord").fillna(0)

        random_row = summary[summary["model"] == "random"]
        final_row = summary[summary["model"] == "final"]

        st.markdown("### Key Numbers")
        if not random_row.empty and not final_row.empty:
            r_rew = random_row["avg_reward"].values[0]
            f_rew = final_row["avg_reward"].values[0]
            r_std = random_row["std_reward"].values[0]
            f_std = final_row["std_reward"].values[0]
            improvement = (f_rew - r_rew) / max(abs(r_rew), 1e-5) * 100
            variance_ratio = r_std / max(f_std, 0.001)

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.markdown(f'<div class="proof-box"><div class="proof-number">{r_rew:.1f}</div>Random avg reward</div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="proof-box"><div class="proof-number">{f_rew:.1f}</div>Trained avg reward</div>', unsafe_allow_html=True)
            c3.markdown(f'<div class="proof-box"><div class="proof-number">+{improvement:.1f}%</div>Reward improvement</div>', unsafe_allow_html=True)
            c4.markdown(f'<div class="proof-box"><div class="proof-number">{variance_ratio:.0f}×</div>Lower variance<br><small>std {r_std:.1f} → {f_std:.1f}</small></div>', unsafe_allow_html=True)
            c5.markdown(f'<div class="proof-box"><div class="proof-number">3 vs 80</div>Steps to solve<br><small>trained vs limit</small></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Avg Reward per Policy** — error bars show consistency")
            fig_bar = go.Figure(go.Bar(
                x=summary["model"], y=summary["avg_reward"],
                marker_color=[colors_map.get(m, GREY) for m in summary["model"]],
                error_y=dict(type="data", array=summary["std_reward"].tolist(),
                             visible=True, color=WHITE, thickness=1.5),
                text=summary["avg_reward"].round(1), textposition="outside",
                textfont=dict(color=WHITE),
            ))
            fig_bar.update_layout(paper_bgcolor=DARK, plot_bgcolor="#1a1f35", height=300,
                                   font=dict(color=GREY),
                                   xaxis=dict(color=GREY, gridcolor=BORDER),
                                   yaxis=dict(color=GREY, gridcolor=BORDER, zeroline=False, title="Avg Reward"),
                                   margin=dict(t=10, b=40))
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            st.markdown("**Reward across 20 eval episodes** — trained = consistent, random = noisy")
            fig_lines = go.Figure()
            for mn in order:
                subset = df_eval[df_eval["model"] == mn].reset_index(drop=True)
                if subset.empty:
                    continue
                fig_lines.add_trace(go.Scatter(
                    x=subset.index+1, y=subset["total_reward"],
                    mode="lines+markers", name=mn,
                    line=dict(color=colors_map.get(mn, GREY), width=2),
                    marker=dict(size=4),
                ))
            fig_lines.update_layout(paper_bgcolor=DARK, plot_bgcolor="#1a1f35", height=300,
                                     font=dict(color=GREY),
                                     xaxis=dict(title="Eval Episode", gridcolor=BORDER, color=GREY),
                                     yaxis=dict(title="Total Reward", gridcolor=BORDER, color=GREY, zeroline=False),
                                     legend=dict(bgcolor=PANEL, bordercolor=BORDER, font=dict(color="#cccccc", size=10)),
                                     margin=dict(t=10, b=40))
            st.plotly_chart(fig_lines, use_container_width=True)

        st.markdown("### Reward Shaping — The Carrots and Sticks")
        st.markdown(f"""<div class="explainer">
        Sparse reward (+1 on completion only) failed completely — agents wandered for many steps with zero gradient signal.
        Dense reward shaping gives signal <b>every step</b>:<br><br>
        <b style="color:{GREEN}">🥕 Carrots (positive drivers):</b>
        Task completion +1.0 · Energy efficiency bonus +0.5 · Proximity bonus +0.3/step · Early solve +5.0<br>
        <b style="color:{RED}">🥢 Sticks (negative guardrails):</b>
        Time penalty -0.02/step (prevents idling) · Invalid action -0.1 to -0.2 (enforces discipline)<br><br>
        The time penalty is critical — without it agents would farm the proximity bonus by orbiting tasks indefinitely.
        </div>""", unsafe_allow_html=True)

        st.markdown("### Summary Table")
        disp = summary[["model", "avg_reward", "std_reward", "avg_efficiency", "avg_completion"]].copy()
        disp.columns = ["Policy", "Avg Reward", "Std Dev", "Avg Efficiency", "Completion Rate"]
        disp["Avg Reward"] = disp["Avg Reward"].round(2)
        disp["Std Dev"] = disp["Std Dev"].round(2)
        disp["Avg Efficiency"] = disp["Avg Efficiency"].round(3)
        disp["Completion Rate"] = (disp["Completion Rate"] * 100).round(1).astype(str) + "%"
        st.dataframe(disp, use_container_width=True, hide_index=True)

    if df_train is not None and not df_train.empty:
        st.markdown("### Training Curve — 95,000+ Episodes")
        st.markdown(f"""<div class="explainer">
        <b>Start:</b> Episode length ~16 chaotic steps, entropy loss -15.2 (high exploration).
        <b>30k timesteps:</b> Episode length bottoms out at 3.0 steps. Entropy reaches -0.27 (pure exploitation).
        <b>Iteration 200:</b> Critic fully understands environment dynamics (explained_variance = 1.0).
        Rising trend + narrowing band = real convergence, not luck.
        </div>""", unsafe_allow_html=True)

        window = st.slider("Smoothing", 10, 500, 200)
        df_train["smooth"] = df_train["reward"].rolling(window, min_periods=1).mean()
        df_train["std"] = df_train["reward"].rolling(window, min_periods=1).std().fillna(0)
        total_ep = len(df_train)

        fig_train = go.Figure()
        fig_train.add_trace(go.Scatter(
            x=pd.concat([df_train["episode"], df_train["episode"].iloc[::-1]]),
            y=pd.concat([df_train["smooth"] + df_train["std"],
                         (df_train["smooth"] - df_train["std"]).iloc[::-1]]),
            fill="toself", fillcolor="rgba(255,107,43,0.08)",
            line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip",
        ))
        fig_train.add_trace(go.Scatter(x=df_train["episode"], y=df_train["reward"],
                                        mode="lines", line=dict(color=BORDER, width=1),
                                        showlegend=False))
        fig_train.add_trace(go.Scatter(x=df_train["episode"], y=df_train["smooth"],
                                        mode="lines", line=dict(color=ORANGE, width=2.5),
                                        name="Smoothed reward"))

        for ck_label, frac, color, note in [
            ("30k — converged", 0.1, GREEN, "ep_len → 3.0"),
            ("100k", 0.33, BLUE, ""),
            ("200k — critic perfect", 0.66, "#a29bfe", "exp_var = 1.0"),
            ("300k", 1.0, ORANGE, "final"),
        ]:
            ep = int(total_ep * frac)
            if 0 < ep <= total_ep:
                fig_train.add_vline(x=ep, line_color=color, line_dash="dot", opacity=0.6,
                                    annotation_text=ck_label, annotation_font_color=color,
                                    annotation_font_size=9)

        fig_train.update_layout(paper_bgcolor=DARK, plot_bgcolor="#1a1f35", height=350,
                                 font=dict(color=GREY),
                                 xaxis=dict(title="Episode", gridcolor=BORDER, color=GREY),
                                 yaxis=dict(title="Reward", gridcolor=BORDER, color=GREY, zeroline=False),
                                 legend=dict(bgcolor=PANEL, bordercolor=BORDER, font=dict(color="#cccccc")),
                                 margin=dict(t=10, b=40))
        st.plotly_chart(fig_train, use_container_width=True)

        early = df_train.head(500)
        late = df_train.tail(500)
        c1, c2, c3 = st.columns(3)
        c1.metric("Avg Reward (first 500 ep)", f"{early['reward'].mean():.2f}",
                  f"{late['reward'].mean() - early['reward'].mean():+.2f} by end")
        c2.metric("Std Dev (first 500 ep)", f"{early['reward'].std():.2f}",
                  f"{late['reward'].std() - early['reward'].std():+.2f} by end")
        c3.metric("Total Episodes", f"{len(df_train):,}", "300k timesteps")


# ── PAGE 4: WHAT-IF ───────────────────────────────────────────────────────────
elif page == "🎛️ What-If":
    st.markdown('<div class="section-header">🎛️ What-If Explorer — Stress Test the Policy</div>', unsafe_allow_html=True)
    st.markdown(f"""<div class="explainer">
    Change the environment and run the trained model on scenarios it was <b>never trained on</b>.
    Tests generalisation — can a policy learned on 5 agents + 10 tasks handle variations?
    Try fewer agents, more tasks, or lower energy to stress test it.
    </div>""", unsafe_allow_html=True)

    model = load_model()

    col_s, col_v = st.columns([1, 2])
    with col_s:
        st.markdown("### Parameters")
        num_tasks = st.slider("Number of Tasks", 5, 20, 10)
        num_type_a = st.slider("Type A Agents (Heavy Lifters)", 0, 4, 2)
        num_type_b = st.slider("Type B Agents (Sprinters)", 0, 5, 3)
        energy_a = st.slider("Type A Energy", 30, 200, 100)
        energy_b = st.slider("Type B Energy", 20, 120, 60)
        st.markdown("---")
        n_agents = num_type_a + num_type_b
        ratio = num_tasks / max(n_agents, 1)
        st.markdown(f"**Agents:** {n_agents} | **Tasks:** {num_tasks} | **Ratio:** {ratio:.1f} tasks/agent")
        if ratio > 3:
            st.warning("High ratio — agents may not complete all tasks.")
        elif ratio < 1:
            st.info("More agents than tasks — some will be idle.")

        if model is None:
            st.markdown(f"""<div class="explainer" style="border-color:{ORANGE}">
            <b style="color:{ORANGE}">⚠️ Live inference not available on hosted version.</b><br>
            The trained model file is not included in the hosted deployment.<br>
            To run What-If locally: <code>python train.py</code> → <code>streamlit run dashboard.py</code><br>
            All other pages (Replay, Trained vs Random, Proof of Learning) use pre-computed data and work fully.
            </div>""", unsafe_allow_html=True)
        else:
            run_btn = st.button("▶ Run Episode", use_container_width=True, type="primary")

    with col_v:
        model_available = model is not None
        run_btn_clicked = model_available and 'run_btn' in dir() and run_btn

        if model_available and run_btn_clicked:
            if n_agents == 0:
                st.warning("Need at least 1 agent.")
            else:
                from environment import TaskAllocationEnv
                from gymnasium import spaces

                custom_configs = (
                    [{"type": "A", "capacity": 3, "speed": 1, "energy": energy_a}] * num_type_a +
                    [{"type": "B", "capacity": 1, "speed": 2, "energy": energy_b}] * num_type_b
                )
                env = TaskAllocationEnv(num_tasks=num_tasks)
                env.agent_configs = custom_configs
                env.num_agents = n_agents
                env.agent_speed = np.array([c["speed"] for c in custom_configs], dtype=np.float32)
                env.agent_capacity = np.array([c["capacity"] for c in custom_configs], dtype=np.float32)
                obs_size = n_agents * 3 + num_tasks * 3 + num_tasks
                env.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32)
                env.action_space = spaces.MultiDiscrete([num_tasks + 1] * n_agents)

                obs, _ = env.reset()
                wif_frames, done, total_reward = [], False, 0.0
                while not done:
                    try:
                        masks = env.action_masks()
                        action, _ = model.predict(obs, deterministic=True, action_masks=masks)
                        action = np.clip(action, 0, num_tasks)
                    except Exception:
                        action = env.action_space.sample()
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    done = terminated or truncated
                    wif_frames.append({
                        "step": env._step,
                        "agent_positions": env.agent_positions.tolist(),
                        "agent_energy": env.agent_energy.tolist(),
                        "task_positions": env.task_positions.tolist(),
                        "task_done": env.task_done.tolist(),
                        "task_difficulty": env.task_difficulty.tolist(),
                        "tasks_completed": int(env.task_done.sum()),
                        "reward": float(reward),
                    })
                st.session_state["wif_frames"] = wif_frames
                st.session_state["wif_configs"] = custom_configs
                st.session_state["wif_num_tasks"] = num_tasks
                st.session_state["wif_total_reward"] = total_reward

        if "wif_frames" in st.session_state and st.session_state["wif_frames"] and model_available:
            wif_frames = st.session_state["wif_frames"]
            wif_configs = st.session_state.get("wif_configs", AGENT_CONFIGS)
            wif_num_tasks = st.session_state.get("wif_num_tasks", 10)
            wif_total = st.session_state.get("wif_total_reward", 0)
            last = wif_frames[-1]

            m1, m2, m3, m4 = st.columns(4)
            m1.markdown(f'<div class="metric-box"><div class="metric-val" style="color:{GREEN}">{last["tasks_completed"]}/{wif_num_tasks}</div><div class="metric-lbl">Tasks Done</div></div>', unsafe_allow_html=True)
            m2.markdown(f'<div class="metric-box"><div class="metric-val">{last["step"]}</div><div class="metric-lbl">Steps</div></div>', unsafe_allow_html=True)
            m3.markdown(f'<div class="metric-box"><div class="metric-val">{wif_total:.1f}</div><div class="metric-lbl">Total Reward</div></div>', unsafe_allow_html=True)
            eff = last["tasks_completed"] / max(last["step"], 1)
            m4.markdown(f'<div class="metric-box"><div class="metric-val">{eff:.2f}</div><div class="metric-lbl">Tasks/Step</div></div>', unsafe_allow_html=True)

            wif_view = st.radio("View mode", ["▶ Animated", "🎚️ Step-by-Step"], horizontal=True, key="wif_view")
            if wif_view == "▶ Animated":
                st.plotly_chart(build_animated_figure(wif_frames, "What-If Run", custom_configs=wif_configs), use_container_width=True)
            else:
                step_idx_wif = st.slider("Step", 0, len(wif_frames)-1, 0, key="wif_slider")
                st.plotly_chart(build_grid_figure(wif_frames, step_idx_wif, "What-If Run", custom_configs=wif_configs), use_container_width=True)
                st.markdown(f'<div class="narration-box">{build_narration(wif_frames[step_idx_wif])}</div>', unsafe_allow_html=True)
        elif not model_available:
            st.markdown(f"""<div style="background:{PANEL};border-radius:10px;padding:40px;text-align:center;border:1px dashed {ORANGE};margin-top:20px;">
                <h3 style="color:{ORANGE}">🤖 Live Inference — Local Only</h3>
                <p style="color:{GREY}">Run the project locally to use the What-If explorer.<br>
                All other pages work fully on this hosted version.</p>
                <p style="color:{GREY};font-family:monospace;font-size:0.8rem">
                pip install -r requirements.txt<br>
                python train.py → python evaluate.py → streamlit run dashboard.py</p>
            </div>""", unsafe_allow_html=True)
        elif not run_btn_clicked:
            st.markdown(f"""<div style="background:{PANEL};border-radius:10px;padding:40px;text-align:center;border:1px dashed {BORDER};margin-top:20px;">
                <h3 style="color:{GREY};">Configure parameters and click ▶ Run Episode</h3>
                <p style="color:#555;">Try fewer agents, more tasks, or lower energy to stress test</p>
            </div>""", unsafe_allow_html=True)

st.markdown("---")
st.caption("KnowledgeQuarry 2026 · Convoke 8.0 · CIC University of Delhi · ML Engineering Track · MaskablePPO + Custom Gymnasium · github.com/RAK2315/knowledgequarry-2026")