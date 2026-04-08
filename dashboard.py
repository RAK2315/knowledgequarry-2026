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
AGENT_COLORS = ["#4a9eff", "#4a9eff", "#ff6b6b", "#ff6b6b", "#ff6b6b"]
AGENT_SYMBOLS = ["square", "square", "circle", "circle", "circle"]
RANDOM_AVG_REWARD = 21.92

st.markdown("""
<style>
.block-container { padding-top: 1rem; padding-bottom: 0.5rem; }
.section-header {
    background: linear-gradient(90deg, #1a1f35, #0e1117);
    border-left: 4px solid #00d4ff;
    padding: 8px 14px; border-radius: 4px;
    color: #ffffff; font-size: 1rem; font-weight: 600;
    margin-bottom: 0.8rem;
}
.explainer {
    background: #1e2130; border-radius: 8px;
    padding: 10px 14px; color: #cccccc;
    font-size: 0.85rem; margin-bottom: 0.8rem;
    border: 1px solid #2d3250; line-height: 1.6;
}
.metric-box {
    background: #1e2130; border-radius: 8px;
    padding: 10px; text-align: center;
    border: 1px solid #2d3250; margin-bottom: 8px;
}
.metric-val { font-size: 1.6rem; font-weight: bold; color: #00d4ff; }
.metric-lbl { font-size: 0.72rem; color: #888888; margin-top: 2px; }
.narration-box {
    background: #12161f; border-radius: 8px;
    padding: 12px 16px; border: 1px solid #00d4ff;
    color: #ffffff; font-size: 0.85rem;
    line-height: 1.8; min-height: 100px;
}
.narration-title { color: #00d4ff; font-weight: bold; margin-bottom: 6px; font-size: 0.8rem; }
.log-complete { color: #00ff88; }
.log-move { color: #4a9eff; }
.log-idle { color: #888888; }
.proof-box {
    background: #1a2e1a; border: 1px solid #00ff88;
    border-radius: 8px; padding: 12px 16px;
    color: #cccccc; font-size: 0.85rem;
}
.proof-number { color: #00ff88; font-size: 1.4rem; font-weight: bold; }
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


def get_agent_configs_for_frames(frames):
    if not frames:
        return AGENT_CONFIGS
    n = len(frames[0]["agent_positions"])
    if n == len(AGENT_CONFIGS):
        return AGENT_CONFIGS
    configs = []
    for i in range(n):
        if i < len(AGENT_CONFIGS):
            configs.append(AGENT_CONFIGS[i])
        else:
            configs.append({"type": "B", "capacity": 1, "speed": 2, "energy": 60})
    return configs


def get_colors_symbols(n, custom_configs=None):
    colors, symbols = [], []
    for i in range(n):
        cfg = custom_configs[i] if custom_configs and i < len(custom_configs) else (AGENT_CONFIGS[i] if i < len(AGENT_CONFIGS) else {"type": "B"})
        if cfg["type"] == "A":
            colors.append("#4a9eff")
            symbols.append("square")
        else:
            colors.append("#ff6b6b")
            symbols.append("circle")
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
    n_agents = len(ap)
    configs = custom_configs if custom_configs else get_agent_configs_for_frames(frames)
    colors, symbols = get_colors_symbols(n_agents, configs)

    positions_used = {}
    offsets = []
    for i in range(n_agents):
        key = (round(ap[i, 0]), round(ap[i, 1]))
        count = positions_used.get(key, 0)
        positions_used[key] = count + 1
        offsets.append(count * 0.3)

    data = []
    task_colors = ["#00ff88" if not d else "#333a4f" for d in td]
    task_sizes = [12 + int(d) * 5 for d in diff]
    task_hover = [
        f"<b>Task {i}</b><br>Position: ({int(task_pos[i,0])},{int(task_pos[i,1])})<br>"
        f"Difficulty: {int(diff[i])}/3<br>{'✅ Done' if td[i] else '⏳ Pending'}"
        for i in range(len(td))
    ]
    data.append(go.Scatter(
        x=task_pos[:, 0], y=task_pos[:, 1],
        mode="markers",
        marker=dict(color=task_colors, size=task_sizes, symbol="diamond",
                    line=dict(color="#ffffff", width=1)),
        hovertext=task_hover, hoverinfo="text",
        name="Tasks", showlegend=True,
    ))

    for i in range(n_agents):
        cfg = configs[i] if i < len(configs) else {"type": "?", "capacity": 1, "speed": 1, "energy": 100}
        energy_pct = ae[i] / cfg["energy"] if cfg["energy"] > 0 else 0
        border_color = "#00ff88" if energy_pct > 0.5 else "#ff9f43" if energy_pct > 0.2 else "#ff6b6b"
        hover = (
            f"<b>Agent {i} (Type {cfg['type']})</b><br>"
            f"Energy: {ae[i]:.1f}/{cfg['energy']} ({energy_pct*100:.0f}%)<br>"
            f"Speed: {cfg['speed']} | Capacity: {cfg['capacity']}<br>"
            f"{'Strong & slow — handles hard tasks' if cfg['type']=='A' else 'Fast & light — handles nearby tasks'}"
        )
        data.append(go.Scatter(
            x=[ap[i, 0] + offsets[i]], y=[ap[i, 1] + offsets[i]],
            mode="markers+text",
            marker=dict(color=colors[i], size=22, symbol=symbols[i],
                        line=dict(color=border_color, width=3)),
            text=[f"{'A' if cfg['type']=='A' else 'B'}{i}"],
            textposition="top center",
            textfont=dict(size=9, color="#ffffff"),
            hovertext=hover, hoverinfo="text",
            name=f"Agent {i} (Type {cfg['type']})", showlegend=True,
        ))

    tasks_done = int(td.sum())
    num_t = len(td)
    fig = go.Figure(data=data)
    fig.update_layout(
        title=dict(
            text=f"{title} | Step {frame['step']} | <b>{tasks_done}/{num_t} tasks done</b>",
            font=dict(color="#ffffff", size=13)
        ),
        paper_bgcolor="#0e1117", plot_bgcolor="#1a1f35",
        xaxis=dict(range=[-0.5, GRID_SIZE - 0.5], showgrid=True, gridcolor="#2d3250",
                   title="Grid X", color="#aaaaaa", zeroline=False, dtick=1, fixedrange=True),
        yaxis=dict(range=[-0.5, GRID_SIZE - 0.5], showgrid=True, gridcolor="#2d3250",
                   title="Grid Y", color="#aaaaaa", zeroline=False, dtick=1, fixedrange=True),
        legend=dict(bgcolor="#1e2130", bordercolor="#2d3250",
                    font=dict(color="#cccccc", size=10), x=1.01, y=1),
        height=420, margin=dict(l=40, r=160, t=50, b=40),
    )
    return fig


def build_animated_figure(frames, title="", custom_configs=None):
    if not frames:
        return go.Figure()

    task_pos = np.array(frames[0]["task_positions"])
    n_agents = len(frames[0]["agent_positions"])
    configs = custom_configs if custom_configs else get_agent_configs_for_frames(frames)
    colors, symbols = get_colors_symbols(n_agents, configs)
    fig_frames = []

    for frame in frames:
        ap = np.array(frame["agent_positions"])
        td = np.array(frame["task_done"])
        diff = np.array(frame["task_difficulty"])
        ae = np.array(frame["agent_energy"])

        positions_used = {}
        offsets = []
        for i in range(n_agents):
            key = (round(ap[i, 0]), round(ap[i, 1]))
            count = positions_used.get(key, 0)
            positions_used[key] = count + 1
            offsets.append(count * 0.3)

        data = []
        task_colors = ["#00ff88" if not d else "#333a4f" for d in td]
        task_hover = [
            f"<b>Task {i}</b><br>Difficulty: {int(diff[i])}/3<br>{'✅ Done' if td[i] else '⏳ Pending'}"
            for i in range(len(td))
        ]
        data.append(go.Scatter(
            x=task_pos[:, 0], y=task_pos[:, 1],
            mode="markers",
            marker=dict(color=task_colors, size=[12 + int(d) * 5 for d in diff],
                        symbol="diamond", line=dict(color="#ffffff", width=1)),
            hovertext=task_hover, hoverinfo="text",
            name="Tasks", showlegend=True,
        ))
        for i in range(n_agents):
            cfg = configs[i] if i < len(configs) else {"type": "?", "capacity": 1, "speed": 1, "energy": 100}
            energy_pct = ae[i] / cfg["energy"] if cfg["energy"] > 0 else 0
            border_color = "#00ff88" if energy_pct > 0.5 else "#ff9f43" if energy_pct > 0.2 else "#ff6b6b"
            data.append(go.Scatter(
                x=[ap[i, 0] + offsets[i]], y=[ap[i, 1] + offsets[i]],
                mode="markers+text",
                marker=dict(color=colors[i], size=22, symbol=symbols[i],
                            line=dict(color=border_color, width=3)),
                text=[f"{'A' if cfg['type']=='A' else 'B'}{i}"],
                textposition="top center",
                textfont=dict(size=9, color="#ffffff"),
                hovertext=f"Agent {i} (Type {cfg['type']}) | Energy: {ae[i]:.1f}/{cfg['energy']}",
                hoverinfo="text",
                name=f"Agent {i} (Type {cfg['type']})", showlegend=True,
            ))

        tasks_done = int(td.sum())
        num_t = len(td)
        fig_frames.append(go.Frame(
            data=data, name=str(frame["step"]),
            layout=go.Layout(
                title_text=f"{title} | Step {frame['step']} | {tasks_done}/{num_t} tasks done"
            )
        ))

    fig = go.Figure(
        data=fig_frames[0].data if fig_frames else [],
        frames=fig_frames,
        layout=go.Layout(
            title=dict(text=title, font=dict(color="#ffffff", size=13)),
            paper_bgcolor="#0e1117", plot_bgcolor="#1a1f35",
            xaxis=dict(range=[-0.5, GRID_SIZE - 0.5], showgrid=True, gridcolor="#2d3250",
                       title="Grid X", color="#aaaaaa", zeroline=False, dtick=1),
            yaxis=dict(range=[-0.5, GRID_SIZE - 0.5], showgrid=True, gridcolor="#2d3250",
                       title="Grid Y", color="#aaaaaa", zeroline=False, dtick=1),
            legend=dict(bgcolor="#1e2130", bordercolor="#2d3250",
                        font=dict(color="#cccccc", size=10), x=1.01, y=1),
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
                currentvalue=dict(prefix="Step: ", font=dict(color="#ffffff", size=12)),
                bgcolor="#1e2130", bordercolor="#2d3250",
                font=dict(color="#aaaaaa"),
            )],
        )
    )
    return fig


def build_narration(frame, policy_label=""):
    if not frame or "action_logs" not in frame:
        return "<i style='color:#888'>No action data. Re-run evaluate.py.</i>"
    step = frame["step"]
    tasks_done = frame["tasks_completed"]
    summary = frame.get("step_summary", f"Step {step}")
    logs = frame["action_logs"]
    lines = [f'<div class="narration-title">📍 {summary}</div>']
    for log in logs:
        if "✅" in log:
            lines.append(f'<span class="log-complete">🟢 {log}</span><br>')
        elif "idle" in log:
            lines.append(f'<span class="log-idle">⚪ {log}</span><br>')
        else:
            lines.append(f'<span class="log-move">🔵 {log}</span><br>')
    num_t = frame.get("tasks_completed", 0)
    total_t = len(frame.get("task_done", [0] * NUM_TASKS))
    if num_t == total_t and total_t > 0:
        lines.append(f'<br><b style="color:#00ff88">✅ All {total_t} tasks completed in {step} steps!</b>')
    return "".join(lines)


def build_reward_chart(frames_so_far, random_avg_total):
    if not frames_so_far:
        return go.Figure()
    steps = [f["step"] for f in frames_so_far]
    cum_rewards = []
    running = 0
    for f in frames_so_far:
        running += f["reward"]
        cum_rewards.append(running)
    max_steps = max(steps) if steps else 1
    random_line = [random_avg_total * (s / max_steps) for s in steps]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=steps, y=random_line, mode="lines", name="Random (avg)",
        line=dict(color="#888888", width=2, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=steps, y=cum_rewards, mode="lines+markers", name="This policy",
        line=dict(color="#00d4ff", width=2.5),
        marker=dict(size=5, color="#00d4ff"),
        fill="tonexty", fillcolor="rgba(0,212,255,0.06)",
    ))
    if len(steps) > 1:
        gap = cum_rewards[-1] - random_line[-1]
        fig.add_annotation(
            x=steps[-1], y=cum_rewards[-1],
            text=f"+{gap:.1f} vs random",
            showarrow=True, arrowhead=2,
            font=dict(color="#00ff88", size=10),
            arrowcolor="#00ff88", ax=-50, ay=-25,
        )
    fig.update_layout(
        paper_bgcolor="#0e1117", plot_bgcolor="#1a1f35", height=200,
        font=dict(color="#aaaaaa", size=10),
        xaxis=dict(title="Step", gridcolor="#2d3250", color="#aaaaaa", zeroline=False),
        yaxis=dict(title="Cumulative Reward", gridcolor="#2d3250", color="#aaaaaa", zeroline=False),
        legend=dict(bgcolor="#1e2130", bordercolor="#2d3250", font=dict(color="#cccccc", size=9)),
        margin=dict(l=50, r=20, t=10, b=35),
    )
    return fig


# ── HEADER ──────────────────────────────────────────────────────────────
st.markdown("## 🤖 Adaptive Multi-Agent Task Allocation")
st.markdown("""<div class="explainer">
<b>What is this?</b> 5 agents must complete 10 tasks on a grid. Agents have different capabilities —
<b style="color:#4a9eff">■ Blue squares = Type A</b> (strong & slow, capacity 3, 100 energy — best for hard tasks) and
<b style="color:#ff6b6b">● Red circles = Type B</b> (fast & weak, capacity 1, 60 energy — best for nearby easy tasks).
<b>🟢 Green diamond = pending task</b> (bigger = harder). <b>⚫ Dark = completed</b>.
The RL model (MaskablePPO) learned <i>who goes where</i> through 95,000+ episodes of trial and error — no rules, no hardcoding.
</div>""", unsafe_allow_html=True)

page = st.radio("", ["🎬 Episode Replay", "⚔️ Trained vs Random", "📈 Proof of Learning", "🎛️ What-If"],
                horizontal=True, label_visibility="collapsed")
st.markdown("---")


# ── PAGE 1: EPISODE REPLAY ───────────────────────────────────────────────
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
                if view_type == "🎚️ Step-by-Step":
                    step_idx = st.slider("Step", 0, len(frames) - 1, 0)
                    frame = frames[step_idx]
                else:
                    step_idx = len(frames) - 1
                    frame = frames[step_idx]

                st.markdown("---")
                tasks_done = frame["tasks_completed"]
                total_r = sum(f["reward"] for f in frames[:step_idx + 1])
                energy_left = sum(frame["agent_energy"])
                total_energy = sum(cfg["energy"] for cfg in AGENT_CONFIGS)

                st.markdown(f'<div class="metric-box"><div class="metric-val" style="color:#00ff88">{tasks_done}/{NUM_TASKS}</div><div class="metric-lbl">Tasks Done</div></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-box"><div class="metric-val">{frame["step"]}</div><div class="metric-lbl">Current Step</div></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-box"><div class="metric-val">{total_r:.1f}</div><div class="metric-lbl">Reward So Far</div></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-box"><div class="metric-val" style="color:#ff9f43">{energy_left:.0f}/{total_energy}</div><div class="metric-lbl">Energy Remaining</div></div>', unsafe_allow_html=True)

                st.markdown("---")
                st.markdown("**Agent Energy**")
                for i, e in enumerate(frame["agent_energy"]):
                    cfg = AGENT_CONFIGS[i] if i < len(AGENT_CONFIGS) else {"type": "?", "energy": 100}
                    pct = max(0.0, min(1.0, e / cfg["energy"]))
                    icon = "🟢" if pct > 0.5 else "🟡" if pct > 0.2 else "🔴"
                    st.progress(float(pct), text=f"{icon} A{i} (Type {cfg['type']}): {e:.0f}/{cfg['energy']}")

        with col_right:
            if frames:
                label = ("early (30k steps)" if "30k" in policy else
                         "mid (100k)" if "100k" in policy else
                         "late (200k)" if "200k" in policy else "final (trained)")

                if view_type == "▶ Animated":
                    fig = build_animated_figure(frames, f"Policy: {label}")
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown("**📋 Full Episode Log — expand any step**")
                    for fr in frames:
                        step_label = fr.get("step_summary", f"Step {fr['step']}")
                        with st.expander(f"**{step_label}**", expanded=(fr["step"] <= 1)):
                            narr = build_narration(fr, policy)
                            st.markdown(f'<div class="narration-box">{narr}</div>', unsafe_allow_html=True)
                else:
                    fig = build_grid_figure(frames, step_idx, f"Policy: {label}")
                    st.plotly_chart(fig, use_container_width=True)

                    col_narr, col_chart = st.columns([1, 1])
                    with col_narr:
                        st.markdown("**📋 What happened this step**")
                        narr = build_narration(frames[step_idx], policy)
                        st.markdown(f'<div class="narration-box">{narr}</div>', unsafe_allow_html=True)
                    with col_chart:
                        st.markdown("**📈 Reward vs Random Baseline**")
                        st.markdown('<div class="explainer" style="font-size:0.78rem">Blue = policy reward accumulating step by step. Dotted grey = random avg. <b>Gap = what the model learned.</b></div>', unsafe_allow_html=True)
                        fig_r = build_reward_chart(frames[:step_idx + 1], RANDOM_AVG_REWARD)
                        st.plotly_chart(fig_r, use_container_width=True)


# ── PAGE 2: TRAINED VS RANDOM ────────────────────────────────────────────
elif page == "⚔️ Trained vs Random":
    comparison = load_comparison_data()
    if not comparison:
        st.error("Run `python evaluate.py` first.")
    else:
        st.markdown('<div class="section-header">⚔️ Same Grid. Same Tasks. Different Brain.</div>', unsafe_allow_html=True)
        st.markdown("""<div class="explainer">
        Both policies start with <b>identical agent positions and identical task layouts</b> (same seed).
        Only the decision-making differs. Watch how the trained policy spreads out cleanly
        while random stumbles, overlaps, and wastes energy.
        </div>""", unsafe_allow_html=True)

        t_frames = comparison["trained"]
        r_frames = comparison["random"]
        t_sum = comparison["trained_summary"]
        r_sum = comparison["random_summary"]

        max_step = max(len(t_frames), len(r_frames))
        step_idx = st.slider("Step (controls both)", 0, max_step - 1, 0, key="cmp_slider")

        col_t, col_r = st.columns(2)

        with col_t:
            st.markdown("### 🤖 Trained Policy (PPO)")
            t_idx = min(step_idx, len(t_frames) - 1)
            tf = t_frames[t_idx]
            t_reward = sum(f["reward"] for f in t_frames[:t_idx + 1])
            m1, m2, m3 = st.columns(3)
            m1.markdown(f'<div class="metric-box"><div class="metric-val" style="color:#00ff88">{tf["tasks_completed"]}/{NUM_TASKS}</div><div class="metric-lbl">Tasks Done</div></div>', unsafe_allow_html=True)
            m2.markdown(f'<div class="metric-box"><div class="metric-val" style="color:#00ff88">{tf["step"]}</div><div class="metric-lbl">Step</div></div>', unsafe_allow_html=True)
            m3.markdown(f'<div class="metric-box"><div class="metric-val" style="color:#00ff88">{t_reward:.1f}</div><div class="metric-lbl">Reward</div></div>', unsafe_allow_html=True)
            fig_t = build_grid_figure(t_frames, t_idx, "Trained")
            st.plotly_chart(fig_t, use_container_width=True)
            narr_t = build_narration(tf, "trained")
            st.markdown(f'<div class="narration-box">{narr_t}</div>', unsafe_allow_html=True)

        with col_r:
            st.markdown("### 🎲 Random Policy (no learning)")
            r_idx = min(step_idx, len(r_frames) - 1)
            rf = r_frames[r_idx]
            r_reward = sum(f["reward"] for f in r_frames[:r_idx + 1])
            m1, m2, m3 = st.columns(3)
            m1.markdown(f'<div class="metric-box"><div class="metric-val" style="color:#ff6b6b">{rf["tasks_completed"]}/{NUM_TASKS}</div><div class="metric-lbl">Tasks Done</div></div>', unsafe_allow_html=True)
            m2.markdown(f'<div class="metric-box"><div class="metric-val" style="color:#ff6b6b">{rf["step"]}</div><div class="metric-lbl">Step</div></div>', unsafe_allow_html=True)
            m3.markdown(f'<div class="metric-box"><div class="metric-val" style="color:#ff6b6b">{r_reward:.1f}</div><div class="metric-lbl">Reward</div></div>', unsafe_allow_html=True)
            fig_r = build_grid_figure(r_frames, r_idx, "Random")
            st.plotly_chart(fig_r, use_container_width=True)
            narr_r = build_narration(rf, "random")
            st.markdown(f'<div class="narration-box">{narr_r}</div>', unsafe_allow_html=True)

        st.markdown("### 📈 Cumulative Reward — Step by Step (same scenario)")
        st.markdown('<div class="explainer">Every point where green is above red = a smarter decision by the trained model. The shaded gap = total learning improvement on this run.</div>', unsafe_allow_html=True)

        t_steps = [f["step"] for f in t_frames]
        r_steps = [f["step"] for f in r_frames]
        t_cum, r_cum = [], []
        tr, rr = 0.0, 0.0
        for f in t_frames:
            tr += f["reward"]
            t_cum.append(tr)
        for f in r_frames:
            rr += f["reward"]
            r_cum.append(rr)

        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Scatter(x=r_steps, y=r_cum, mode="lines", name="🎲 Random",
                                      line=dict(color="#ff6b6b", width=2, dash="dot")))
        fig_cmp.add_trace(go.Scatter(x=t_steps, y=t_cum, mode="lines", name="🤖 Trained",
                                      line=dict(color="#00ff88", width=2.5),
                                      fill="tonexty", fillcolor="rgba(0,255,136,0.07)"))
        if step_idx < len(t_frames):
            fig_cmp.add_vline(x=step_idx + 1, line_color="#ffffff", line_dash="dot", opacity=0.4,
                              annotation_text="current step", annotation_font_color="#aaaaaa")
        fig_cmp.update_layout(
            paper_bgcolor="#0e1117", plot_bgcolor="#1a1f35", height=280,
            font=dict(color="#aaaaaa"),
            xaxis=dict(title="Step", gridcolor="#2d3250", color="#aaaaaa"),
            yaxis=dict(title="Cumulative Reward", gridcolor="#2d3250", color="#aaaaaa", zeroline=False),
            legend=dict(bgcolor="#1e2130", bordercolor="#2d3250", font=dict(color="#cccccc")),
            margin=dict(t=10, b=40),
        )
        st.plotly_chart(fig_cmp, use_container_width=True)


# ── PAGE 3: PROOF OF LEARNING ─────────────────────────────────────────────
elif page == "📈 Proof of Learning":
    st.markdown('<div class="section-header">📈 Proof — Did the Model Actually Learn?</div>', unsafe_allow_html=True)

    df_eval = load_eval_metrics()
    df_train = load_training_metrics()

    if df_eval is not None and not df_eval.empty:
        available_models = df_eval["model"].unique().tolist()
        order = [m for m in ["random", "early_30k", "mid_100k", "late_200k", "final"] if m in available_models]
        colors_map = {"random": "#888888", "early_30k": "#ff9f43",
                      "mid_100k": "#4a9eff", "late_200k": "#a29bfe", "final": "#00ff88"}

        summary = df_eval.groupby("model").agg(
            avg_reward=("total_reward", "mean"),
            std_reward=("total_reward", "std"),
            avg_efficiency=("efficiency", "mean"),
            avg_completion=("completion_rate", "mean"),
        ).reset_index()
        summary = summary[summary["model"].isin(order)].copy()
        summary["_order"] = summary["model"].map({m: i for i, m in enumerate(order)})
        summary = summary.sort_values("_order").drop(columns="_order").fillna(0)

        random_row = summary[summary["model"] == "random"]
        final_row = summary[summary["model"] == "final"]

        st.markdown("### Key Numbers")
        if not random_row.empty and not final_row.empty:
            r_rew = random_row["avg_reward"].values[0]
            f_rew = final_row["avg_reward"].values[0]
            improvement = (f_rew - r_rew) / max(abs(r_rew), 1e-5) * 100
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f'<div class="proof-box"><div class="proof-number">{r_rew:.1f}</div>Random avg reward<br><small>across 20 episodes</small></div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="proof-box"><div class="proof-number">{f_rew:.1f}</div>Trained avg reward<br><small>across 20 episodes</small></div>', unsafe_allow_html=True)
            c3.markdown(f'<div class="proof-box"><div class="proof-number">+{improvement:.1f}%</div>Reward improvement<br><small>trained vs random</small></div>', unsafe_allow_html=True)
            c4.markdown(f'<div class="proof-box"><div class="proof-number">3</div>Avg steps to finish<br><small>vs 80 step limit</small></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Average Reward per Policy** — error bars show consistency")
            fig_bar = go.Figure(go.Bar(
                x=summary["model"],
                y=summary["avg_reward"],
                marker_color=[colors_map.get(m, "#aaaaaa") for m in summary["model"]],
                error_y=dict(type="data", array=summary["std_reward"].tolist(),
                             visible=True, color="#ffffff", thickness=1.5),
                text=summary["avg_reward"].round(1),
                textposition="outside",
                textfont=dict(color="#ffffff"),
            ))
            fig_bar.update_layout(
                paper_bgcolor="#0e1117", plot_bgcolor="#1a1f35", height=320,
                font=dict(color="#aaaaaa"),
                xaxis=dict(color="#aaaaaa", gridcolor="#2d3250"),
                yaxis=dict(color="#aaaaaa", gridcolor="#2d3250", zeroline=False, title="Avg Reward"),
                margin=dict(t=10, b=40),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            st.markdown("**Reward across 20 eval episodes** — trained is higher and more consistent")
            fig_lines = go.Figure()
            for model_name in order:
                subset = df_eval[df_eval["model"] == model_name].reset_index(drop=True)
                if subset.empty:
                    continue
                fig_lines.add_trace(go.Scatter(
                    x=subset.index + 1, y=subset["total_reward"],
                    mode="lines+markers", name=model_name,
                    line=dict(color=colors_map.get(model_name, "#aaaaaa"), width=2),
                    marker=dict(size=4),
                ))
            fig_lines.update_layout(
                paper_bgcolor="#0e1117", plot_bgcolor="#1a1f35", height=320,
                font=dict(color="#aaaaaa"),
                xaxis=dict(title="Eval Episode", gridcolor="#2d3250", color="#aaaaaa"),
                yaxis=dict(title="Total Reward", gridcolor="#2d3250", color="#aaaaaa", zeroline=False),
                legend=dict(bgcolor="#1e2130", bordercolor="#2d3250", font=dict(color="#cccccc", size=10)),
                margin=dict(t=10, b=40),
            )
            st.plotly_chart(fig_lines, use_container_width=True)

        st.markdown("### Summary Table")
        disp = summary[["model", "avg_reward", "avg_efficiency", "avg_completion"]].copy()
        disp.columns = ["Policy", "Avg Reward", "Avg Efficiency", "Completion Rate"]
        disp["Avg Reward"] = disp["Avg Reward"].round(2)
        disp["Avg Efficiency"] = disp["Avg Efficiency"].round(3)
        disp["Completion Rate"] = (disp["Completion Rate"] * 100).round(1).astype(str) + "%"
        st.dataframe(disp, use_container_width=True, hide_index=True)

    if df_train is not None and not df_train.empty:
        st.markdown("### Training Curve — Reward Over 95,000+ Episodes")
        st.markdown("""<div class="explainer">
        The model starts with zero knowledge and improves through trial and error.
        <b>Rising trend</b> = learning. <b>Narrowing shaded band</b> = policy stabilising (less random, more consistent).
        Vertical lines mark checkpoints saved during training.
        </div>""", unsafe_allow_html=True)

        window = st.slider("Smoothing", 10, 500, 200)
        df_train["smooth"] = df_train["reward"].rolling(window, min_periods=1).mean()
        df_train["std"] = df_train["reward"].rolling(window, min_periods=1).std().fillna(0)

        fig_train = go.Figure()
        fig_train.add_trace(go.Scatter(
            x=pd.concat([df_train["episode"], df_train["episode"].iloc[::-1]]),
            y=pd.concat([df_train["smooth"] + df_train["std"],
                         (df_train["smooth"] - df_train["std"]).iloc[::-1]]),
            fill="toself", fillcolor="rgba(0,212,255,0.07)",
            line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip",
        ))
        fig_train.add_trace(go.Scatter(
            x=df_train["episode"], y=df_train["reward"],
            mode="lines", line=dict(color="#2d3250", width=1),
            showlegend=False, name="Raw",
        ))
        fig_train.add_trace(go.Scatter(
            x=df_train["episode"], y=df_train["smooth"],
            mode="lines", line=dict(color="#00d4ff", width=2.5),
            name="Smoothed reward",
        ))
        total_ep = len(df_train)
        for label_ck, frac, color in [("30k", 0.1, "#ff9f43"), ("100k", 0.33, "#4a9eff"),
                                       ("200k", 0.66, "#a29bfe"), ("300k", 1.0, "#00ff88")]:
            ep = int(total_ep * frac)
            if 0 < ep <= total_ep:
                fig_train.add_vline(x=ep, line_color=color, line_dash="dot", opacity=0.5,
                                    annotation_text=label_ck, annotation_font_color=color)
        fig_train.update_layout(
            paper_bgcolor="#0e1117", plot_bgcolor="#1a1f35", height=350,
            font=dict(color="#aaaaaa"),
            xaxis=dict(title="Episode", gridcolor="#2d3250", color="#aaaaaa"),
            yaxis=dict(title="Reward", gridcolor="#2d3250", color="#aaaaaa", zeroline=False),
            legend=dict(bgcolor="#1e2130", bordercolor="#2d3250", font=dict(color="#cccccc")),
            margin=dict(t=10, b=40),
        )
        st.plotly_chart(fig_train, use_container_width=True)

        early = df_train.head(500)
        late = df_train.tail(500)
        c1, c2, c3 = st.columns(3)
        c1.metric("Avg Reward (first 500 ep)", f"{early['reward'].mean():.2f}",
                  f"{late['reward'].mean() - early['reward'].mean():+.2f} by end")
        c2.metric("Std Dev (first 500 ep)", f"{early['reward'].std():.2f}",
                  f"{late['reward'].std() - early['reward'].std():+.2f} by end")
        c3.metric("Total Episodes", f"{len(df_train):,}", "300k timesteps")


# ── PAGE 4: WHAT-IF ───────────────────────────────────────────────────────
elif page == "🎛️ What-If":
    st.markdown('<div class="section-header">🎛️ What-If Explorer — Stress Test the Policy</div>', unsafe_allow_html=True)
    st.markdown("""<div class="explainer">
    Change the environment and run the trained model on scenarios it was never trained on.
    This tests <b>generalisation</b> — can a policy learned on 5 agents + 10 tasks handle variations?
    Try reducing agents, increasing tasks, or cutting energy to stress test it.
    </div>""", unsafe_allow_html=True)

    model = load_model()

    col_s, col_v = st.columns([1, 2])
    with col_s:
        st.markdown("### Parameters")
        num_tasks = st.slider("Number of Tasks", 5, 20, 10)
        num_type_a = st.slider("Type A Agents (strong, slow)", 0, 4, 2)
        num_type_b = st.slider("Type B Agents (fast, weak)", 0, 5, 3)
        energy_a = st.slider("Type A Energy", 30, 200, 100)
        energy_b = st.slider("Type B Energy", 20, 120, 60)
        st.markdown("---")
        n_agents = num_type_a + num_type_b
        ratio = num_tasks / max(n_agents, 1)
        st.markdown(f"**Agents:** {n_agents} | **Tasks:** {num_tasks} | **Ratio:** {ratio:.1f} tasks/agent")
        if ratio > 3:
            st.warning("High task/agent ratio — agents may not complete all tasks.")
        elif ratio < 1:
            st.info("More agents than tasks — some will be idle.")
        run_btn = st.button("▶ Run Episode", use_container_width=True, type="primary")

    with col_v:
        if run_btn:
            if model is None:
                st.error("No model found. Run `python train.py` first.")
            elif n_agents == 0:
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

        if "wif_frames" in st.session_state and st.session_state["wif_frames"]:
            wif_frames = st.session_state["wif_frames"]
            wif_configs = st.session_state.get("wif_configs", AGENT_CONFIGS)
            wif_num_tasks = st.session_state.get("wif_num_tasks", 10)
            wif_total_reward = st.session_state.get("wif_total_reward", 0)
            last = wif_frames[-1]

            m1, m2, m3, m4 = st.columns(4)
            m1.markdown(f'<div class="metric-box"><div class="metric-val" style="color:#00ff88">{last["tasks_completed"]}/{wif_num_tasks}</div><div class="metric-lbl">Tasks Done</div></div>', unsafe_allow_html=True)
            m2.markdown(f'<div class="metric-box"><div class="metric-val">{last["step"]}</div><div class="metric-lbl">Steps</div></div>', unsafe_allow_html=True)
            m3.markdown(f'<div class="metric-box"><div class="metric-val">{wif_total_reward:.1f}</div><div class="metric-lbl">Total Reward</div></div>', unsafe_allow_html=True)
            eff = last["tasks_completed"] / max(last["step"], 1)
            m4.markdown(f'<div class="metric-box"><div class="metric-val">{eff:.2f}</div><div class="metric-lbl">Tasks/Step</div></div>', unsafe_allow_html=True)

            wif_view = st.radio("View mode", ["▶ Animated", "🎚️ Step-by-Step"], horizontal=True, key="wif_view")

            if wif_view == "▶ Animated":
                fig_wif = build_animated_figure(wif_frames, "What-If Run", custom_configs=wif_configs)
                st.plotly_chart(fig_wif, use_container_width=True)
            else:
                step_idx_wif = st.slider("Step", 0, len(wif_frames) - 1, 0, key="wif_slider")
                fig_wif = build_grid_figure(wif_frames, step_idx_wif, "What-If Run", custom_configs=wif_configs)
                st.plotly_chart(fig_wif, use_container_width=True)
                narr_wif = build_narration(wif_frames[step_idx_wif])
                st.markdown(f'<div class="narration-box">{narr_wif}</div>', unsafe_allow_html=True)
        elif not run_btn:
            st.markdown("""
            <div style="background:#1e2130;border-radius:10px;padding:40px;text-align:center;
            border:1px dashed #2d3250;margin-top:20px;">
                <h3 style="color:#aaaaaa;">Configure parameters and click ▶ Run Episode</h3>
                <p style="color:#555;">Try fewer agents, more tasks, or lower energy to stress test the policy</p>
            </div>""", unsafe_allow_html=True)

st.markdown("---")
st.caption("KnowledgeQuarry 2026 · Convoke 8.0 · CIC University of Delhi · ML Engineering Track · MaskablePPO + Custom Gymnasium")