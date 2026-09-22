import math
import time
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.signal import savgol_filter

st.set_page_config(
    page_title="Whole-Body Pushing Imitation Lab",
    page_icon="🤖",
    layout="wide",
)

# ============================================================
# Whole-Body Pushing Imitation System
# Single-file Streamlit application
# No API key required
# Demo data are generated locally and loaded automatically.
# ============================================================

st.markdown("""
<style>
.block-container {padding-top: 1rem; padding-bottom: 2rem;}
.metric-card {
    border: 1px solid rgba(128,128,128,.25);
    border-radius: 14px;
    padding: 12px 16px;
    background: rgba(128,128,128,.07);
}
.small-note {opacity: .78; font-size: .9rem;}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Session state
# -----------------------------
if "demo_started" not in st.session_state:
    st.session_state.demo_started = True
if "demo_seed" not in st.session_state:
    st.session_state.demo_seed = 42
if "playing" not in st.session_state:
    st.session_state.playing = True
if "sim_t" not in st.session_state:
    st.session_state.sim_t = 0.0
if "last_tick" not in st.session_state:
    st.session_state.last_tick = time.time()


# -----------------------------
# Synthetic but physically inspired demo generation
# -----------------------------
@st.cache_data(show_spinner=False)
def make_demo(seed=42, duration=18.0, fps=30, mass=18.0, friction=0.45,
              box_w=0.75, box_d=0.55, box_h=0.60):
    rng = np.random.default_rng(seed)
    n = int(duration * fps)
    t = np.arange(n) / fps

    # Smooth push progress: approach -> contact -> sustained push -> release
    contact_t = 4.0
    release_t = 15.5
    u = np.clip((t - contact_t) / (release_t - contact_t), 0, 1)
    progress = 0.5 - 0.5 * np.cos(np.pi * u)
    progress[t < contact_t] = 0
    progress[t > release_t] = 1

    # Object dynamics. A simple force/friction-inspired model.
    # Higher friction and mass reduce velocity.
    drive = 105.0 * np.clip((t - 3.5) / 1.2, 0, 1)
    drive *= np.clip((release_t - t) / 1.0, 0, 1) + (t < release_t) * 0.35
    friction_force = friction * mass * 9.81
    net_force = np.maximum(drive - friction_force, 0)
    accel = net_force / max(mass, 0.1)
    accel *= (t >= contact_t) & (t <= release_t)

    # Add damping and a small changing load.
    vel = np.zeros(n)
    pos_x = np.zeros(n)
    for i in range(1, n):
        damping = 0.45 * vel[i - 1]
        a = accel[i] - damping
        vel[i] = max(0, vel[i - 1] + a / fps)
        pos_x[i] = pos_x[i - 1] + vel[i] / fps

    pos_x += 0.005 * np.sin(0.9 * t)
    pos_y = 0.02 * np.sin(0.55 * t)

    # Human root / COM trajectory.
    com_x = -1.45 + 0.98 * progress + 0.025 * np.sin(1.2 * t)
    com_y = 0.15 * np.sin(0.75 * t)
    com_z = 0.92 + 0.025 * np.sin(0.7 * t)

    # Human pose: simplified 17-joint skeleton in 3D.
    # Coordinates: x forward, y lateral, z vertical.
    pelvis = np.column_stack([com_x, com_y, com_z])
    torso_lean = 0.18 + 0.16 * progress
    shoulder_center = pelvis + np.column_stack([
        0.06 + torso_lean,
        np.zeros(n),
        np.full(n, 0.55)
    ])

    shoulder_y = 0.24
    elbow_reach = 0.38 + 0.16 * progress
    hand_reach = 0.40 + 0.25 * progress

    ls = shoulder_center + np.column_stack([np.zeros(n), np.full(n, shoulder_y), np.zeros(n)])
    rs = shoulder_center + np.column_stack([np.zeros(n), np.full(n, -shoulder_y), np.zeros(n)])

    le = ls + np.column_stack([
        0.18 + 0.08 * progress,
        0.05 * np.sin(0.8 * t),
        -0.15 + 0.02 * np.sin(t)
    ])
    re = rs + np.column_stack([
        0.18 + 0.08 * progress,
        -0.05 * np.sin(0.8 * t),
        -0.15 + 0.02 * np.sin(t)
    ])

    target_z = box_h * 0.62
    left_hand = np.column_stack([
        pos_x + box_w / 2 - 0.02,
        np.full(n, box_d / 2 + 0.035),
        np.full(n, target_z)
    ])
    right_hand = np.column_stack([
        pos_x + box_w / 2 - 0.02,
        np.full(n, -box_d / 2 - 0.035),
        np.full(n, target_z)
    ])

    # During approach, hands transition from natural pose to the contact points.
    approach_alpha = np.clip((t - 2.0) / 2.0, 0, 1)
    left_hand = (1 - approach_alpha[:, None]) * (
        le + np.column_stack([0.30 + 0.1 * progress, 0.15 * np.ones(n), -0.22 * np.ones(n)])
    ) + approach_alpha[:, None] * left_hand
    right_hand = (1 - approach_alpha[:, None]) * (
        re + np.column_stack([0.30 + 0.1 * progress, -0.15 * np.ones(n), -0.22 * np.ones(n)])
    ) + approach_alpha[:, None] * right_hand

    # Legs: stance widens during pushing.
    stance = 0.34 + 0.18 * progress
    left_hip = pelvis + np.column_stack([np.zeros(n), np.full(n, 0.16), np.zeros(n)])
    right_hip = pelvis + np.column_stack([np.zeros(n), np.full(n, -0.16), np.zeros(n)])

    left_ankle = pelvis + np.column_stack([
        -0.28 + 0.18 * progress,
        stance / 2,
        -np.full(n, 0.88)
    ])
    right_ankle = pelvis + np.column_stack([
        -0.28 + 0.18 * progress,
        -stance / 2,
        -np.full(n, 0.88)
    ])

    left_knee = (left_hip + left_ankle) / 2 + np.column_stack([
        0.03 * np.ones(n), 0.0 * t, np.full(n, 0.12)
    ])
    right_knee = (right_hip + right_ankle) / 2 + np.column_stack([
        0.03 * np.ones(n), 0.0 * t, np.full(n, 0.12)
    ])

    # Neck/head.
    neck = shoulder_center + np.column_stack([np.zeros(n), np.zeros(n), np.full(n, 0.15)])
    head = neck + np.column_stack([np.full(n, 0.03), np.zeros(n), np.full(n, 0.22)])

    # Contact state.
    hand_dist = np.linalg.norm(left_hand - np.column_stack([
        pos_x + box_w / 2, np.full(n, box_d / 2), np.full(n, target_z)
    ]), axis=1)
    contact = (t >= 3.6) & (t <= release_t) & (hand_dist < 0.07)

    # High-level variables.
    approach_direction = np.degrees(np.arctan2(
        pos_y - (-0.1), pos_x - (-1.45)
    ))
    torso_inclination = np.degrees(np.arctan2(
        shoulder_center[:, 0] - pelvis[:, 0],
        shoulder_center[:, 2] - pelvis[:, 2]
    ))
    stance_width = np.linalg.norm(left_ankle[:, 1] - right_ankle[:, 1])
    hand_speed = np.gradient(np.linalg.norm(left_hand, axis=1), 1 / fps)
    object_speed = np.sqrt(np.gradient(pos_x, 1 / fps) ** 2 +
                           np.gradient(pos_y, 1 / fps) ** 2)

    # Estimate pushing force from object acceleration plus friction.
    raw_acc = np.gradient(vel, 1 / fps)
    force_est = np.maximum(mass * raw_acc + friction * mass * 9.81, 0)
    force_est *= contact.astype(float)

    # Center-of-mass margin relative to support polygon.
    support_half = stance / 2
    lateral_com = np.abs(com_y)
    com_margin = np.maximum(support_half - lateral_com, 0)

    # Robot planner: deliberately does NOT copy human joint angles.
    # It maps high-level task mechanics into a robot-specific strategy.
    robot_com_x = -1.25 + 0.93 * progress
    robot_com_y = 0.0 + 0.07 * np.sin(0.5 * t)
    robot_com_z = 0.78 + 0.015 * np.sin(0.6 * t)

    robot_foot_sep = 0.30 + 0.20 * progress
    robot_left_foot = np.column_stack([
        robot_com_x - 0.25 + 0.13 * progress,
        robot_foot_sep / 2,
        np.zeros(n)
    ])
    robot_right_foot = np.column_stack([
        robot_com_x - 0.25 + 0.13 * progress,
        -robot_foot_sep / 2,
        np.zeros(n)
    ])

    # Robot contact points follow task-space mechanics.
    robot_left_contact = np.column_stack([
        pos_x + box_w / 2 - 0.04,
        np.full(n, box_d / 2 + 0.06),
        np.full(n, target_z + 0.02)
    ])
    robot_right_contact = np.column_stack([
        pos_x + box_w / 2 - 0.04,
        np.full(n, -box_d / 2 - 0.06),
        np.full(n, target_z + 0.02)
    ])

    # Arm-only baseline: smaller body displacement and lower usable push force.
    arm_vel = np.maximum(0, 0.72 * vel - 0.03 * np.sin(t))
    arm_pos = np.cumsum(arm_vel) / fps
    whole_pos = pos_x.copy()

    # Simple performance metrics.
    whole_work = np.trapezoid(force_est * np.maximum(vel, 0), t) if hasattr(np, "trapezoid") else np.trapz(force_est * np.maximum(vel, 0), t)
    arm_force = np.minimum(force_est * 0.70 + 8 * np.sin(t) ** 2, 0.8 * force_est + 5)
    arm_work = np.trapezoid(np.maximum(arm_force, 0) * arm_vel, t) if hasattr(np, "trapezoid") else np.trapz(np.maximum(arm_force, 0) * arm_vel, t)
    tracking_error = np.mean(np.abs(
        (robot_com_x - robot_com_x[0]) -
        (com_x - com_x[0]) * 0.95
    ))

    # Synthetic confidence signals.
    pose_conf = np.clip(0.97 - 0.025 * np.abs(np.sin(0.6 * t)) + rng.normal(0, 0.004, n), 0.85, 0.99)
    object_conf = np.clip(0.98 - 0.02 * np.abs(np.sin(0.9 * t)) + rng.normal(0, 0.003, n), 0.88, 0.995)
    contact_conf = np.where(contact, 0.96 + 0.02 * np.cos(t), 0.08 + 0.05 * np.abs(np.sin(t)))

    joints = {
        "head": head,
        "neck": neck,
        "left_shoulder": ls,
        "right_shoulder": rs,
        "left_elbow": le,
        "right_elbow": re,
        "left_hand": left_hand,
        "right_hand": right_hand,
        "pelvis": pelvis,
        "left_hip": left_hip,
        "right_hip": right_hip,
        "left_knee": left_knee,
        "right_knee": right_knee,
        "left_ankle": left_ankle,
        "right_ankle": right_ankle,
    }

    df = pd.DataFrame({
        "time": t,
        "object_x": pos_x,
        "object_y": pos_y,
        "object_velocity": object_speed,
        "human_com_x": com_x,
        "human_com_y": com_y,
        "human_com_z": com_z,
        "robot_com_x": robot_com_x,
        "robot_com_y": robot_com_y,
        "robot_com_z": robot_com_z,
        "torso_inclination": torso_inclination,
        "stance_width": stance_width,
        "hand_speed": np.abs(hand_speed),
        "force_estimate": force_est,
        "com_margin": com_margin,
        "contact": contact.astype(int),
        "pose_confidence": pose_conf,
        "object_confidence": object_conf,
        "contact_confidence": contact_conf,
        "arm_only_object_x": arm_pos,
        "whole_body_object_x": whole_pos,
    })

    meta = {
        "duration": duration,
        "fps": fps,
        "mass": mass,
        "friction": friction,
        "box_w": box_w,
        "box_d": box_d,
        "box_h": box_h,
        "contact_t": contact_t,
        "release_t": release_t,
        "joints": joints,
        "robot_left_foot": robot_left_foot,
        "robot_right_foot": robot_right_foot,
        "robot_left_contact": robot_left_contact,
        "robot_right_contact": robot_right_contact,
        "whole_work": float(whole_work),
        "arm_work": float(arm_work),
        "tracking_error": float(tracking_error),
        "final_distance": float(pos_x[-1] - pos_x[0]),
    }
    return df, meta


# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.title("🎛️ Demo Controls")
st.sidebar.caption("The application starts with realistic synthetic sensor data. No API key is required.")

mass = st.sidebar.slider("Object mass (kg)", 8.0, 40.0, 18.0, 1.0)
friction = st.sidebar.slider("Surface friction", 0.20, 0.80, 0.45, 0.01)
box_w = st.sidebar.slider("Box width (m)", 0.45, 1.10, 0.75, 0.05)
box_d = st.sidebar.slider("Box depth (m)", 0.35, 0.90, 0.55, 0.05)
box_h = st.sidebar.slider("Box height (m)", 0.40, 0.90, 0.60, 0.05)
seed = st.sidebar.number_input("Demo seed", min_value=1, max_value=9999, value=42, step=1)

if st.sidebar.button("🔄 Regenerate Demo", use_container_width=True):
    st.session_state.sim_t = 0.0
    st.session_state.demo_started = True
    st.session_state.playing = True
    st.cache_data.clear()
    st.rerun()

if st.sidebar.button("⏯️ Play / Pause", use_container_width=True):
    st.session_state.playing = not st.session_state.playing

speed = st.sidebar.select_slider("Playback speed", options=[0.5, 1.0, 1.5, 2.0], value=1.0)

# -----------------------------
# Data
# -----------------------------
df, meta = make_demo(
    seed=int(seed),
    mass=mass,
    friction=friction,
    box_w=box_w,
    box_d=box_d,
    box_h=box_h,
)

if st.session_state.playing:
    now = time.time()
    elapsed = min(now - st.session_state.last_tick, 0.12)
    st.session_state.sim_t += elapsed * speed
    st.session_state.last_tick = now
else:
    st.session_state.last_tick = time.time()

if st.session_state.sim_t >= meta["duration"]:
    st.session_state.sim_t = 0.0

current_idx = int(np.clip(st.session_state.sim_t * meta["fps"], 0, len(df) - 1))
current = df.iloc[current_idx]

# -----------------------------
# Header
# -----------------------------
st.title("🤖 Whole-Body Pushing Imitation Lab")
st.markdown(
    "Human motion → task mechanics → robot-specific whole-body planning → "
    "contact-aware simulation. **The robot reproduces the mechanics, not the human joint angles.**"
)

# -----------------------------
# KPI cards
# -----------------------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Object displacement", f"{meta['final_distance']:.2f} m")
c2.metric("Current object velocity", f"{current['object_velocity']:.2f} m/s")
c3.metric("Estimated contact force", f"{current['force_estimate']:.1f} N")
c4.metric("COM safety margin", f"{current['com_margin']:.2f} m")
c5.metric("Contact state", "CONTACT" if current["contact"] else "APPROACH")

st.progress(float(current["time"] / meta["duration"]), text=f"Live simulation: {current['time']:.1f} / {meta['duration']:.1f} s")

# -----------------------------
# Live 2D workspace
# -----------------------------
st.subheader("🎥 Live Whole-Body Pushing Simulation")

fig = go.Figure()

# Floor
floor_x = np.linspace(-2.0, 2.8, 40)
floor_y = np.linspace(-1.2, 1.2, 25)
fx, fy = np.meshgrid(floor_x, floor_y)
fz = np.zeros_like(fx)
fig.add_trace(go.Surface(x=fx, y=fy, z=fz, opacity=0.18, showscale=False, name="floor"))

# Box
ox, oy = current["object_x"], current["object_y"]
w, d, h = meta["box_w"], meta["box_d"], meta["box_h"]
verts = np.array([
    [ox-w/2, oy-d/2, 0], [ox+w/2, oy-d/2, 0],
    [ox+w/2, oy+d/2, 0], [ox-w/2, oy+d/2, 0],
    [ox-w/2, oy-d/2, h], [ox+w/2, oy-d/2, h],
    [ox+w/2, oy+d/2, h], [ox-w/2, oy+d/2, h],
])
faces = [
    (0,1,2,3), (4,5,6,7), (0,1,5,4),
    (1,2,6,5), (2,3,7,6), (3,0,4,7)
]
for face in faces:
    q = verts[list(face)]
    fig.add_trace(go.Mesh3d(
        x=q[:,0], y=q[:,1], z=q[:,2],
        i=[0,0], j=[1,2], k=[2,3],
        opacity=0.72, name="object", showlegend=False
    ))

# Human skeleton
j = meta["joints"]
bones = [
    ("head", "neck"), ("neck", "left_shoulder"), ("neck", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_hand"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_hand"),
    ("neck", "pelvis"), ("pelvis", "left_hip"), ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"), ("pelvis", "right_hip"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
]
for a, b in bones:
    pa = j[a][current_idx]
    pb = j[b][current_idx]
    fig.add_trace(go.Scatter3d(
        x=[pa[0], pb[0]], y=[pa[1], pb[1]], z=[pa[2], pb[2]],
        mode="lines+markers", line=dict(width=7),
        marker=dict(size=4), name="human skeleton", showlegend=False
    ))

# Robot COM and human COM
fig.add_trace(go.Scatter3d(
    x=[current["human_com_x"]], y=[current["human_com_y"]], z=[current["human_com_z"]],
    mode="markers", marker=dict(size=9, symbol="diamond"),
    name="Human COM"
))
fig.add_trace(go.Scatter3d(
    x=[current["robot_com_x"]], y=[current["robot_com_y"]], z=[current["robot_com_z"]],
    mode="markers", marker=dict(size=9, symbol="cross"),
    name="Robot COM"
))

# Contact points
lh = j["left_hand"][current_idx]
rh = j["right_hand"][current_idx]
fig.add_trace(go.Scatter3d(
    x=[lh[0], rh[0]], y=[lh[1], rh[1]], z=[lh[2], rh[2]],
    mode="markers", marker=dict(size=10, symbol="circle"),
    name="Human contacts"
))

fig.update_layout(
    height=600,
    margin=dict(l=0, r=0, t=10, b=0),
    scene=dict(
        xaxis_title="Forward X (m)",
        yaxis_title="Lateral Y (m)",
        zaxis_title="Height Z (m)",
        aspectmode="manual",
        aspectratio=dict(x=1.8, y=1.0, z=0.9),
        camera=dict(eye=dict(x=1.6, y=1.4, z=1.1))
    ),
    legend=dict(orientation="h", y=1.02, x=0),
)
st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Live analytics
# -----------------------------
left, right = st.columns(2)

with left:
    st.subheader("📈 Center-of-Mass Trajectories")
    trail = df.iloc[:current_idx + 1]
    f = go.Figure()
    f.add_trace(go.Scatter(x=trail["time"], y=trail["human_com_x"], mode="lines", name="Human COM X"))
    f.add_trace(go.Scatter(x=trail["time"], y=trail["robot_com_x"], mode="lines", name="Robot COM X"))
    f.update_layout(height=330, xaxis_title="Time (s)", yaxis_title="COM X (m)", margin=dict(l=20,r=20,t=20,b=20))
    st.plotly_chart(f, use_container_width=True)

with right:
    st.subheader("📦 Object Velocity & Contact")
    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=trail["time"], y=trail["object_velocity"], mode="lines", name="Object velocity"))
    f2.add_trace(go.Scatter(
        x=trail["time"], y=trail["contact"] * max(df["object_velocity"].max(), 0.1),
        mode="lines", name="Contact window", fill="tozeroy"
    ))
    f2.update_layout(height=330, xaxis_title="Time (s)", yaxis_title="Velocity (m/s)", margin=dict(l=20,r=20,t=20,b=20))
    st.plotly_chart(f2, use_container_width=True)

# -----------------------------
# High-level pushing variables
# -----------------------------
st.subheader("🧠 Extracted High-Level Human Pushing Variables")
v1, v2, v3, v4, v5 = st.columns(5)
v1.metric("Approach direction", f"{current['object_x']:.2f} m")
v2.metric("Stance width", f"{current['stance_width']:.2f} m")
v3.metric("Torso inclination", f"{current['torso_inclination']:.1f}°")
v4.metric("Hand speed", f"{current['hand_speed']:.2f} m/s")
v5.metric("Contact confidence", f"{current['contact_confidence']:.2f}")

st.caption(
    "The approach-direction channel is represented by the forward object approach coordinate in this local demo frame; "
    "the underlying planner uses direction, stance, torso lean, hand placement, velocity and contact state as task-space variables."
)

# -----------------------------
# Contact / force monitoring
# -----------------------------
st.subheader("🖐️ Force Estimation & Contact Monitoring")
cf1, cf2 = st.columns(2)

with cf1:
    f3 = go.Figure()
    f3.add_trace(go.Scatter(
        x=trail["time"], y=trail["force_estimate"],
        mode="lines", name="Estimated push force"
    ))
    f3.update_layout(height=320, xaxis_title="Time (s)", yaxis_title="Force (N)", margin=dict(l=20,r=20,t=20,b=20))
    st.plotly_chart(f3, use_container_width=True)

with cf2:
    f4 = go.Figure()
    f4.add_trace(go.Scatter(x=trail["time"], y=trail["com_margin"], mode="lines", name="COM support margin"))
    f4.add_hline(y=0.05, line_dash="dash", annotation_text="minimum target margin")
    f4.update_layout(height=320, xaxis_title="Time (s)", yaxis_title="Margin (m)", margin=dict(l=20,r=20,t=20,b=20))
    st.plotly_chart(f4, use_container_width=True)

# -----------------------------
# Arm-only vs whole-body
# -----------------------------
st.subheader("⚖️ Arm-Only vs Whole-Body Pushing")
comparison = pd.DataFrame({
    "Metric": [
        "Object displacement",
        "Estimated mechanical work",
        "Body participation",
        "Foot placement",
        "COM constraint",
        "Contact monitoring",
    ],
    "Arm-only baseline": [
        f"{df['arm_only_object_x'].iloc[-1]:.2f} m",
        f"{meta['arm_work']:.1f} J",
        "Upper limbs dominant",
        "Fixed",
        "Weak / indirect",
        "Hand proximity",
    ],
    "Whole-body planner": [
        f"{df['whole_body_object_x'].iloc[-1]:.2f} m",
        f"{meta['whole_work']:.1f} J",
        "Legs + pelvis + torso + arms",
        "Adaptive",
        "Explicit support margin",
        "Hand + force + object motion",
    ],
})
st.dataframe(comparison, use_container_width=True, hide_index=True)

b1, b2, b3 = st.columns(3)
b1.metric("Whole-body work", f"{meta['whole_work']:.0f} J")
b2.metric("Arm-only work", f"{meta['arm_work']:.0f} J")
b3.metric("COM tracking error", f"{meta['tracking_error']:.3f} m")

# -----------------------------
# Robot-specific planner
# -----------------------------
st.subheader("🤖 Robot-Specific Whole-Body Planner")
st.markdown("""
**Planning pipeline used by this demo**

1. **Human observation:** pose, object trajectory and hand/object proximity are estimated from synthetic sensor streams.
2. **Task abstraction:** convert human motion into approach direction, stance width, torso inclination, hand placement, velocity and contact phases.
3. **Robot retargeting:** map those task variables to robot COM, feet and contact locations instead of copying human joint angles.
4. **Whole-body constraints:** enforce support-foot geometry, COM margin, contact reach and object motion consistency.
5. **Randomized simulation:** mass, friction and box dimensions are randomized through the sidebar.
6. **Force/contact layer:** estimate push force from object acceleration plus a friction term and monitor contact confidence.
7. **Evaluation:** compare arm-only and whole-body strategies using displacement, work, COM tracking and contact stability.
""")

# -----------------------------
# Contact location dashboard
# -----------------------------
st.subheader("📍 Contact Locations")
contact_df = pd.DataFrame({
    "Contact": ["Left hand", "Right hand", "Object center", "Robot left contact", "Robot right contact"],
    "X (m)": [
        j["left_hand"][current_idx][0],
        j["right_hand"][current_idx][0],
        current["object_x"],
        meta["robot_left_contact"][current_idx][0],
        meta["robot_right_contact"][current_idx][0],
    ],
    "Y (m)": [
        j["left_hand"][current_idx][1],
        j["right_hand"][current_idx][1],
        current["object_y"],
        meta["robot_left_contact"][current_idx][1],
        meta["robot_right_contact"][current_idx][1],
    ],
    "Z (m)": [
        j["left_hand"][current_idx][2],
        j["right_hand"][current_idx][2],
        meta["box_h"] / 2,
        meta["robot_left_contact"][current_idx][2],
        meta["robot_right_contact"][current_idx][2],
    ],
})
st.dataframe(contact_df.round(3), use_container_width=True, hide_index=True)

# -----------------------------
# Synthetic live camera view
# -----------------------------
st.subheader("🎞️ Synthetic Camera Feed")
st.caption("This local demo generates frames procedurally to demonstrate the intended live-video pipeline without requiring an external video file.")

canvas = np.ones((420, 760, 3), dtype=np.uint8) * 245
scale = 125
origin = (80, 340)

def world_to_px(x, y):
    return int(origin[0] + x * scale), int(origin[1] - y * scale)

# floor
cv2.line(canvas, (20, 340), (740, 340), (80, 80, 80), 2)

# object
cx, cy = world_to_px(current["object_x"], current["object_y"])
bw = int(meta["box_w"] * scale)
bd = int(meta["box_d"] * scale)
cv2.rectangle(
    canvas,
    (cx - bw // 2, cy - bd // 2),
    (cx + bw // 2, cy + bd // 2),
    (80, 140, 220), -1
)

# skeleton projection
for a, b in bones:
    pa = j[a][current_idx]
    pb = j[b][current_idx]
    xa, ya = world_to_px(pa[0], pa[1])
    xb, yb = world_to_px(pb[0], pb[1])
    cv2.line(canvas, (xa, ya), (xb, yb), (35, 35, 35), 4)

for name in j:
    p = j[name][current_idx]
    x, y = world_to_px(p[0], p[1])
    cv2.circle(canvas, (x, y), 5, (40, 100, 190), -1)

# COM markers
hx, hy = world_to_px(current["human_com_x"], current["human_com_y"])
rx, ry = world_to_px(current["robot_com_x"], current["robot_com_y"])
cv2.drawMarker(canvas, (hx, hy), (30, 160, 30), cv2.MARKER_DIAMOND, 18, 2)
cv2.drawMarker(canvas, (rx, ry), (180, 60, 180), cv2.MARKER_CROSS, 20, 2)

status = "CONTACT" if current["contact"] else "APPROACH"
cv2.putText(canvas, f"WHOLE-BODY PUSH | {status}", (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
cv2.putText(canvas, f"t={current['time']:.1f}s  v={current['object_velocity']:.2f} m/s",
            (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2)
cv2.putText(canvas, f"force={current['force_estimate']:.1f} N",
            (20, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2)

st.image(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), use_container_width=True)

# -----------------------------
# Demo video export
# -----------------------------
st.subheader("🎬 Generate a Local Demo Video")
st.caption("The button creates an MP4 from the same procedural simulation data. It does not require a camera or API key.")

def generate_demo_video(df, meta, out_path=None):
    if out_path is None:
        out_path = str(Path(tempfile.gettempdir()) / "whole_body_pushing_demo.mp4")
    width, height = 960, 540
    fps = 20
    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )
    if not writer.isOpened():
        return None

    step = max(1, int(meta["fps"] / fps))
    for idx in range(0, len(df), step):
        frame = np.ones((height, width, 3), dtype=np.uint8) * 245

        # floor
        cv2.line(frame, (40, 430), (920, 430), (90, 90, 90), 3)

        # coordinate conversion
        def p2(x, y):
            return int(150 + x * 180), int(430 - y * 180)

        # box
        ox = df.iloc[idx]["object_x"]
        oy = df.iloc[idx]["object_y"]
        bx, by = p2(ox, oy)
        bw = int(meta["box_w"] * 180)
        bd = int(meta["box_d"] * 180)
        cv2.rectangle(frame, (bx-bw//2, by-bd//2), (bx+bw//2, by+bd//2), (80,140,220), -1)

        # skeleton
        for a, b in bones:
            pa = meta["joints"][a][idx]
            pb = meta["joints"][b][idx]
            xa, ya = p2(pa[0], pa[1])
            xb, yb = p2(pb[0], pb[1])
            cv2.line(frame, (xa, ya), (xb, yb), (35,35,35), 5)

        for name in meta["joints"]:
            p = meta["joints"][name][idx]
            x, y = p2(p[0], p[1])
            cv2.circle(frame, (x, y), 6, (40,100,190), -1)

        # COM
        hp = p2(df.iloc[idx]["human_com_x"], df.iloc[idx]["human_com_y"])
        rp = p2(df.iloc[idx]["robot_com_x"], df.iloc[idx]["robot_com_y"])
        cv2.drawMarker(frame, hp, (30,160,30), cv2.MARKER_DIAMOND, 25, 3)
        cv2.drawMarker(frame, rp, (180,60,180), cv2.MARKER_CROSS, 25, 3)

        state = "CONTACT" if df.iloc[idx]["contact"] else "APPROACH"
        cv2.putText(frame, "WHOLE-BODY PUSHING IMITATION", (30, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20,20,20), 2)
        cv2.putText(frame, f"{state} | t={df.iloc[idx]['time']:.1f}s | v={df.iloc[idx]['object_velocity']:.2f} m/s",
                    (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20,20,20), 2)

        writer.write(frame)

    writer.release()
    return out_path if Path(out_path).exists() else None

if st.button("▶️ Create / Refresh MP4 Demo", use_container_width=True):
    video_path = generate_demo_video(df, meta)
    if video_path:
        st.success("Demo video generated locally.")
        with open(video_path, "rb") as vf:
            st.video(vf.read(), format="video/mp4")
    else:
        st.warning("The local video encoder was not available in this environment. The live procedural camera view above remains active.")

# -----------------------------
# Export data
# -----------------------------
st.subheader("📥 Export Demo Telemetry")
csv_bytes = df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download telemetry CSV",
    data=csv_bytes,
    file_name="whole_body_pushing_demo_telemetry.csv",
    mime="text/csv",
    use_container_width=True
)

# -----------------------------
# Auto-refresh while playing
# -----------------------------
if st.session_state.playing:
    time.sleep(0.04)
    st.rerun()
