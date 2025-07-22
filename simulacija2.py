import streamlit as st
import numpy as np
from scipy.signal import convolve2d
import pandas as pd
import plotly.express as px

# Title and description
st.set_page_config(layout="centered")
st.markdown("<h2 style='text-align: center;'>🎰 Casino Game Simulation – Monte Carlo Analysis</h2>",
            unsafe_allow_html=True)
st.write("This simulation uses the Monte Carlo method to estimate rewards based on random cell marking and game rules.")

# Sidebar – Game Parameters
st.sidebar.header("🔧 Game Parameters")

P_MARK = st.sidebar.slider(
    "Probability of marking a cell (P_MARK)",
    min_value=0.01,
    max_value=0.1,
    value=0.025,
    step=0.005,
    format="%.3f"
)
LIVES = st.sidebar.slider("Number of lives", 1, 5, 3)
STAKE = st.sidebar.number_input("Stake per game (STAKE)", value=20)
REWARD_3x3 = st.sidebar.number_input("Reward for 3x3 block", value=110)
REWARD_2x2 = st.sidebar.number_input("Reward for 2x2 block", value=40)
MATRIX_SIZE = st.sidebar.selectbox(
    "Matrix size", [4, 5, 6, 7, 8], index=2)
N_SIMULATIONS = st.sidebar.number_input(
    "Number of Monte Carlo simulations", value=100000)

# Convolution filters
filter_3x3 = np.ones((3, 3), dtype=int)
filter_2x2 = np.ones((2, 2), dtype=int)

# Single game simulation


def simulate_single_game():
    matrix = np.zeros((MATRIX_SIZE, MATRIX_SIZE), dtype=bool)
    lives = LIVES

    while lives > 0:
        newly_marked = (np.random.rand(
            MATRIX_SIZE, MATRIX_SIZE) < P_MARK) & (~matrix)
        if not newly_marked.any():
            lives -= 1
        else:
            matrix |= newly_marked
            lives = LIVES

    marked_int = matrix.astype(int)
    reward = 0

    conv_3x3 = convolve2d(marked_int, filter_3x3, mode='valid')
    reward += np.sum(conv_3x3 == 9) * REWARD_3x3

    conv_2x2 = convolve2d(marked_int, filter_2x2, mode='valid')
    reward += np.sum(conv_2x2 == 4) * REWARD_2x2

    return reward


if st.button("🎲 Run Monte Carlo Simulation"):
    with st.spinner("Simulation in progress..."):
        outcomes = [simulate_single_game() for _ in range(int(N_SIMULATIONS))]

    total_reward = sum(outcomes)
    avg_reward = total_reward / N_SIMULATIONS
    win_to_stake = avg_reward / STAKE
    profit_pct = abs((win_to_stake - 1) * 100)
    label = "gain" if win_to_stake >= 1 else "loss"
    prob_double = sum(r >= 40 for r in outcomes) / N_SIMULATIONS
    prob_zero = sum(r == 0 for r in outcomes) / N_SIMULATIONS

    # Statistics display
    st.subheader("📊 Simulation Results:")
    st.markdown(f"- **Total reward:** {total_reward}")
    st.markdown(f"- **Average reward per game:** **{avg_reward:.2f} credits**")
    st.markdown(
        f"- **Win-to-stake ratio:** {win_to_stake:.4f} → **{profit_pct:.2f}% {label} per game**")
    st.sidebar.markdown(
        "ℹ️ **Win-to-stake ratio** measures how much you earn on average per unit staked. If it's >1, you're profitable.")
    st.markdown(f"- **Probability of reward ≥ 40:** {prob_double:.2%}")
    st.markdown(f"- **Probability of losing everything:** {prob_zero:.2%}")

    # Histogram
    df = pd.DataFrame({'Reward': outcomes})
    fig = px.histogram(df, x="Reward", nbins=100,
                       title="Distribution of Rewards per Game",
                       labels={"Reward": "Reward per Game (credits)"},
                       template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
