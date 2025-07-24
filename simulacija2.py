import streamlit as st
import numpy as np
from scipy.signal import convolve2d
import pandas as pd
import plotly.express as px
from collections import Counter

# Title and description
st.set_page_config(layout="centered")
st.markdown("<h2 style='text-align: center;'>🎰 Casino Game Simulation </h2>",
            unsafe_allow_html=True)
# st.write("This simulation uses the Monte Carlo method to estimate rewards based on random cell marking and game rules.")

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
    "Number of simulations", value=100000)

# Convolution kernels
filter_3x3 = np.ones((3, 3), dtype=int)
filter_2x2 = np.ones((2, 2), dtype=int)


def calculate_rewards_adjusted(matrix_bool, filter_3x3, filter_2x2, REWARD_3x3, REWARD_2x2):
    marked_int = matrix_bool.astype(int)

    # Original reward (overlapping allowed)
    reward = 0
    conv_3x3 = convolve2d(marked_int, filter_3x3, mode='valid')
    mask_3x3 = (conv_3x3 == 9)
    reward += np.sum(mask_3x3) * REWARD_3x3

    conv_2x2 = convolve2d(marked_int, filter_2x2, mode='valid')
    mask_2x2 = (conv_2x2 == 4)
    reward += np.sum(mask_2x2) * REWARD_2x2

    # Adjusted reward: 2x2 not overlapping with 3x3
    reward2 = 0
    used_mask = np.zeros_like(marked_int, dtype=bool)
    for i in range(conv_3x3.shape[0]):
        for j in range(conv_3x3.shape[1]):
            if conv_3x3[i, j] == 9:
                used_mask[i:i+3, j:j+3] = True
    reward2 += np.sum(mask_3x3) * REWARD_3x3

    for i in range(conv_2x2.shape[0]):
        for j in range(conv_2x2.shape[1]):
            if conv_2x2[i, j] == 4:
                if not used_mask[i:i+2, j:j+2].any():
                    reward2 += REWARD_2x2
                    used_mask[i:i+2, j:j+2] = True

    return reward, reward2


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

    reward, reward2 = calculate_rewards_adjusted(
        matrix, filter_3x3, filter_2x2, REWARD_3x3, REWARD_2x2
    )
    return reward, reward2


if st.button("🎲 Run Monte Carlo Simulation"):
    with st.spinner("Simulation in progress..."):
        outcomes = [simulate_single_game() for _ in range(int(N_SIMULATIONS))]
        rewards_all, rewards_strict = zip(*outcomes)

    total_reward = sum(rewards_all)
    avg_reward = total_reward / N_SIMULATIONS
    win_to_stake = avg_reward / STAKE
    prob_double = sum(r >= 40 for r in rewards_all) / N_SIMULATIONS
    prob_zero = sum(r == 0 for r in rewards_all) / N_SIMULATIONS

    total_reward_strict = sum(rewards_strict)
    avg_reward_strict = total_reward_strict / N_SIMULATIONS
    win_to_stake_strict = avg_reward_strict / STAKE
    prob_double_strict = sum(r >= 40 for r in rewards_strict) / N_SIMULATIONS
    prob_zero_strict = sum(r == 0 for r in rewards_strict) / N_SIMULATIONS

    profit_pct = abs((win_to_stake - 1) * 100)
    profit_pct_strict = abs((win_to_stake_strict - 1) * 100)
    label = "gain" if win_to_stake >= 1 else "loss"
    label_strict = "gain" if win_to_stake_strict >= 1 else "loss"

    reward_counts_strict = Counter(rewards_strict)
    df_strict = pd.DataFrame.from_dict(
        reward_counts_strict, orient='index').reset_index()
    df_strict.columns = ['Reward', 'Count']
    df_strict = df_strict.sort_values('Reward')
    reward_counts_all = Counter(rewards_all)
    df_all = pd.DataFrame.from_dict(
        reward_counts_all, orient='index').reset_index()
    df_all.columns = ['Reward', 'Count']
    df_all = df_all.sort_values('Reward')

    # Statistics display
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            "<h5 style='text-align: left;'>📊 With Overlap</h5>", unsafe_allow_html=True)
        formatted_total_reward = f"{total_reward:,}".replace(",", " ")
        st.markdown(f"- **Total reward:** {total_reward:,.0f} credits")
        st.markdown(
            f"- **Average reward per game:** **{avg_reward:.2f} credits**")
        st.markdown(
            f"- **Win-to-stake ratio:** {win_to_stake:.4f} → **{profit_pct:.2f}% {label} per game**")
        st.sidebar.markdown(
            "ℹ️ **Win-to-stake ratio** measures how much you earn on average per unit staked. If it's >1, you're profitable.")
        st.markdown(f"- **Probability of reward ≥ 40:** {prob_double:.5%}")
        st.markdown(f"- **Probability of losing everything:** {prob_zero:.5%}")

        fig_all = px.bar(
            df_all,
            x="Reward",
            y="Count",
            title="Distribution of Rewards per Game ",
            labels={"Reward": "Reward per Game", "Count": "Frequency"},
            template="plotly_white"
        )
        st.plotly_chart(fig_all, use_container_width=True)

    with col2:
        st.markdown(
            "<h5 style='text-align: left;'>📊 No 2x2 Overlap with 3x3</h5>", unsafe_allow_html=True)
        formatted_total_reward = f"{total_reward_strict:,}".replace(",", " ")
        st.markdown(
            f"- **Total reward:** {total_reward_strict:,.0f} credits")
        st.markdown(
            f"- **Average reward per game:** **{avg_reward_strict:.2f} credits**")

        st.markdown(
            f"- **Win-to-stake ratio:** {win_to_stake_strict:.4f} → **{profit_pct_strict:.2f}% {label_strict} per game**")
        st.markdown(
            f"- **Probability of reward ≥ 40:** {prob_double_strict:.5%}")
        st.markdown(
            f"- **Probability of losing everything:** {prob_zero_strict:.5%}")

        # Bar chart

        fig_strict = px.bar(
            df_strict,
            x="Reward",
            y="Count",
            title="Distribution of Rewards per Game",
            labels={"Reward": "Reward per Game", "Count": "Frequency"},
            template="plotly_white"
        )
        st.plotly_chart(fig_strict, use_container_width=True)

    st.markdown("<h5 style='text-align: left;'> Combined Reward Distribution (No Zeros)</h5>",
                unsafe_allow_html=True)

    df_all_filtered = df_all[df_all["Reward"] != 0].copy()
    df_all_filtered["Type"] = "With Overlap"

    df_strict_filtered = df_strict[df_strict["Reward"] != 0].copy()
    df_strict_filtered["Type"] = "No 2x2 Overlap"

    df_combined = pd.concat(
        [df_all_filtered, df_strict_filtered], ignore_index=True)

    fig_combined = px.bar(
        df_combined,
        x="Reward",
        y="Count",
        color="Type",
        barmode="group",
        title=None,
        labels={"Reward": "Reward per Game", "Count": "Frequency"},
        template="plotly_white"
    )
    st.plotly_chart(fig_combined, use_container_width=True)
