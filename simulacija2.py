import streamlit as st
import numpy as np
from scipy.signal import convolve2d
import pandas as pd
import plotly.express as px

# Naslov i opis
st.set_page_config(layout="centered")
st.markdown("<h2 style='text-align: center;'>🎰 Simulacija Kazino Igre – Monte Carlo Analiza</h2>",
            unsafe_allow_html=True)
st.write("Ova simulacija koristi Monte Carlo metodu za procenu dobitaka na osnovu nasumičnog označavanja ćelija i pravila igre.")

# Sidebar – parametri
st.sidebar.header("🔧 Parametri igre")

P_MARK = st.sidebar.slider(
    "Verovatnoća označavanja ćelije (P_MARK)",
    min_value=0.01,
    max_value=0.1,
    value=0.025,
    step=0.005,
    format="%.3f"
)
LIVES = st.sidebar.slider("Broj života", 1, 5, 3)
STAKE = st.sidebar.number_input("Ulog po igri (STAKE)", value=20)
REWARD_3x3 = st.sidebar.number_input("Nagrada za 3x3 blok", value=110)
REWARD_2x2 = st.sidebar.number_input("Nagrada za 2x2 blok", value=40)
MATRIX_SIZE = st.sidebar.selectbox(
    "Veličina matrice", [4, 5, 6, 7, 8], index=2)
N_SIMULATIONS = st.sidebar.number_input(
    "Broj Monte Carlo simulacija", value=10000)

# Konvolucioni filteri
filter_3x3 = np.ones((3, 3), dtype=int)
filter_2x2 = np.ones((2, 2), dtype=int)

# Simulacija jedne igre


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


# Pokretanje simulacije
if st.button("🎲 Pokreni Monte Carlo simulaciju"):
    with st.spinner("Simulacija u toku..."):
        outcomes = [simulate_single_game() for _ in range(int(N_SIMULATIONS))]

    total_reward = sum(outcomes)
    avg_reward = total_reward / N_SIMULATIONS
    win_to_stake = avg_reward / STAKE
    prob_double = sum(r >= 40 for r in outcomes) / N_SIMULATIONS
    prob_zero = sum(r == 0 for r in outcomes) / N_SIMULATIONS

    # Prikaz statistike
    st.subheader("📊 Rezultati simulacije:")
    st.markdown(f"- **Ukupna nagrada:** {total_reward}")
    st.markdown(f"- **Prosečan dobitak po igri:** {avg_reward:.2f}")
    st.markdown(f"- **Odnos dobitka i uloga:** {win_to_stake:.4f}")
    st.markdown(f"- **Verovatnoća dobitka ≥ 40:** {prob_double:.2%}")
    st.markdown(f"- **Verovatnoća da igrač izgubi sve:** {prob_zero:.2%}")

    # Interaktivni histogram
    df = pd.DataFrame({'Dobitak': outcomes})
    fig = px.histogram(df, x="Dobitak", nbins=100,
                       title="Distribucija dobitaka po igri",
                       labels={"Dobitak": "Dobitak po igri (krediti)"},
                       template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
