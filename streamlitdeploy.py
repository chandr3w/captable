#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Use a dark theme for matplotlib plots.
plt.style.use('dark_background')

# ==============================================================================
# INITIAL SETUP
# ==============================================================================
st.set_page_config(layout="wide")
st.title("Multi‑Round Cap Table Simulator")

st.markdown("""
This simulator handles multiple financing rounds. In each priced round the following operations occur in order:

1. **Issue Option Pool Shares:**  
   New option pool shares are issued as a fraction of the final post‑round total:
   \[
   x = P\,T.
   \]

2. **Convert Outstanding SAFEs:**  
   All outstanding SAFE notes convert together. For each SAFE note, let
   \[
   r_i = \frac{\text{investment}_i}{\text{valuation cap}_i}.
   \]
   Define the aggregate conversion ratio:
   \[
   R = \sum_i r_i.
   \]
   Then the combined SAFE conversion shares are calculated as
   \[
   s = \frac{R}{\,1-R}\,\Bigl(S_0 + x\Bigr),
   \]
   which ensures that the intermediate SAFE ownership (i.e. \(s/(S_0+x+s)\)) equals \(R\).

3. **Issue New Priced Investor Shares:**  
   New investor shares are issued so that the investor’s final ownership is
   \[
   \frac{p}{T} = \frac{I}{V+I},
   \]
   i.e. 
   \[
   p = \frac{I}{V+I}\,T.
   \]

The final total post‑round share count is
\[
T = S_0 + x + s + p.
\]

Substituting in \(x = P\,T\) and \(s = \frac{R}{1-R}(S_0+P\,T)\) and \(p = \frac{I}{V+I}\,T\), we have:
\[
T = S_0 + P\,T + \frac{R}{1-R}\,(S_0+P\,T) + \frac{I}{V+I}\,T.
\]

We solve this equation using fsolve.
An intermediate table shows the ownership percentages of the Founders, SAFE Conversion, and Option Pool (before investor dilution) so that you can verify the SAFE conversion yields an intermediate ownership of \(R = \frac{A}{V_{\text{SAFE}}}\) (if a single note) or the appropriate aggregate for multiple notes.
""")

# ------------------------------------------------------------------------------
# Founders & Pre‑Financing Setup
# ------------------------------------------------------------------------------
st.sidebar.header("Initial Setup")
S0 = st.sidebar.number_input("Founders' Initial Common Shares (S₀)", value=1_000_000, step=1000)

st.sidebar.header("Pre‑Financing Shareholders")
num_founders = st.sidebar.selectbox("Number of Founders", options=range(1, 6))
founder_shares = {}
for i in range(num_founders):
    col1, col2 = st.sidebar.columns(2)
    with col1:
        name = st.text_input(f"Founder {i+1} Name", value=f"Founder {i+1}", key=f"founder_name_{i}")
    with col2:
        pct = st.number_input(f"Founder {i+1} Ownership (%)", min_value=0.0, max_value=100.0,
                               value=100/num_founders, step=1.0, key=f"founder_pct_{i}")
    founder_shares[name] = S0 * (pct/100)
    
# Founders' shares remain fixed.
F0 = sum(founder_shares.values())
cap_table = founder_shares.copy()
founder_ids = list(founder_shares.keys())

# ------------------------------------------------------------------------------
# Financing Rounds Setup (Up to 5 Rounds)
# ------------------------------------------------------------------------------
st.sidebar.header("Financing Rounds")
total_rounds = st.sidebar.selectbox("Total Number of Rounds", options=[1,2,3,4,5], index=0)

rounds = []  # Will hold round parameters.
for r in range(1, total_rounds+1):
    st.sidebar.subheader(f"Round {r} Settings")
    round_type = st.sidebar.selectbox(f"Round {r} Type", options=["SAFE", "Priced"], key=f"round_type_{r}")
    round_dict = {"round_number": r, "type": round_type}
    
    if round_type == "SAFE":
        num_safe = st.sidebar.number_input(f"Round {r} Number of SAFE Notes", value=1, min_value=1, step=1, key=f"safe_num_{r}")
        safes = []
        for j in range(num_safe):
            st.sidebar.markdown(f"**SAFE {j+1} Terms for Round {r}:**")
            inv = st.sidebar.number_input(f"SAFE {j+1} Investment Amount ($)", value=100_000, step=1000, key=f"safe_investment_{r}_{j}")
            cap_val = st.sidebar.number_input(f"SAFE {j+1} Valuation Cap ($)", value=5_000_000, step=100_000, key=f"safe_cap_{r}_{j}")
            disc = st.sidebar.selectbox(f"SAFE {j+1} Discount (%)", options=[0,5,10,15,20,25,30], key=f"safe_discount_{r}_{j}")
            # For this model, the discount will not affect conversion beyond the ratio.
            safes.append({
                "investment": inv,
                "valuation_cap": cap_val,
                "discount": disc / 100.0  # Unused in this formula
            })
        round_dict["safes"] = safes
    else:  # Priced round.
        round_dict["pre_money_valuation"] = st.sidebar.number_input(f"Round {r} Pre‑Money Valuation ($)",
                                                                    value=10_000_000, step=500_000, key=f"pre_money_{r}")
        # Target option pool percentage: default 0% for Round 1; 10% for rounds 2+.
        if r == 1:
            pool_target = st.sidebar.number_input(f"Round {r} Option Pool Target (%)", value=0.0, step=1.0, key=f"option_pool_{r}")
        else:
            pool_target = st.sidebar.number_input(f"Round {r} Option Pool Target (%)", value=10.0, step=1.0, key=f"option_pool_{r}")
        round_dict["option_pool"] = pool_target / 100.0
        round_dict["investment"] = st.sidebar.number_input(f"Round {r} Investment Amount ($)",
                                                           value=2_000_000, step=100_000, key=f"round_investment_{r}")
    rounds.append(round_dict)

# ------------------------------------------------------------------------------
# SIMULATION STATE INITIALIZATION
# ------------------------------------------------------------------------------
st.header("Round‑by‑Round Simulation Results")

# Accumulate outstanding SAFE notes until converted.
outstanding_safe_notes = []  # List of dictionaries.
# New classes issued in priced rounds.
issued_classes = {}  # e.g., {"Priced R2 Investor": shares, "Option Pool R2": shares, "SAFE R2 Conversion": shares}
# Snapshots of each round.
snapshots = []

def build_snapshot(details=""):
    all_classes = {}
    all_classes.update(cap_table)
    all_classes.update(issued_classes)
    total = sum(all_classes.values())
    rows = []
    for cls, shares in all_classes.items():
        rows.append({"Shareholder": cls, "Shares": shares, "% Ownership": f"{100 * shares / total:,.2f}%"})
    return pd.DataFrame(rows), details

# ------------------------------------------------------------------------------
# Equation function to solve for T using fsolve.
# ------------------------------------------------------------------------------
def total_shares_eq(T, S0, P_target, I_amt, V_pre, safe_investments, safe_caps):
    # x: option pool shares issued
    x = P_target * T
    # Compute aggregate SAFE ratio: r_i = investment_i/valuation_cap_i.
    if safe_investments:
        R_total = sum([inv/cap for inv, cap in zip(safe_investments, safe_caps)])
        # SAFE conversion shares are computed using the conversion base (S₀ + x):
        s = (R_total / (1 - R_total)) * (S0 + x)
    else:
        s = 0
    # Investor shares are computed to ensure investor fraction = I/(V+I):
    p = (I_amt / (V_pre + I_amt)) * T
    return S0 + x + s + p - T

# ------------------------------------------------------------------------------
# SIMULATE ROUNDS
# ------------------------------------------------------------------------------
for rnd in rounds:
    round_label = f"Round {rnd['round_number']} - {rnd['type']}"
    if rnd["type"] == "SAFE":
        # For a SAFE round, simply record the SAFE note inputs.
        for note in rnd["safes"]:
            note["round_number"] = rnd["round_number"]
            outstanding_safe_notes.append(note)
        details = f"SAFE Round {rnd['round_number']}: Issued {len(rnd['safes'])} SAFE note(s). Outstanding SAFE notes: {len(outstanding_safe_notes)}."
        df_snap, _ = build_snapshot(details=details)
        snapshots.append({"round": round_label, "df": df_snap, "details": details})
    else:
        # ---- PRICED ROUND ----
        V_pre = rnd["pre_money_valuation"]    # Pre‑money valuation
        I_amt = rnd["investment"]             # Investment amount
        P_target = rnd["option_pool"]         # Option pool target fraction
        # Investor target fraction is now defined as I/(V+I)
        x_target = I_amt / (V_pre + I_amt)
        if x_target + P_target >= 1:
            st.error(f"Round {rnd['round_number']} error: investor target {x_target:.2f} + option pool target {P_target:.2f} must be less than 1.")
            st.stop()
        
        # ---- STEP 1: Collect SAFE note inputs for this round.
        safe_investments = []
        safe_caps = []
        if outstanding_safe_notes:
            for note in outstanding_safe_notes:
                safe_investments.append(note["investment"])
                safe_caps.append(note["valuation_cap"])
        else:
            safe_investments = []
            safe_caps = []
        
        # ---- STEP 2: Solve for Final Total Shares T using fsolve.
        args = (F0, P_target, I_amt, V_pre, safe_investments, safe_caps)
        # Use an initial guess; if no SAFEs, use closed-form; if SAFEs exist, start a bit higher.
        if not safe_investments:
            initial_guess = [F0 / (1 - P_target - I_amt/(V_pre+I_amt))]
        else:
            initial_guess = [F0 * 1.2]
        T_solution, info, ier, msg = fsolve(total_shares_eq, initial_guess, args=args, full_output=True)
        if ier != 1:
            st.error(f"Round {rnd['round_number']} fsolve did not converge: {msg}")
            st.stop()
        T_final = T_solution[0]
        
        # ---- STEP 3: Back-calculate each class.
        x_calculated = P_target * T_final
        if safe_investments:
            R_total = sum([inv/cap for inv, cap in zip(safe_investments, safe_caps)])
            s_calculated = (R_total / (1 - R_total)) * (F0 + x_calculated)
        else:
            s_calculated = 0
        p_calculated = (I_amt / (V_pre + I_amt)) * T_final
        
        # ---- STEP 4: Build an intermediate snapshot (before investor dilution)
        T_intermediate = F0 + x_calculated + s_calculated
        founders_pct = 100 * F0 / T_intermediate
        safe_pct = 100 * s_calculated / T_intermediate if T_intermediate else 0
        pool_pct = 100 * x_calculated / T_intermediate if T_intermediate else 0
        intermediate_df = pd.DataFrame({
            "Share Class": ["Founders", "SAFE Conversion", "Option Pool"],
            "Shares": [F0, s_calculated, x_calculated],
            "% Ownership": [f"{founders_pct:,.2f}%", f"{safe_pct:,.2f}%", f"{pool_pct:,.2f}%"]
        })
        # Here, note that each SAFE note's intermediate ownership (if taken individually) would be:
        # For each note with ratio r_i = inv/cap, its conversion would be computed as s_i = (r_i/(1 - r_i))*(F0 + x_calculated).
        # However, to prevent dilution among SAFE notes, we use the aggregate conversion:
        #   s_total = (R_total/(1 - R_total))*(F0+x_calculated).
        
        # ---- STEP 5: Record this priced round's issuances.
        details = f"Priced Round {rnd['round_number']}: Pre‑money = ${V_pre:,.0f}, Investment = ${I_amt:,.0f}.\n"
        details += f"Investor target (p/T) = {100*x_target:.2f}% (I/(V+I)), Option pool target = {100*P_target:.2f}%.\n"
        if safe_investments:
            safe_label = f"SAFE R{rnd['round_number']} Conversion"
            issued_classes[safe_label] = s_calculated
            details += f"Converted {len(safe_investments)} SAFE note(s) (aggregate R = {R_total:.4f}) yield {s_calculated:,.0f} shares (conversion base = S₀ + option pool).\n"
            outstanding_safe_notes = []  # Clear all after conversion.
        inv_label = f"Priced R{rnd['round_number']} Investor"
        pool_label = f"Option Pool R{rnd['round_number']}"
        issued_classes[inv_label] = p_calculated
        issued_classes[pool_label] = x_calculated
        details += (f"Issued {p_calculated:,.0f} investor shares and {x_calculated:,.0f} option pool shares.\n"
                    f"Final fully diluted total T = {T_final:,.0f} shares (Investor fraction = {100*p_calculated/T_final:,.2f}%).")
        
        df_snap, _ = build_snapshot(details=details)
        snapshots.append({"round": round_label, "df": df_snap, "details": details, "intermediate": intermediate_df})
        
# ------------------------------------------------------------------------------
# DISPLAY SNAPSHOTS SIDE‑BY‑SIDE
# ------------------------------------------------------------------------------
st.header("Cap Table Snapshots by Round")
for snap in snapshots:
    st.subheader(snap["round"])
    st.write(snap["details"])
    col1, col2, col3 = st.columns(3)
    with col1:
        st.dataframe(snap["df"])
    with col2:
        if "intermediate" in snap:
            st.subheader("Intermediate (Pre-Investor)")
            st.dataframe(snap["intermediate"])
        else:
            st.write("No intermediate snapshot.")
    with col3:
        df_num = snap["df"][snap["df"]["Shares"].apply(lambda x: isinstance(x, (int, float)))]
        if not df_num.empty:
            fig, ax = plt.subplots()
            ax.pie(df_num["Shares"], labels=df_num["Shareholder"], autopct="%1.1f%%")
            ax.set_title(snap["round"] + " Breakdown")
            st.pyplot(fig)
            
# ------------------------------------------------------------------------------
# FINAL CUMULATIVE CAP TABLE
# ------------------------------------------------------------------------------
st.header("Final Cumulative Cap Table")
final_cap = {}
final_cap.update(cap_table)
final_cap.update(issued_classes)
total_final = sum(final_cap.values())
final_data = []
for cls, shares in final_cap.items():
    final_data.append({
        "Shareholder": cls,
        "Shares": shares,
        "% Ownership": f"{100 * shares / total_final:,.2f}%"
    })
st.dataframe(pd.DataFrame(final_data))

