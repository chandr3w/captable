#!/usr/bin/env python
# coding: utf-8

# In[3]:


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
st.set_page_config(layout="wide", page_title="Cap Table Simulator", page_icon="https://atas.vc/img/favicon.png")
st.image('https://atas.vc/img/logo.png', width=200)
st.markdown(
    "This open source model was developed by [Andrew Chan](https://www.linkedin.com/in/chandr3w/) "
    "from [Atas VC](https://atas.vc/)."
)

st.title("Multi‑Round Cap Table Simulator")
st.markdown(r"""### Outlining Calculations 
**1. Option Pool Issuance** $x = P_{\mathrm{target}}\,T$. 

**2. SAFE Conversion** For each SAFE note with investment $A_i$ and cap $V_{\mathrm{cap},i}$ define $r_i = \frac{A_i}{V_{\mathrm{cap},i}}$, $R = \sum_i r_i$, and $s = \frac{R}{1 - R}(S_0 + x)$. 

**3. Investor Shares** An investment $I$ into a pre-money valuation $V$ yields $p = \frac{I}{V + I}\,T$. Finally we solve $T = S_0 + x + s + p$ for $T$ using `fsolve`, and back-calculate each class’s share count.""", unsafe_allow_html=True)



# ------------------------------------------------------------------------------
# Founders & Pre‑Financing Setup
# ------------------------------------------------------------------------------
st.sidebar.header("Initial Setup")
S0 = st.sidebar.number_input("Founders' Initial Common Shares (S₀)", value=1_000_000, step=1000)

st.sidebar.header("Pre‑Financing Shareholders")
num_founders = st.sidebar.selectbox(
    "Number of Founders",
    options=range(1, 6),
    index=1  # defaults to the second option, i.e. 2 founders
)
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
total_rounds = st.sidebar.selectbox("Total Number of Rounds", options=[1,2,3,4,5], index=1)

rounds = []
for r in range(1, total_rounds+1):
    st.sidebar.subheader(f"Round {r} Settings")
    # default first SAFE (index=0), second Priced (index=1), others SAFE
    default_type_index = 0 if r == 1 else 1 if r == 2 else 0
    round_type = st.sidebar.selectbox(
        f"Round {r} Type",
        options=["SAFE", "Priced"],
        index=default_type_index,
        key=f"round_type_{r}"
    )
    round_dict = {"round_number": r, "type": round_type}
    
    if round_type == "SAFE":
        num_safe = st.sidebar.number_input(f"Round {r} Number of SAFE Notes", value=1, min_value=1, step=1, key=f"safe_num_{r}")
        safes = []
        for j in range(num_safe):
            st.sidebar.markdown(f"**SAFE {j+1} Terms for Round {r}:**")
            inv = st.sidebar.number_input(f"SAFE {j+1} Investment Amount ($)", value=1_000_000, step=1000, key=f"safe_investment_{r}_{j}")
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
                                                                    value=4_000_000, step=500_000, key=f"pre_money_{r}")
        # Target option pool percentage: default 0% for Round 1; 10% for rounds 2+.
        if r == 1:
            pool_target = st.sidebar.number_input(f"Round {r} Option Pool Target (%)", value=0.0, step=1.0, key=f"option_pool_{r}")
        else:
            pool_target = st.sidebar.number_input(f"Round {r} Option Pool Target (%)", value=10.0, step=1.0, key=f"option_pool_{r}")
        round_dict["option_pool"] = pool_target / 100.0
        round_dict["investment"] = st.sidebar.number_input(f"Round {r} Investment Amount ($)",
                                                           value=1_000_000, step=100_000, key=f"round_investment_{r}")
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
    # Will hold, for each round, the hypothetical “priced‐round” cap table
    hypothetical_shares = {}
    if rnd["type"] == "SAFE":
        # ---- ACTUAL SAFE HANDLING ----
        # Record each SAFE note
        for note in rnd["safes"]:
            note["round_number"] = rnd["round_number"]
            outstanding_safe_notes.append(note)

        # Build snapshot of the actual SAFE state
        details = (
            f"SAFE Round {rnd['round_number']}: Issued {len(rnd['safes'])} SAFE note(s). "
            f"Outstanding SAFE notes: {len(outstanding_safe_notes)}."
        )
        df_snap, _ = build_snapshot(details=details)
        snapshots.append({"round": round_label, "df": df_snap, "details": details})

        # ---- HYPOTHETICAL PRICED-ONLY CAP TABLE UP TO THIS ROUND ----
        S_base = F0
        hypo   = cap_table.copy() 

        # Iterate through all rounds up to and including this one
        for past in rounds[: rnd["round_number"]]:
            if past["type"] == "SAFE":
                # Treat SAFE as a priced round at its cap
                for note in past["safes"]:
                    I_amt = note["investment"]
                    V_pre = note["valuation_cap"]
                    P_target = 0.0
            else:
                # Use real priced-round terms
                I_amt    = past["investment"]
                V_pre    = past["pre_money_valuation"]
                P_target = past["option_pool"]

            # Solve for total shares T_hyp using same eqn
            args = (S_base, P_target, I_amt, V_pre, [], [])
            # Initial guess: if no pool, closed-form; else use same strategy as priced
            if P_target == 0:
                guess = [S_base / (1 - I_amt/(V_pre + I_amt))]
            else:
                guess = [S_base * 1.2]
            T_hyp, info, ier, msg = fsolve(total_shares_eq, guess, args=args, full_output=True)
            T_hyp = T_hyp[0] if ier == 1 else S_base

            # Back-calculate slices
            x_hyp = P_target * T_hyp
            s_hyp = 0
            p_hyp = (I_amt / (V_pre + I_amt)) * T_hyp

            # Accumulate into hypo dict
            if x_hyp:
                hypo[f"Option Pool R{past['round_number']}"] = x_hyp
            if p_hyp:
                label = f"Priced Equity R{past['round_number']}"
                hypo[label] = hypo.get(label, 0) + p_hyp

            # Advance base for next iteration
            S_base = T_hyp

        # Store hypothetical cap table for plotting
        hypothetical_shares[rnd["round_number"]] = hypo


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
            safe_label = f"SAFE R{rnd['round_number'] - 1} Conversion"
            issued_classes[safe_label] = s_calculated
            details += f"Converted {len(safe_investments)} SAFE note(s) (aggregate R = {R_total:.4f}) yield {s_calculated:,.0f} shares (conversion base = S₀ + option pool).\n"
            outstanding_safe_notes = []  # Clear all after conversion.
        inv_label = f"Priced Equity R{rnd['round_number']}"
        pool_label = f"Option Pool R{rnd['round_number']}"
        issued_classes[inv_label] = p_calculated
        issued_classes[pool_label] = x_calculated
        details += (f"Issued {p_calculated:,.0f} investor shares and {x_calculated:,.0f} option pool shares.\n"
                    f"Final fully diluted total T = {T_final:,.0f} shares (Investor fraction = {100*p_calculated/T_final:,.2f}%).")
        
        df_snap, _ = build_snapshot(details=details)
        snapshots.append({"round": round_label, "df": df_snap, "details": details, "intermediate": intermediate_df})
        
# ------------------------------------------------------------------------------
# DISPLAY SNAPSHOTS WITH SAFE vs HYPOTHETICAL-ONLY CHARTS
# ------------------------------------------------------------------------------

st.header("Cap Table Snapshots by Round")


# Color Mapping
all_labels = set()
for snap in snapshots:
    df = snap["df"]
    numeric = df["Shares"].apply(lambda x: isinstance(x, (int, float)))
    all_labels.update(df.loc[numeric, "Shareholder"].tolist())

# 2) Collect all slice labels from hypothetical cap tables
for hypo in hypothetical_shares.values():
    all_labels.update(hypo.keys())

# 3) Build a stable mapping from label -> color
cmap = plt.get_cmap("tab20")
sorted_labels = sorted(all_labels)
color_map = {
    label: cmap(i / len(sorted_labels))
    for i, label in enumerate(sorted_labels)
}

for idx, snap in enumerate(snapshots):
    rnd        = rounds[idx]
    round_type = rnd["type"]

    st.subheader(snap["round"])
    st.write(snap["details"])

    # ——— Show only the actual cap table DataFrame ———
    st.dataframe(snap["df"])

    # ——— For PRICED rounds only: two fixed-size, high-contrast pies side by side ———
    if round_type == "Priced":
        st.write("")  # force a line break
        left, right = st.columns(2)

        # Prepare actual cap table data
        df_act     = snap["df"][snap["df"]["Shares"].apply(lambda x: isinstance(x, (int, float)))]
        labels_act = df_act["Shareholder"].tolist()
        sizes_act  = df_act["Shares"].tolist()

        # Prepare hypothetical priced‐only cap table data using exact formulas
        S_base = F0
        hypo   = cap_table.copy()   # cap_table is your dict of {founder_name: shares}


        for past in rounds[: idx + 1]:
            if past["type"] == "SAFE":
                # SAFE → conversion at cap: s = (R/(1-R)) * S_base
                for note in past["safes"]:
                    inv = note["investment"]
                    cap = note["valuation_cap"]
                    R   = inv / cap
                    s_hyp = (R / (1 - R)) * S_base
                    label = f"Priced Equity R{past['round_number']}"
                    hypo[label] = hypo.get(label, 0) + s_hyp
                    S_base += s_hyp
            else:
                # Priced round → solve closed‐form:
                # fr = I/(V_pre + I), T = S_base/(1 - P_target - fr)
                inv      = past["investment"]
                V_pre    = past["pre_money_valuation"]
                P_target = past["option_pool"]
                fr       = inv / (V_pre + inv)
                T_hyp    = S_base / (1 - P_target - fr)
                x_hyp    = P_target * T_hyp
                p_hyp    = fr * T_hyp

                if x_hyp > 0:
                    hypo[f"Option Pool R{past['round_number']}"] = x_hyp
                if p_hyp > 0:
                    hypo[f"Priced Equity R{past['round_number']}"] = p_hyp

                S_base = T_hyp

        labels_hyp = list(hypo.keys())
        sizes_hyp  = list(hypo.values())

        # Unified color mapping for consistency
        all_labels = labels_act + [lbl for lbl in labels_hyp if lbl not in labels_act]
        cmap       = plt.get_cmap("tab20")
        color_map  = {lbl: cmap(i) for i, lbl in enumerate(all_labels)}
        
        # sort actual slices ascending
        act_pairs = sorted(zip(sizes_act, labels_act), key=lambda x: x[0])
        sizes_act_sorted, labels_act_sorted = zip(*act_pairs)

        # sort hypothetical slices ascending
        hyp_pairs = sorted(zip(sizes_hyp, labels_hyp), key=lambda x: x[0])
        sizes_hyp_sorted, labels_hyp_sorted = zip(*hyp_pairs)

        # ── Combine both pies into a single figure with two subplots ──
        fig, (ax_act, ax_hyp) = plt.subplots(
            1, 2,
            figsize=(12, 6),
            facecolor='black',
            constrained_layout=True
        )

        # Left subplot: Actual Cap Table (sorted)
        ax_act.pie(
            sizes_act_sorted,
            labels=labels_act_sorted,
            colors=[color_map[l] for l in labels_act_sorted],
            autopct="%1.1f%%",
            textprops={"color": "white", "fontsize": 12},
            wedgeprops={"edgecolor": "black", "linewidth": 1}
        )
        ax_act.set_title("Actual Cap Table", color="white", fontsize=16)
        ax_act.set_aspect("equal")

        # Right subplot: Hypothetical Priced-Only (sorted)
        ax_hyp.pie(
            sizes_hyp_sorted,
            labels=labels_hyp_sorted,
            colors=[color_map[l] for l in labels_hyp_sorted],
            autopct="%1.1f%%",
            textprops={"color": "white", "fontsize": 12},
            wedgeprops={"edgecolor": "black", "linewidth": 1}
        )
        ax_hyp.set_title("Hypothetical Priced-Only", color="white", fontsize=16)
        ax_hyp.set_aspect("equal")

        # Display the combined figure
        st.pyplot(fig, use_container_width=True)




            
# ------------------------------------------------------------------------------
# FINAL CUMULATIVE CAP TABLE WITH HYPOTHETICAL PRICED-ONLY COLUMN
# ------------------------------------------------------------------------------
st.header("Final Cumulative Cap Table")

# 1) Build the actual final cap table
final_cap = {}
final_cap.update(cap_table)
final_cap.update(issued_classes)

# 2) Build the hypothetical “priced-only” final cap table
#    (start from founders individually)
hypo_final = cap_table.copy()
S_base = sum(cap_table.values())

for past in rounds:
    if past["type"] == "SAFE":
        # SAFE → priced conversion at cap
        for note in past["safes"]:
            inv    = note["investment"]
            capval = note["valuation_cap"]
            r      = inv / capval
            s      = (r / (1 - r)) * S_base
            key    = f"Priced Equity R{past['round_number']}"
            hypo_final[key] = hypo_final.get(key, 0) + s
            S_base += s
    else:
        # Priced round → exact closed-form
        inv      = past["investment"]
        V_pre    = past["pre_money_valuation"]
        P_target = past["option_pool"]
        fr       = inv / (V_pre + inv)               # investor fraction
        T_hyp    = S_base / (1 - P_target - fr)      # total post-round shares
        x        = P_target * T_hyp                  # option pool
        p        = fr * T_hyp                        # investor shares

        if x > 0:
            hypo_final[f"Option Pool R{past['round_number']}"] = x
        if p > 0:
            hypo_final[f"Priced Equity R{past['round_number']}"] = p

        S_base = T_hyp

# 3) Compute totals for percentages
total_actual = sum(final_cap.values())
total_hyp    = sum(hypo_final.values())

# 4) Assemble a DataFrame with both actual and hypothetical shares
rows = []
for holder in sorted(set(final_cap) | set(hypo_final)):
    actual = final_cap.get(holder, 0)
    hyp    = hypo_final.get(holder, 0)
    rows.append({
        "Shareholder":           holder,
        "Actual Shares":         actual,
        "Hypothetical Shares":   hyp,
        "% Ownership (Actual)":  f"{100 * actual / total_actual:,.2f}%",
        "% Ownership (Hypothetical)": f"{100 * hyp / total_hyp:,.2f}%"
    })

st.dataframe(pd.DataFrame(rows))





