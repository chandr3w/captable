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
# Founders & Pre-Financing Setup
# ------------------------------------------------------------------------------
st.sidebar.header("Initial Setup")
S0 = st.sidebar.number_input("Founders' Initial Common Shares (S₀)", value=1_000_000, step=1000)

st.sidebar.header("Pre-Financing Shareholders")
num_founders = st.sidebar.selectbox("Number of Founders", options=range(1, 6), index=1)
founder_shares = {}
for i in range(num_founders):
    col1, col2 = st.sidebar.columns(2)
    with col1:
        name = st.text_input(f"Founder {i+1} Name", value=f"Founder {i+1}", key=f"founder_name_{i}")
    with col2:
        pct = st.number_input(f"Founder {i+1} Ownership (%)", min_value=0.0, max_value=100.0,
                               value=100/num_founders, step=1.0, key=f"founder_pct_{i}")
    founder_shares[name] = S0 * (pct/100)

F0 = sum(founder_shares.values())
cap_table = founder_shares.copy()

# ------------------------------------------------------------------------------
# Financing Rounds Setup (Up to 5 Rounds)
# ------------------------------------------------------------------------------
st.sidebar.header("Financing Rounds")
total_rounds = st.sidebar.selectbox("Total Number of Rounds", options=[1,2,3,4,5], index=1)

rounds = []
for r in range(1, total_rounds+1):
    st.sidebar.subheader(f"Round {r} Settings")
    default_type_index = 0 if r == 1 else 1 if r == 2 else 0
    round_type = st.sidebar.selectbox(f"Round {r} Type", options=["SAFE", "Priced"],
                                      index=default_type_index, key=f"round_type_{r}")
    rd = {"round_number": r, "type": round_type}
    if round_type == "SAFE":
        num_safe = st.sidebar.number_input(f"Round {r} # of SAFE Notes", min_value=1, value=1, step=1, key=f"safe_num_{r}")
        safes = []
        for j in range(num_safe):
            st.sidebar.markdown(f"**SAFE {j+1} Terms (Round {r})**")
            inv = st.sidebar.number_input(f"SAFE {j+1} Investment ($)", value=1_000_000, step=1000, key=f"safe_inv_{r}_{j}")
            cap = st.sidebar.number_input(f"SAFE {j+1} Cap ($)", value=5_000_000, step=100_000, key=f"safe_cap_{r}_{j}")
            disc = st.sidebar.selectbox(f"SAFE {j+1} Discount (%)", options=[0,5,10,15,20,25,30], key=f"safe_disc_{r}_{j}")
            safes.append({"investment": inv, "valuation_cap": cap, "discount": disc/100})
        rd["safes"] = safes
    else:
        rd["pre_money_valuation"] = st.sidebar.number_input(f"Round {r} Pre-Money Valuation ($)", value=4_000_000, step=500_000, key=f"pre_{r}")
        pool_default = 0.0 if r == 1 else 10.0
        rd["option_pool"] = st.sidebar.number_input(f"Round {r} Option Pool Target (%)", value=pool_default, step=1.0, key=f"pool_{r}")/100
        rd["investment"] = st.sidebar.number_input(f"Round {r} Investment Amount ($)", value=1_000_000, step=100_000, key=f"inv_{r}")
    rounds.append(rd)

# ------------------------------------------------------------------------------
# Simulation Boilerplate
# ------------------------------------------------------------------------------
st.header("Round-by-Round Simulation Results")
outstanding_safe_notes = []
issued_classes = {}
snapshots = []

def build_snapshot(details=""):
    allc = {}
    allc.update(cap_table)
    allc.update(issued_classes)
    total = sum(allc.values())
    rows = [{"Shareholder": k, "Shares": v, "% Ownership": f"{100*v/total:,.2f}%"} for k,v in allc.items()]
    return pd.DataFrame(rows), details

def total_shares_eq(T, S0, P_target, I_amt, V_pre, safe_invs, safe_caps):
    x = P_target * T
    if safe_invs:
        R = sum(inv/cap for inv,cap in zip(safe_invs, safe_caps))
        s = (R/(1-R))*(S0 + x)
    else:
        s = 0
    p = (I_amt/(V_pre + I_amt))*T
    return S0 + x + s + p - T

# --- Run Each Round ---
for rnd in rounds:
    lbl = f"Round {rnd['round_number']} - {rnd['type']}"
    # SAFE handling
    if rnd["type"] == "SAFE":
        for note in rnd["safes"]:
            note["round_number"] = rnd["round_number"]
            outstanding_safe_notes.append(note)
        df_snap, details = build_snapshot(f"SAFE Round {rnd['round_number']}: issued {len(rnd['safes'])} SAFE(s).")
        snapshots.append({"round": lbl, "df": df_snap, "details": details})
    else:
        # collect outstanding SAFEs
        safe_invs = [n["investment"] for n in outstanding_safe_notes]
        safe_caps = [n["valuation_cap"] for n in outstanding_safe_notes]
        args = (F0, rnd["option_pool"], rnd["investment"], rnd["pre_money_valuation"], safe_invs, safe_caps)
        guess = [F0 * 1.2] if safe_invs else [F0/(1 - rnd["option_pool"] - rnd["investment"]/(rnd["pre_money_valuation"]+rnd["investment"]))]
        sol, info, ier, msg = fsolve(total_shares_eq, guess, args=args, full_output=True)
        if ier != 1:
            st.error(f"Round {rnd['round_number']} failed to converge: {msg}")
            st.stop()
        T_final = sol[0]
        # back-calc
        x_calc = rnd["option_pool"]*T_final
        R = sum(inv/cap for inv,cap in zip(safe_invs, safe_caps)) if safe_invs else 0
        s_calc = (R/(1-R))*(F0 + x_calc) if safe_invs else 0
        p_calc = (rnd["investment"]/(rnd["pre_money_valuation"]+rnd["investment"]))*T_final
        # record
        if safe_invs:
            issued_classes[f"SAFE R{rnd['round_number']} Conversion"] = s_calc
            outstanding_safe_notes = []
        issued_classes[f"Priced Equity R{rnd['round_number']}"] = p_calc
        issued_classes[f"Option Pool R{rnd['round_number']}"] = x_calc
        details = (
            f"Priced Round {rnd['round_number']}: Pre-money ${rnd['pre_money_valuation']:,.0f}, "
            f"Investment ${rnd['investment']:,.0f}. "
            f"Issued {p_calc:,.0f} shares to investors and {x_calc:,.0f} option pool shares."
        )
        df_snap, _ = build_snapshot(details)
        snapshots.append({"round": lbl, "df": df_snap, "details": details})

# ------------------------------------------------------------------------------
# DISPLAY PIE CHARTS FOR EVERY ROUND
# ------------------------------------------------------------------------------
st.header("Cap Table Snapshots by Round")

for idx, snap in enumerate(snapshots):
    rnd_info = rounds[idx]
    rnum = rnd_info["round_number"]

    st.subheader(f"After Round {rnum}")
    st.write(snap["details"])
    st.dataframe(snap["df"])

    # Prepare Actual
    df_act = snap["df"][snap["df"]["Shares"].apply(lambda x: isinstance(x, (int, float)))]
    labels_act = df_act["Shareholder"].tolist()
    sizes_act = df_act["Shares"].tolist()

    # Prepare Hypothetical Priced-Only up to this round
    hypo = cap_table.copy()
    S_base = F0
    for past in rounds[: idx+1]:
        if past["type"] == "SAFE":
            for note in past["safes"]:
                inv, capval = note["investment"], note["valuation_cap"]
                r = inv/capval
                s = (r/(1-r))*S_base
                key = f"SAFE→Priced R{past['round_number']}"
                hypo[key] = hypo.get(key, 0) + s
                S_base += s
        else:
            fr = past["investment"] / (past["pre_money_valuation"] + past["investment"])
            T_hyp = S_base / (1 - past["option_pool"] - fr)
            x = past["option_pool"] * T_hyp
            p = fr * T_hyp
            hypo[f"Option Pool R{past['round_number']}"] = x
            hypo[f"Priced Equity R{past['round_number']}"] = p
            S_base = T_hyp

    labels_hyp = list(hypo.keys())
    sizes_hyp = list(hypo.values())

    # Unified color map
    all_lbls = labels_act + [l for l in labels_hyp if l not in labels_act]
    cmap = plt.get_cmap("tab20")
    color_map = {lbl: cmap(i/len(all_lbls)) for i,lbl in enumerate(all_lbls)}

    # Sort ascending for visibility
    act_pairs = sorted(zip(sizes_act, labels_act), key=lambda x: x[0])
    hyp_pairs = sorted(zip(sizes_hyp, labels_hyp), key=lambda x: x[0])
    sizes_act_s, labels_act_s = zip(*act_pairs)
    sizes_hyp_s, labels_hyp_s = zip(*hyp_pairs)

    # Draw
    fig, (ax_act, ax_hyp) = plt.subplots(1, 2, figsize=(12,6), facecolor='black', constrained_layout=True)
    ax_act.pie(sizes_act_s, labels=labels_act_s,
               colors=[color_map[l] for l in labels_act_s],
               autopct="%1.1f%%", textprops={"color":"white","fontsize":12},
               wedgeprops={"edgecolor":"black","linewidth":1})
    ax_act.set_title(f"Actual Cap Table After R{rnum}", color="white", fontsize=16)
    ax_act.set_aspect("equal")

    ax_hyp.pie(sizes_hyp_s, labels=labels_hyp_s,
               colors=[color_map[l] for l in labels_hyp_s],
               autopct="%1.1f%%", textprops={"color":"white","fontsize":12},
               wedgeprops={"edgecolor":"black","linewidth":1})
    ax_hyp.set_title(f"Hypothetical Priced-Only After R{rnum}", color="white", fontsize=16)
    ax_hyp.set_aspect("equal")

    st.pyplot(fig, use_container_width=True)

# ------------------------------------------------------------------------------
# FINAL CUMULATIVE CAP TABLE
# ------------------------------------------------------------------------------
st.header("Final Cumulative Cap Table")

final_cap = {**cap_table, **issued_classes}
hypo_final = cap_table.copy()
S_base = sum(cap_table.values())
for past in rounds:
    if past["type"] == "SAFE":
        for note in past["safes"]:
            inv, capval = note["investment"], note["valuation_cap"]
            r = inv/capval
            s = (r/(1-r))*S_base
            hypo_final[f"SAFE→Priced R{past['round_number']}"] = hypo_final.get(f"SAFE→Priced R{past['round_number']}", 0) + s
            S_base += s
    else:
        fr = past["investment"] / (past["pre_money_valuation"]+past["investment"])
        T_hyp = S_base / (1 - past["option_pool"] - fr)
        x = past["option_pool"] * T_hyp
        p = fr * T_hyp
        hypo_final[f"Option Pool R{past['round_number']}"] = x
        hypo_final[f"Priced Equity R{past['round_number']}"] = p
        S_base = T_hyp

total_actual = sum(final_cap.values())
total_hyp = sum(hypo_final.values())

rows = []
for holder in sorted(set(final_cap) | set(hypo_final)):
    a = final_cap.get(holder, 0)
    h = hypo_final.get(holder, 0)
    rows.append({
        "Shareholder": holder,
        "Actual Shares": a,
        "Hypothetical Shares": h,
        "% Ownership (Actual)": f"{100*a/total_actual:,.2f}%",
        "% Ownership (Hypothetical)": f"{100*h/total_hyp:,.2f}%"
    })

st.dataframe(pd.DataFrame(rows))


# In[ ]:




