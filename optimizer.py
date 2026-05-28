# ============================================================
# MyTimes 6-File System — Fair KS Optimizer + Output Builder
# ============================================================
import pandas as pd
import streamlit as st

try:
    import pulp as pl
except Exception:
    pl = None

from config_styles import (
    LATE_ENTRY_CUTOFF_WEEK,
    MAX_SUBJECTS,
    MAX_CLASSES_SAME_SUBJECT,
    TARGET_KS,
    W_FAIRNESS,
    W_PREF,
)
from data_utils import get_pref_score, get_pref_label, get_compensation_points, get_preference_reason, is_available_for_class


def solve_allocation(dfc, dfl, pref):
    if pl is None:
        st.error("PuLP belum install. Sila install: pip install pulp")
        st.stop()

    classes = dfc["kelas_id"].tolist()
    lecturers = dfl["nama"].tolist()
    subjects = sorted(dfc["kod_kursus"].unique().tolist())

    credit = dfc.set_index("kelas_id")["ks"].astype(int).to_dict()
    cls_subject = dfc.set_index("kelas_id")["kod_kursus"].to_dict()
    active = dfl.set_index("nama")["active"].to_dict()
    class_rows = dfc.set_index("kelas_id")
    lect_rows = dfl.set_index("nama")

    wajib_ajar = [
        l for l in lecturers
        if active[l] and int(lect_rows.loc[l, "minggu_mula_available"]) <= LATE_ENTRY_CUTOFF_WEEK
    ]

    target_ks = round(int(dfc["ks"].sum()) / max(len(wajib_ajar), 1))

    prob = pl.LpProblem("MyTimes_Fair_KS", pl.LpMinimize)
    x = pl.LpVariable.dicts("x", (classes, lecturers), 0, 1, cat="Binary")
    y = pl.LpVariable.dicts("y", (lecturers, subjects), 0, 1, cat="Binary")
    under = pl.LpVariable.dicts("under", lecturers, lowBound=0)
    over = pl.LpVariable.dicts("over", lecturers, lowBound=0)

    for c in classes:
        prob += pl.lpSum(x[c][l] for l in lecturers) == 1

    for l in lecturers:
        if not active[l]:
            for c in classes:
                prob += x[c][l] == 0

    for c in classes:
        for l in lecturers:
            if not is_available_for_class(lect_rows.loc[l], class_rows.loc[c]):
                prob += x[c][l] == 0

    for l in lecturers:
        total_load = pl.lpSum(credit[c] * x[c][l] for c in classes)
        start_week = int(lect_rows.loc[l, "minggu_mula_available"])

        if l in wajib_ajar:
            personal_min = int(lect_rows.loc[l, "min_ks"])
            personal_max = int(lect_rows.loc[l, "max_ks"])
            personal_target = min(max(target_ks, personal_min), personal_max)

            prob += total_load >= personal_min
            prob += total_load <= personal_max
            prob += pl.lpSum(x[c][l] for c in classes) >= 1
            prob += personal_target - total_load <= under[l]
            prob += total_load - personal_target <= over[l]
        else:
            prob += under[l] == 0
            prob += over[l] == 0
            if active[l] and start_week > LATE_ENTRY_CUTOFF_WEEK:
                prob += total_load == 0

    for c in classes:
        s = cls_subject[c]
        for l in lecturers:
            prob += x[c][l] <= y[l][s]

    for l in lecturers:
        prob += pl.lpSum(y[l][s] for s in subjects) <= MAX_SUBJECTS
        for s in subjects:
            subject_classes = [c for c in classes if cls_subject[c] == s]
            prob += pl.lpSum(x[c][l] for c in subject_classes) <= MAX_CLASSES_SAME_SUBJECT

    compensation_map = lect_rows["compensation_points"].to_dict() if "compensation_points" in lect_rows.columns else {l: 0 for l in lecturers}

    def adjusted_pref_reward(lname, subject):
        base = get_pref_score(lname, subject, pref)
        # Compensation only boosts subjects that are actually in the lecturer preference list.
        # It must not reward a non-preferred assignment.
        return base + int(compensation_map.get(lname, 0)) if base > 0 else 0

    preference_reward = pl.lpSum(
        credit[c] * adjusted_pref_reward(l, cls_subject[c]) * x[c][l]
        for c in classes for l in lecturers
    )
    fairness_penalty = pl.lpSum(under[l] + over[l] for l in wajib_ajar)
    prob += W_FAIRNESS * fairness_penalty - W_PREF * preference_reward

    solver = pl.PULP_CBC_CMD(msg=False, timeLimit=240)
    prob.solve(solver)
    status = pl.LpStatus[prob.status]

    assigned_rows = []
    if status == "Optimal":
        for c in classes:
            for l in lecturers:
                if float(pl.value(x[c][l]) or 0) > 0.5:
                    assigned_rows.append({"kelas_id": c, "pensyarah": l})

    return status, pd.DataFrame(assigned_rows), target_ks


def build_outputs(dfc_active, df_closed, dfl, pref, assigned_df, target_ks=None):
    lect_lookup = dfl.set_index("nama")
    class_lookup = dfc_active.set_index("kelas_id")
    rows = []

    if not assigned_df.empty:
        for _, ar in assigned_df.iterrows():
            cid = ar["kelas_id"]
            lname = ar["pensyarah"]
            r = class_lookup.loc[cid]
            subj = r["kod_kursus"]
            lrow = lect_lookup.loc[lname]
            start_week = int(lrow["minggu_mula_available"])

            needs_temp_cover = start_week > int(r["minggu_mula_kelas"]) and start_week <= LATE_ENTRY_CUTOFF_WEEK
            pref_label = get_pref_label(lname, subj, dfl)
            pref_score = get_pref_score(lname, subj, pref)
            comp_points = get_compensation_points(pref_label)
            reason = get_preference_reason(pref_label)

            rows.append({
                "kelas_id": cid,
                "kod_kursus": subj,
                "kelas_baru": r["kelas_baru"],
                "status_kelas": r["status_kelas"],
                "KS": int(r["ks"]),
                "saiz_kelas": int(r.get("saiz_kelas", 0)),
                "pensyarah_utama": lname,
                "peranan": lrow["peranan"],
                "preference_match": pref_label,
                "preference_score": pref_score,
                "allocation_reason": reason,
                "compensation_points_next_semester": comp_points,
                "minggu_mula_kelas": int(r["minggu_mula_kelas"]),
                "minggu_akhir_kelas": int(r["minggu_akhir_kelas"]),
                "minggu_mula_pensyarah": start_week,
                "minggu_akhir_pensyarah": int(lrow["minggu_akhir_available"]),
                "perlu_cover_sementara": "YA" if needs_temp_cover else "TIDAK",
                "minggu_cover_sementara": f"{int(r['minggu_mula_kelas'])}-{start_week - 1}" if needs_temp_cover else "",
                "pensyarah_cover_sementara": "",
                "catatan": "Pensyarah masuk lewat. Perlu pensyarah sementara cover minggu awal." if needs_temp_cover else "",
                "perincian": r.get("perincian", ""),
            })

    df_assign = pd.DataFrame(rows)

    if not df_assign.empty:
        for idx, row in df_assign.iterrows():
            if row["perlu_cover_sementara"] == "YA":
                same_subject = df_assign[
                    (df_assign["kod_kursus"] == row["kod_kursus"])
                    & (df_assign["pensyarah_utama"] != row["pensyarah_utama"])
                ].copy()
                if not same_subject.empty:
                    lecturer_load = (
                        df_assign.groupby("pensyarah_utama")["KS"].sum().reset_index()
                        .rename(columns={"pensyarah_utama": "calon_cover", "KS": "jumlah_KS"})
                    )
                    candidate = same_subject[["pensyarah_utama"]].drop_duplicates().rename(columns={"pensyarah_utama": "calon_cover"})
                    candidate = candidate.merge(lecturer_load, on="calon_cover", how="left").sort_values("jumlah_KS", ascending=True)
                    if not candidate.empty:
                        df_assign.loc[idx, "pensyarah_cover_sementara"] = candidate.iloc[0]["calon_cover"]

    summary_rows = []
    for _, lrow in dfl.iterrows():
        lname = lrow["nama"]
        tmp = df_assign[df_assign["pensyarah_utama"] == lname] if not df_assign.empty else pd.DataFrame()
        total_ks = int(tmp["KS"].sum()) if not tmp.empty else 0
        total_class = int(tmp["kelas_id"].nunique()) if not tmp.empty else 0
        subjects = sorted(tmp["kod_kursus"].unique().tolist()) if not tmp.empty else []
        if not tmp.empty and "preference_score" in tmp.columns:
            # KS-weighted preference satisfaction: Choice 1=100, Choice 2=80, ..., Not Preferred=0.
            preference_score = round((tmp["preference_score"] * tmp["KS"]).sum() / max(tmp["KS"].sum(), 1), 1)
            worst_pref = tmp.sort_values("preference_score").iloc[0]["preference_match"]
            compensation_points = int(tmp["compensation_points_next_semester"].max())
        else:
            preference_score = 0.0
            worst_pref = "Unassigned"
            compensation_points = get_compensation_points("Unassigned") if bool(lrow["active"]) else 0

        if not bool(lrow["active"]):
            status_load = "TIDAK AKTIF / CUTI"
        elif int(lrow["minggu_mula_available"]) > LATE_ENTRY_CUTOFF_WEEK:
            status_load = "MASUK SELEPAS MINGGU 10"
        elif total_ks < int(lrow["min_ks"]):
            status_load = "UNDERLOAD"
        elif total_ks > int(lrow["max_ks"]):
            status_load = "OVERLOAD"
        else:
            status_load = "ADIL"

        detail_list = []
        if not tmp.empty:
            for subj, g in tmp.groupby("kod_kursus"):
                cls = ", ".join(g["kelas_baru"].astype(str).tolist())
                cr = int(g["KS"].sum())
                detail_list.append(f"{subj}: {cr} KS ({cls})")

        summary_rows.append({
            "pensyarah": lname,
            "peranan": lrow["peranan"],
            "status_pensyarah": lrow["status"],
            "aktif": bool(lrow["active"]),
            "minggu_mula_available": int(lrow["minggu_mula_available"]),
            "minggu_akhir_available": int(lrow["minggu_akhir_available"]),
            "minimum_KS": int(lrow["min_ks"]),
            "maksimum_KS": int(lrow["max_ks"]),
            "jumlah_KS": total_ks,
            "jumlah_kelas": total_class,
            "bil_subjek": len(subjects),
            "senarai_subjek": ", ".join(subjects),
            "perincian_mengajar": " | ".join(detail_list),
            "beza_dari_target": total_ks - (target_ks if target_ks is not None else TARGET_KS),
            "status_load": status_load,
            "preference_score": preference_score,
            "lowest_preference_match": worst_pref,
            "compensation_points_next_semester": compensation_points,
            "next_semester_priority": "High" if compensation_points >= 20 else ("Medium" if compensation_points >= 10 else "Normal"),
        })

    df_summary = pd.DataFrame(summary_rows)
    assigned_ids = set(df_assign["kelas_id"]) if not df_assign.empty else set()
    df_unassigned = dfc_active[~dfc_active["kelas_id"].isin(assigned_ids)].copy()
    df_temp_cover = df_assign[df_assign["perlu_cover_sementara"] == "YA"].copy() if not df_assign.empty else pd.DataFrame()

    df_status = pd.DataFrame([{
        "jumlah_kelas_aktif": len(dfc_active),
        "jumlah_kelas_tutup": len(df_closed),
        "kelas_diagih": len(assigned_ids),
        "kelas_tidak_diagih": len(df_unassigned),
        "jumlah_KS_aktif": int(dfc_active["ks"].sum()),
        "KS_diagih": int(df_assign["KS"].sum()) if not df_assign.empty else 0,
        "jumlah_pensyarah": len(dfl),
        "pensyarah_aktif": int(dfl["active"].sum()),
        "pensyarah_adil": int((df_summary["status_load"] == "ADIL").sum()),
        "pensyarah_underload": int((df_summary["status_load"] == "UNDERLOAD").sum()),
        "pensyarah_overload": int((df_summary["status_load"] == "OVERLOAD").sum()),
        "kes_cover_sementara": len(df_temp_cover),
        "target_purata_KS": target_ks if target_ks is not None else TARGET_KS,
    }])

    return df_assign, df_summary, df_temp_cover, df_unassigned, df_status
