# ============================================================
# MyTimes 6-File System — Emergency Reallocation Engine
# ============================================================
import pandas as pd


EMERGENCY_LOG_COLUMNS = [
    "case_no", "emergency_type", "emergency_reason", "emergency_lecturer",
    "class_id", "subject_code", "class_group",
    "original_week_before", "replacement_week", "original_week_continue",
    "replacement_lecturer", "subject_KS", "replacement_weeks",
    "KS_before_replacement", "KS_added_full_class", "KS_after_replacement",
    "minimum_KS", "maximum_KS", "remaining_capacity_after",
    "same_subject_experience", "eligibility_note", "emergency_decision_reason",
    "KS_calculation_method", "status"
]


def ensure_emergency_log(session_state):
    if "emergency_log" not in session_state or session_state["emergency_log"] is None:
        session_state["emergency_log"] = pd.DataFrame(columns=EMERGENCY_LOG_COLUMNS)


def _safe_bool(x):
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in {"true", "1", "yes", "y", "aktif", "active"}


def _decision_reason(row, emergency_reason):
    reasons = []
    if int(row.get("same_subject_experience", 0)) == 1:
        reasons.append("already teaches / has experience with the same subject")
    if bool(row.get("below_min_before", False)) and bool(row.get("meets_min_after", False)):
        reasons.append("helps the lecturer reach the minimum KS requirement")
    elif bool(row.get("below_min_before", False)):
        reasons.append("improves an underload lecturer's KS position")
    reasons.append("still within maximum KS after taking the full class credit")
    if emergency_reason:
        reasons.append(f"emergency reason: {emergency_reason}")
    return "; ".join(reasons).capitalize() + "."


def compute_emergency_reallocation(
    df_assign,
    df_summary,
    emergency_log,
    emergency_lecturer,
    start_week,
    end_week,
    emergency_reason="",
    emergency_type="Temporary class replacement",
):
    """
    Business rule:
    - Emergency replacement is counted as FULL subject/class KS per replaced class.
    - Example: if a replacement lecturer currently has 8 KS and replaces one 4-KS class,
      the new emergency-adjusted load becomes 12 KS. Two 4-KS classes become 16 KS.
    - Candidate selection prioritises:
        1. lecturer who has same-subject experience,
        2. lecturer below minimum KS who can reach/improve toward minimum,
        3. lecturer with lower current KS,
        4. lecturer who remains within maximum KS.
    """
    if df_assign.empty or df_summary.empty:
        return pd.DataFrame(columns=EMERGENCY_LOG_COLUMNS)

    emergency_classes = df_assign[df_assign["pensyarah_utama"] == emergency_lecturer].copy()
    if emergency_classes.empty:
        return pd.DataFrame(columns=EMERGENCY_LOG_COLUMNS)

    # Current base KS from fair allocation. Include minimum and maximum rules.
    required_cols = [
        "pensyarah", "jumlah_KS", "minimum_KS", "maksimum_KS", "aktif",
        "minggu_mula_available", "minggu_akhir_available", "senarai_subjek"
    ]
    available_cols = [c for c in required_cols if c in df_summary.columns]
    current_load = df_summary[available_cols].copy()
    if "minimum_KS" not in current_load.columns:
        current_load["minimum_KS"] = 0
    if "maksimum_KS" not in current_load.columns:
        current_load["maksimum_KS"] = 999
    current_load = current_load.rename(columns={"jumlah_KS": "jumlah_KS_asal"})

    # Add previous emergency load so repeated emergency cases are calculated correctly.
    if emergency_log is not None and not emergency_log.empty:
        log = emergency_log.copy()
        # Backward-compatible with older log column names.
        if "replacement_lecturer" not in log.columns and "pensyarah_pengganti" in log.columns:
            log["replacement_lecturer"] = log["pensyarah_pengganti"]
        if "KS_added_full_class" not in log.columns and "KS_pengganti" in log.columns:
            log["KS_added_full_class"] = log["KS_pengganti"]
        if "status" in log.columns:
            log = log[log["status"] == "OK"]
        previous = (
            log.groupby("replacement_lecturer")["KS_added_full_class"]
            .sum()
            .reset_index()
            .rename(columns={"replacement_lecturer": "pensyarah", "KS_added_full_class": "KS_emergency_before"})
        ) if not log.empty else pd.DataFrame(columns=["pensyarah", "KS_emergency_before"])
        current_load = current_load.merge(previous, on="pensyarah", how="left")
    else:
        current_load["KS_emergency_before"] = 0.0

    current_load["KS_emergency_before"] = current_load["KS_emergency_before"].fillna(0.0)
    current_load["jumlah_KS_semasa"] = current_load["jumlah_KS_asal"].astype(float) + current_load["KS_emergency_before"].astype(float)
    current_load["minimum_KS"] = pd.to_numeric(current_load["minimum_KS"], errors="coerce").fillna(0).astype(float)
    current_load["maksimum_KS"] = pd.to_numeric(current_load["maksimum_KS"], errors="coerce").fillna(999).astype(float)
    current_load["aktif"] = current_load["aktif"].apply(_safe_bool)

    if emergency_log is None or emergency_log.empty:
        case_no = 1
    else:
        if "case_no" in emergency_log.columns:
            case_no = int(pd.to_numeric(emergency_log["case_no"], errors="coerce").max()) + 1
        elif "kes_no" in emergency_log.columns:
            case_no = int(pd.to_numeric(emergency_log["kes_no"], errors="coerce").max()) + 1
        else:
            case_no = 1

    rows = []

    for _, row in emergency_classes.iterrows():
        class_start = int(row["minggu_mula_kelas"])
        class_end = int(row["minggu_akhir_kelas"])
        overlap_start = max(class_start, int(start_week))
        overlap_end = min(class_end, int(end_week))

        if overlap_start > overlap_end:
            continue

        replacement_weeks = overlap_end - overlap_start + 1

        # FULL KS per replaced class, not prorated by week.
        ks_full_class = round(float(row["KS"]), 2)

        candidates = current_load[
            (current_load["pensyarah"] != emergency_lecturer)
            & (current_load["aktif"] == True)
            & (current_load["minggu_mula_available"] <= overlap_start)
            & (current_load["minggu_akhir_available"] >= overlap_end)
        ].copy()

        if candidates.empty:
            rows.append({
                "case_no": case_no,
                "emergency_type": emergency_type,
                "emergency_reason": emergency_reason,
                "emergency_lecturer": emergency_lecturer,
                "class_id": row["kelas_id"],
                "subject_code": row["kod_kursus"],
                "class_group": row["kelas_baru"],
                "original_week_before": f"{class_start}-{overlap_start - 1}" if class_start < overlap_start else "",
                "replacement_week": f"{overlap_start}-{overlap_end}",
                "original_week_continue": f"{overlap_end + 1}-{class_end}" if overlap_end < class_end else "",
                "replacement_lecturer": "NO ELIGIBLE CANDIDATE",
                "subject_KS": ks_full_class,
                "replacement_weeks": replacement_weeks,
                "KS_before_replacement": 0,
                "KS_added_full_class": ks_full_class,
                "KS_after_replacement": 0,
                "minimum_KS": 0,
                "maximum_KS": 0,
                "remaining_capacity_after": 0,
                "same_subject_experience": "No",
                "eligibility_note": "No active lecturer is available for the selected emergency weeks.",
                "emergency_decision_reason": "Failed because no active lecturer is available for the selected emergency period.",
                "KS_calculation_method": "Full subject/class KS per replaced class",
                "status": "FAILED",
            })
            continue

        candidates["same_subject_experience"] = candidates["senarai_subjek"].astype(str).apply(
            lambda x: 1 if str(row["kod_kursus"]) in x else 0
        )
        candidates["KS_after_replacement"] = candidates["jumlah_KS_semasa"] + ks_full_class
        candidates["remaining_capacity_after"] = candidates["maksimum_KS"] - candidates["KS_after_replacement"]
        candidates["below_min_before"] = candidates["jumlah_KS_semasa"] < candidates["minimum_KS"]
        candidates["meets_min_after"] = candidates["KS_after_replacement"] >= candidates["minimum_KS"]
        candidates["underload_gap_before"] = (candidates["minimum_KS"] - candidates["jumlah_KS_semasa"]).clip(lower=0)

        eligible = candidates[candidates["KS_after_replacement"] <= candidates["maksimum_KS"]].copy()

        if eligible.empty:
            rows.append({
                "case_no": case_no,
                "emergency_type": emergency_type,
                "emergency_reason": emergency_reason,
                "emergency_lecturer": emergency_lecturer,
                "class_id": row["kelas_id"],
                "subject_code": row["kod_kursus"],
                "class_group": row["kelas_baru"],
                "original_week_before": f"{class_start}-{overlap_start - 1}" if class_start < overlap_start else "",
                "replacement_week": f"{overlap_start}-{overlap_end}",
                "original_week_continue": f"{overlap_end + 1}-{class_end}" if overlap_end < class_end else "",
                "replacement_lecturer": "NO ELIGIBLE CANDIDATE",
                "subject_KS": ks_full_class,
                "replacement_weeks": replacement_weeks,
                "KS_before_replacement": 0,
                "KS_added_full_class": ks_full_class,
                "KS_after_replacement": 0,
                "minimum_KS": 0,
                "maximum_KS": 0,
                "remaining_capacity_after": 0,
                "same_subject_experience": "No",
                "eligibility_note": "All available lecturers would exceed maximum KS if this full class credit is added.",
                "emergency_decision_reason": "Failed because all candidate lecturers exceed maximum KS after adding the full class credit.",
                "KS_calculation_method": "Full subject/class KS per replaced class",
                "status": "FAILED",
            })
            continue

        # Priority: same subject, underload who reaches minimum, bigger underload gap, lowest current load.
        eligible = eligible.sort_values(
            ["same_subject_experience", "meets_min_after", "underload_gap_before", "jumlah_KS_semasa", "remaining_capacity_after"],
            ascending=[False, False, False, True, False]
        )
        selected = eligible.iloc[0]
        replacement_lecturer = selected["pensyarah"]

        # Update current_load immediately so if the same lecturer receives 2 classes,
        # total KS becomes 8 -> 12 -> 16, etc.
        current_load.loc[current_load["pensyarah"] == replacement_lecturer, "jumlah_KS_semasa"] = float(selected["KS_after_replacement"])

        rows.append({
            "case_no": case_no,
            "emergency_type": emergency_type,
            "emergency_reason": emergency_reason,
            "emergency_lecturer": emergency_lecturer,
            "class_id": row["kelas_id"],
            "subject_code": row["kod_kursus"],
            "class_group": row["kelas_baru"],
            "original_week_before": f"{class_start}-{overlap_start - 1}" if class_start < overlap_start else "",
            "replacement_week": f"{overlap_start}-{overlap_end}",
            "original_week_continue": f"{overlap_end + 1}-{class_end}" if overlap_end < class_end else "",
            "replacement_lecturer": replacement_lecturer,
            "subject_KS": ks_full_class,
            "replacement_weeks": replacement_weeks,
            "KS_before_replacement": round(float(selected["jumlah_KS_semasa"]), 2),
            "KS_added_full_class": ks_full_class,
            "KS_after_replacement": round(float(selected["KS_after_replacement"]), 2),
            "minimum_KS": round(float(selected["minimum_KS"]), 2),
            "maximum_KS": round(float(selected["maksimum_KS"]), 2),
            "remaining_capacity_after": round(float(selected["remaining_capacity_after"]), 2),
            "same_subject_experience": "Yes" if int(selected["same_subject_experience"]) == 1 else "No",
            "eligibility_note": "Eligible: active, available during emergency weeks, and within maximum KS after full-credit replacement.",
            "emergency_decision_reason": _decision_reason(selected, emergency_reason),
            "KS_calculation_method": "Full subject/class KS per replaced class",
            "status": "OK",
        })

    return pd.DataFrame(rows, columns=EMERGENCY_LOG_COLUMNS)
