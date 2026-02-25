import numpy as np
import pandas as pd



import numpy as np
import pandas as pd

def add_vel_acc_from_position(
    df: pd.DataFrame,
    pos_cols=("Head_x", "Head_y", "Head_z"),
    time_col="ASIL",          # e.g., "ASIL" or "lsl_timestamp"
    fs=None,                # sampling rate in Hz (use if time_col is None)
    smooth_window=0,        # 0 = no smoothing, else odd int like 5, 9, 11
) -> pd.DataFrame:
    """
    Compute linear velocity and acceleration from position columns.

    Adds: vx, vy, vz, ax, ay, az

    Requirements:
      - Either provide time_col (seconds) OR fs (Hz).
      - Positions should be numeric; output units follow input units.

    smooth_window:
      - Optional moving average applied to position before differentiating.
      - Must be odd integer > 1. Example: 9.
    """
    out = df.copy()

    # --- Get time step(s)
    if time_col is not None:
        t = pd.to_numeric(out[time_col], errors="coerce").to_numpy()
        # dt between samples (same length as data); first dt copied from second
        dt = np.diff(t)
        if len(dt) == 0:
            raise ValueError("Not enough samples to differentiate.")
        # Protect against zeros/negatives (can happen if timestamps repeat/out of order)
        # We'll replace invalid dt with median of valid dt
        valid = dt > 0
        if not np.any(valid):
            raise ValueError("All dt are non-positive. Check timestamp column.")
        dt_med = np.median(dt[valid])
        dt = np.where(valid, dt, dt_med)
        dt = np.concatenate(([dt[0]], dt))  # length N
    else:
        if fs is None:
            raise ValueError("Provide either time_col or fs.")
        dt = np.full(len(out), 1.0 / float(fs))

    # --- Position array (N,3)
    P = out.loc[:, list(pos_cols)].apply(pd.to_numeric, errors="coerce").to_numpy()

    # Optional smoothing (moving average), helps reduce noisy derivatives
    if smooth_window and smooth_window > 1:
        if smooth_window % 2 == 0:
            raise ValueError("smooth_window must be an odd integer (e.g., 5, 9, 11).")
        # simple centered moving average per column
        P_s = pd.DataFrame(P).rolling(smooth_window, center=True, min_periods=1).mean().to_numpy()
    else:
        P_s = P

    # --- Velocity using central differences: v = dP/dt
    vx = np.empty(len(out)); vy = np.empty(len(out)); vz = np.empty(len(out))
    # forward/backward at ends, central in middle
    # For variable dt, use numpy.gradient with coordinates
    if time_col is not None:
        # numpy.gradient can take x-coordinates for non-uniform spacing
        gx = np.gradient(P_s[:, 0], t)
        gy = np.gradient(P_s[:, 1], t)
        gz = np.gradient(P_s[:, 2], t)
    else:
        # uniform spacing
        gx = np.gradient(P_s[:, 0], dt[0])
        gy = np.gradient(P_s[:, 1], dt[0])
        gz = np.gradient(P_s[:, 2], dt[0])

    out["vx"], out["vy"], out["vz"] = gx, gy, gz

    # --- Acceleration: a = dv/dt
    if time_col is not None:
        ax = np.gradient(out["vx"].to_numpy(), t)
        ay = np.gradient(out["vy"].to_numpy(), t)
        az = np.gradient(out["vz"].to_numpy(), t)
    else:
        ax = np.gradient(out["vx"].to_numpy(), dt[0])
        ay = np.gradient(out["vy"].to_numpy(), dt[0])
        az = np.gradient(out["vz"].to_numpy(), dt[0])

    out["ax"], out["ay"], out["az"] = ax, ay, az
    return out


# ---------------- Example usage ----------------
# If you have timestamps in seconds:
# df_mocap2 = add_vel_acc_from_position(df_mocap, time_col="ASIL", smooth_window=9)

# If you don't have timestamps and MoCap is 120 Hz:
# df_mocap2 = add_vel_acc_from_position(df_mocap, fs=120, smooth_window=9)

# Then select the new columns:
# df_mocap2[["vx","vy","vz","ax","ay","az"]].head()

EPS = 1e-9

def _robust_dt(t: np.ndarray) -> float:
    """Median dt, ignoring non-increasing timestamps."""
    t = np.asarray(t, dtype=float)
    dt = np.diff(t)
    dt = dt[dt > 0]
    return float(np.median(dt)) if dt.size else np.nan

def _mag(xyz: np.ndarray) -> np.ndarray:
    return np.linalg.norm(xyz, axis=1)

def _percentile(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, q)) if x.size else np.nan

def _basic_stats(prefix: str, x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {f"{prefix}_{k}": np.nan for k in ["mean","std","median","p95","max","min"]}
    return {
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_std": float(np.std(x, ddof=0)),
        f"{prefix}_median": float(np.median(x)),
        f"{prefix}_p95": float(np.percentile(x, 95)),
        f"{prefix}_max": float(np.max(x)),
        f"{prefix}_min": float(np.min(x)),
    }

def _time_above(prefix: str, x: np.ndarray, thresholds: list[float]) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    out = {}
    if x.size == 0:
        for thr in thresholds:
            out[f"{prefix}_pct_gt_{thr}"] = np.nan
        return out
    for thr in thresholds:
        out[f"{prefix}_pct_gt_{thr}"] = float(np.mean(x > thr) * 100.0)
    return out

def _jerk_from_acc(a_xyz: np.ndarray, dt: float) -> np.ndarray:
    """Jerk magnitude from acceleration (finite diff)."""
    if not np.isfinite(dt) or dt <= 0:
        return np.full((a_xyz.shape[0],), np.nan)
    da = np.vstack([np.zeros(3), np.diff(a_xyz, axis=0)])
    j = da / dt
    return _mag(j)

def _quat_angle_between(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Incremental rotation angle between consecutive quaternions.
    q assumed [qw, qx, qy, qz] and roughly normalized.
    angle = 2*acos(|dot(q_t, q_{t-1})|)
    """
    # dot product between consecutive quats
    dot = np.sum(q1 * q2, axis=1)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    return 2.0 * np.arccos(dot)  # radians

def extract_mocap_head_features(
    df_mocap: pd.DataFrame,
    time_col: str = "ASIL",
    t_start: float | None = None,
    t_end: float | None = None,
    speed_thresholds: list[float] = [0.3, 0.5, 1.0],
    angvel_thresholds: list[float] = [30.0, 60.0, 120.0],  # deg/s
) -> dict:
    """
    MoCap head pose features from Head_x/y/z and Head_qw/qx/qy/qz.
    Returns a dict of scalar features for the provided time slice.
    """
    df = df_mocap.copy()
    if t_start is not None:
        df = df[df[time_col] >= t_start]
    if t_end is not None:
        df = df[df[time_col] <= t_end]
    df = df.sort_values(time_col)

    out = {"n_samples": int(len(df))}

    if len(df) < 3:
        # Not enough samples to differentiate reliably
        out.update({k: np.nan for k in [
            "duration_s","path_length","net_displacement","tortuosity",
            "speed_mean","speed_p95","acc_mean","acc_p95","jerk_p95",
            "angvel_p95_deg_s","total_rot_deg"
        ]})
        return out

    t = df[time_col].to_numpy(dtype=float)
    dt = _robust_dt(t)
    out["duration_s"] = float(t[-1] - t[0]) if np.isfinite(t[-1] - t[0]) else np.nan
    out["dt_median_s"] = dt

    # Position
    p = df[["Head_x","Head_y","Head_z"]].to_numpy(dtype=float)

    
    dp = np.diff(p, axis=0)
    step = np.linalg.norm(dp, axis=1)
    out["path_length"] = float(np.nansum(step))
    out["net_displacement"] = float(np.linalg.norm(p[-1] - p[0]))
    out["tortuosity"] = float(out["path_length"] / (out["net_displacement"] + EPS))

    out["pos_range_x"] = float(np.nanmax(p[:,0]) - np.nanmin(p[:,0]))
    out["pos_range_y"] = float(np.nanmax(p[:,1]) - np.nanmin(p[:,1]))
    out["pos_range_z"] = float(np.nanmax(p[:,2]) - np.nanmin(p[:,2]))
    out["pos_std_x"] = float(np.nanstd(p[:,0]))
    out["pos_std_y"] = float(np.nanstd(p[:,1]))
    out["pos_std_z"] = float(np.nanstd(p[:,2]))

    # Velocity & acceleration from position
    if np.isfinite(dt) and dt > 0:
        v = np.vstack([np.zeros(3), np.diff(p, axis=0) / dt])
        a = np.vstack([np.zeros(3), np.diff(v, axis=0) / dt])
        speed = _mag(v)
        accmag = _mag(a)
        jerk = _jerk_from_acc(a, dt)
    else:
        speed = np.full((len(df),), np.nan)
        accmag = np.full((len(df),), np.nan)
        jerk = np.full((len(df),), np.nan)

    out.update(_basic_stats("speed", speed))
    out.update(_basic_stats("acc", accmag))
    out.update(_basic_stats("jerk", jerk))
    out.update(_time_above("speed", speed, speed_thresholds))

    # Rotation features from quaternion
    q = df[["Head_qw","Head_qx","Head_qy","Head_qz"]].to_numpy(dtype=float)
    # normalize (safe)
    q_norm = np.linalg.norm(q, axis=1, keepdims=True)
    q = q / (q_norm + EPS)

    # incremental angle between consecutive quaternions
    ang_rad = _quat_angle_between(q[1:], q[:-1])  # length N-1
    total_rot_deg = float(np.nansum(np.rad2deg(ang_rad)))
    out["total_rot_deg"] = total_rot_deg

    if np.isfinite(dt) and dt > 0:
        angvel_deg_s = np.rad2deg(ang_rad) / dt
        # pad to length N for consistency
        angvel_deg_s = np.concatenate([[0.0], angvel_deg_s])
    else:
        angvel_deg_s = np.full((len(df),), np.nan)

    out.update(_basic_stats("angvel_deg_s", angvel_deg_s))
    out.update(_time_above("angvel_deg_s", angvel_deg_s, angvel_thresholds))

    return out


def extract_hmd_kinematic_features(
    df_hmd: pd.DataFrame,
    time_col: str = "lsl_timestamp",   # you can set to "ASIL" if that's your timeline
    t_start: float | None = None,
    t_end: float | None = None,
    speed_thresholds: list[float] = [0.3, 0.5, 1.0],
    acc_thresholds: list[float] = [1.0, 2.0, 5.0],  # m/s^2 (adjust to your units)
) -> dict:
    """
    HMD features from vx/vy/vz and ax/ay/az (and optional perf signals).
    Returns a dict of scalar features for the provided time slice.
    """
    df = df_hmd.copy()
    if t_start is not None:
        df = df[df[time_col] >= t_start]
    if t_end is not None:
        df = df[df[time_col] <= t_end]
    df = df.sort_values(time_col)

    out = {"n_samples": int(len(df))}
    if len(df) < 3:
        out.update({k: np.nan for k in [
            "duration_s","speed_mean","speed_p95","acc_mean","acc_p95","jerk_p95",
            "framerate_mean","framerate_p5","timewrap_p95"
        ]})
        return out

    t = df[time_col].to_numpy(dtype=float)
    dt = _robust_dt(t)
    out["duration_s"] = float(t[-1] - t[0]) if np.isfinite(t[-1] - t[0]) else np.nan
    out["dt_median_s"] = dt

    v = df[["vx","vy","vz"]].to_numpy(dtype=float)
    a = df[["ax","ay","az"]].to_numpy(dtype=float)

    speed = _mag(v)
    accmag = _mag(a)

    out.update(_basic_stats("speed", speed))
    out.update(_basic_stats("acc", accmag))
    out.update(_time_above("speed", speed, speed_thresholds))
    out.update(_time_above("acc", accmag, acc_thresholds))

    # jerk from acceleration (needs dt)
    jerk = _jerk_from_acc(a, dt) if np.isfinite(dt) and dt > 0 else np.full((len(df),), np.nan)
    out.update(_basic_stats("jerk", jerk))

    # Performance / QoS covariates if present
    if "framerate" in df.columns:
        fr = df["framerate"].to_numpy(dtype=float)
        fr = fr[np.isfinite(fr)]
        out["framerate_mean"] = float(np.mean(fr)) if fr.size else np.nan
        out["framerate_p5"] = float(np.percentile(fr, 5)) if fr.size else np.nan
    else:
        out["framerate_mean"] = np.nan
        out["framerate_p5"] = np.nan

    for col in ["render", "timewrap", "postpresent", "displayfrequency"]:
        if col in df.columns:
            x = df[col].to_numpy(dtype=float)
            out.update(_basic_stats(col, x))
        else:
            # keep it simple: omit if not present
            pass

    # A commonly useful single "bad spikes" indicator:
    if "timewrap" in df.columns:
        out["timewrap_p95"] = _percentile(df["timewrap"].to_numpy(dtype=float), 95)
    else:
        out["timewrap_p95"] = np.nan

    return out
