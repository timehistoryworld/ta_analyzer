import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.linalg import svd
from scipy.ndimage import gaussian_filter1d
import io

# ──────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="TA Data Analyzer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 20px;
        font-weight: 500;
    }
    div[data-testid="stMetric"] {
        background-color: #f0f2f6;
        padding: 12px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Transient Absorption Data Analyzer")

# ──────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────

@st.cache_data
def parse_ta_data(uploaded_file):
    """Parse TA CSV: row0=time delays, col0=wavelengths, rest=ΔOD."""
    content = uploaded_file.read().decode("utf-8")
    uploaded_file.seek(0)
    lines = content.strip().replace("\r\n", "\n").replace("\r", "\n").split("\n")
    
    reader = [line.split(",") for line in lines]
    header = reader[0]
    times = np.array([float(x) for x in header])
    
    wavelengths = []
    data_rows = []
    for row in reader[1:]:
        if len(row) < 2:
            continue
        wavelengths.append(float(row[0]))
        vals = [float(x) for x in row[1:]]
        data_rows.append(vals)
    
    wavelengths = np.array(wavelengths)
    time_delays = times[1:]  # first element is placeholder (0)
    data_matrix = np.array(data_rows)  # shape: (n_wavelengths, n_times)
    
    return wavelengths, time_delays, data_matrix


def background_subtract(wavelengths, time_delays, data, t_range):
    """Subtract average of pre-time-zero data as background."""
    t_min, t_max = t_range
    mask = (time_delays >= t_min) & (time_delays <= t_max)
    if np.sum(mask) == 0:
        return data
    bg = np.nanmean(data[:, mask], axis=1, keepdims=True)
    return data - bg


def detect_chirp(wavelengths, time_delays, data, threshold_frac=0.5):
    """Auto-detect time-zero for each wavelength using derivative maximum."""
    pos_mask = time_delays > 0
    pos_times = time_delays[pos_mask]
    pos_data = data[:, pos_mask]
    
    t0_list = []
    wl_valid = []
    
    for i in range(len(wavelengths)):
        trace = pos_data[i, :]
        # Use absolute value of derivative
        if len(trace) < 5:
            continue
        dt = np.diff(pos_times)
        dy = np.diff(trace)
        deriv = np.abs(dy / dt)
        
        # Smooth derivative
        if len(deriv) > 7:
            deriv_smooth = savgol_filter(deriv, min(7, len(deriv) - (1 if len(deriv) % 2 == 0 else 0)), 2)
        else:
            deriv_smooth = deriv
        
        max_deriv = np.max(deriv_smooth)
        overall_signal = np.max(np.abs(trace)) - np.min(np.abs(trace[:min(5, len(trace))]))
        
        if overall_signal > 0.002:  # signal threshold
            idx_max = np.argmax(deriv_smooth)
            t0_est = (pos_times[idx_max] + pos_times[idx_max + 1]) / 2
            t0_list.append(t0_est)
            wl_valid.append(wavelengths[i])
    
    return np.array(wl_valid), np.array(t0_list)


def fit_chirp_polynomial(wl_valid, t0_list, poly_order=3):
    """Fit polynomial to chirp curve t0(λ)."""
    if len(wl_valid) < poly_order + 1:
        return None
    coeffs = np.polyfit(wl_valid, t0_list, poly_order)
    return coeffs


def apply_chirp_correction(wavelengths, time_delays, data, coeffs):
    """Shift each wavelength's kinetic trace by chirp offset."""
    from scipy.interpolate import interp1d
    
    t0_ref = np.polyval(coeffs, np.median(wavelengths))
    corrected = np.copy(data)
    
    for i, wl in enumerate(wavelengths):
        t0_wl = np.polyval(coeffs, wl)
        shift = t0_wl - t0_ref
        
        # Interpolate shifted trace
        shifted_times = time_delays - shift
        valid = np.isfinite(data[i, :])
        if np.sum(valid) < 3:
            continue
        
        try:
            f = interp1d(shifted_times[valid], data[i, valid],
                         kind='linear', bounds_error=False, fill_value=0.0)
            corrected[i, :] = f(time_delays)
        except Exception:
            pass
    
    return corrected


def smooth_data(data, window=5, poly_order=2, axis=0):
    """Apply Savitzky-Golay smoothing along specified axis."""
    if window < poly_order + 2:
        window = poly_order + 2
    if window % 2 == 0:
        window += 1
    
    smoothed = np.copy(data)
    if axis == 0:  # smooth along wavelength
        for j in range(data.shape[1]):
            try:
                smoothed[:, j] = savgol_filter(data[:, j], window, poly_order)
            except Exception:
                pass
    else:  # smooth along time
        for i in range(data.shape[0]):
            try:
                smoothed[i, :] = savgol_filter(data[i, :], window, poly_order)
            except Exception:
                pass
    return smoothed


def replace_zeros_with_nan(data):
    """Replace exact zeros with NaN for cleaner visualization."""
    result = np.copy(data).astype(float)
    # Only replace zeros in columns where ALL values are zero (missing data)
    for j in range(data.shape[1]):
        if np.all(data[:, j] == 0):
            result[:, j] = np.nan
    return result


# ── Exponential Models ──

def mono_exp(t, A1, tau1, y0):
    return y0 + A1 * np.exp(-t / tau1)

def bi_exp(t, A1, tau1, A2, tau2, y0):
    return y0 + A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2)

def tri_exp(t, A1, tau1, A2, tau2, A3, tau3, y0):
    return y0 + A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2) + A3 * np.exp(-t / tau3)


def fit_kinetic_trace(t, y, model_type="bi", t_start=0.0):
    """Fit kinetic trace with exponential decay model."""
    mask = (t >= t_start) & np.isfinite(y)
    t_fit = t[mask]
    y_fit = y[mask]
    
    if len(t_fit) < 5:
        return None, None, None
    
    y0_guess = y_fit[-1]
    A_total = y_fit[0] - y0_guess
    t_range = t_fit[-1] - t_fit[0]
    
    try:
        if model_type == "mono":
            p0 = [A_total, t_range / 5, y0_guess]
            bounds = ([-np.inf, 0.01, -np.inf], [np.inf, t_range * 10, np.inf])
            popt, pcov = curve_fit(mono_exp, t_fit, y_fit, p0=p0, bounds=bounds, maxfev=10000)
            y_calc = mono_exp(t_fit, *popt)
            param_names = ["A₁", "τ₁ (ps)", "y₀"]
            
        elif model_type == "bi":
            p0 = [A_total * 0.6, t_range / 20, A_total * 0.4, t_range / 2, y0_guess]
            bounds = ([-np.inf, 0.01, -np.inf, 0.1, -np.inf],
                      [np.inf, t_range * 5, np.inf, t_range * 10, np.inf])
            popt, pcov = curve_fit(bi_exp, t_fit, y_fit, p0=p0, bounds=bounds, maxfev=10000)
            y_calc = bi_exp(t_fit, *popt)
            param_names = ["A₁", "τ₁ (ps)", "A₂", "τ₂ (ps)", "y₀"]
            
        elif model_type == "tri":
            p0 = [A_total * 0.4, t_range / 50, A_total * 0.35, t_range / 5,
                  A_total * 0.25, t_range / 1, y0_guess]
            bounds = ([-np.inf, 0.01, -np.inf, 0.1, -np.inf, 1.0, -np.inf],
                      [np.inf, t_range * 2, np.inf, t_range * 5, np.inf, t_range * 10, np.inf])
            popt, pcov = curve_fit(tri_exp, t_fit, y_fit, p0=p0, bounds=bounds, maxfev=10000)
            y_calc = tri_exp(t_fit, *popt)
            param_names = ["A₁", "τ₁ (ps)", "A₂", "τ₂ (ps)", "A₃", "τ₃ (ps)", "y₀"]
        else:
            return None, None, None
        
        # R² calculation
        ss_res = np.sum((y_fit - y_calc) ** 2)
        ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        result = {
            "params": dict(zip(param_names, popt)),
            "r_squared": r_squared,
            "t_fit": t_fit,
            "y_fit": y_fit,
            "y_calc": y_calc,
            "residuals": y_fit - y_calc,
        }
        return result, popt, param_names
        
    except Exception as e:
        return None, None, str(e)


# ──────────────────────────────────────────────
# Sidebar: Data Upload
# ──────────────────────────────────────────────

with st.sidebar:
    st.header("📂 Data Upload")
    uploaded_file = st.file_uploader(
        "Upload TA data (CSV)",
        type=["csv"],
        help="Format: Row 0 = time delays, Col 0 = wavelengths, Matrix = ΔOD"
    )
    
    if uploaded_file:
        wavelengths, time_delays, raw_data = parse_ta_data(uploaded_file)
        
        st.success(f"✅ Data loaded: {uploaded_file.name}")
        st.caption(f"**Wavelengths:** {wavelengths[0]:.1f} – {wavelengths[-1]:.1f} nm ({len(wavelengths)} pts)")
        st.caption(f"**Time delays:** {time_delays[0]:.3f} – {time_delays[-1]:.1f} ps ({len(time_delays)} pts)")
        st.caption(f"**ΔOD range:** {np.nanmin(raw_data):.4f} – {np.nanmax(raw_data):.4f}")
        
        # Store in session state
        if "processed_data" not in st.session_state:
            st.session_state.processed_data = raw_data.copy()
        if "processing_log" not in st.session_state:
            st.session_state.processing_log = []

if not uploaded_file:
    st.info("👈 **Upload a CSV file** in the sidebar to begin analysis.")
    st.markdown("""
    ### Expected CSV Format
    ```
    0,       t₁,     t₂,     t₃,    ...
    λ₁,   ΔOD₁₁,  ΔOD₁₂, ΔOD₁₃,  ...
    λ₂,   ΔOD₂₁,  ΔOD₂₂, ΔOD₂₃,  ...
    ...
    ```
    - **Row 0:** Time delays (ps)
    - **Column 0:** Wavelengths (nm)
    - **Data matrix:** ΔOD values
    
    ### Features
    1. **Preprocessing** — Background subtraction, chirp correction, smoothing
    2. **Visualization** — 2D heatmap, spectral slices, kinetic traces
    3. **Kinetic Fitting** — Mono/bi/tri-exponential decay fitting
    4. **SVD Analysis** — Singular value decomposition & component analysis
    """)
    st.stop()

# ──────────────────────────────────────────────
# Main Tabs
# ──────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "🔧 Preprocessing",
    "📊 Visualization",
    "📈 Kinetic Fitting",
    "🧩 SVD Analysis",
])

# ══════════════════════════════════════════════
# TAB 1: PREPROCESSING
# ══════════════════════════════════════════════
with tab1:
    st.header("Preprocessing Pipeline")
    
    working_data = raw_data.copy()
    log = []
    
    col_a, col_b = st.columns([1, 1])
    
    with col_a:
        st.subheader("1️⃣ Background Subtraction")
        do_bg = st.checkbox("Enable background subtraction", value=True)
        
        neg_times = time_delays[time_delays < 0]
        if len(neg_times) > 0 and do_bg:
            bg_range = st.slider(
                "Background time range (ps)",
                float(neg_times[0]), 0.0,
                (float(neg_times[0]), -0.05),
                step=0.01,
                help="Average ΔOD in this range will be subtracted"
            )
            working_data = background_subtract(wavelengths, time_delays, working_data, bg_range)
            log.append(f"BG subtraction: [{bg_range[0]:.3f}, {bg_range[1]:.3f}] ps")
        elif do_bg:
            st.warning("No negative time delay data available for background subtraction.")
    
    with col_b:
        st.subheader("2️⃣ Smoothing")
        do_smooth = st.checkbox("Enable smoothing (Savitzky-Golay)", value=False)
        
        if do_smooth:
            smooth_axis = st.radio("Smooth along:", ["Wavelength", "Time", "Both"], horizontal=True)
            smooth_window = st.slider("Window size", 3, 21, 5, step=2)
            smooth_poly = st.slider("Polynomial order", 1, 5, 2)
            
            if smooth_axis in ["Wavelength", "Both"]:
                working_data = smooth_data(working_data, smooth_window, smooth_poly, axis=0)
                log.append(f"Smoothed (λ): window={smooth_window}, poly={smooth_poly}")
            if smooth_axis in ["Time", "Both"]:
                working_data = smooth_data(working_data, smooth_window, smooth_poly, axis=1)
                log.append(f"Smoothed (t): window={smooth_window}, poly={smooth_poly}")
    
    st.divider()
    st.subheader("3️⃣ Chirp (GVD) Correction")
    do_chirp = st.checkbox("Enable chirp correction", value=False)
    
    if do_chirp:
        col_c1, col_c2 = st.columns([1, 1])
        
        with col_c1:
            poly_order = st.slider("Polynomial order for chirp fit", 2, 6, 3)
            
            wl_valid, t0_detected = detect_chirp(wavelengths, time_delays, working_data)
            
            if len(wl_valid) > poly_order + 1:
                # Remove outliers (IQR method)
                q1, q3 = np.percentile(t0_detected, [25, 75])
                iqr = q3 - q1
                inlier_mask = (t0_detected >= q1 - 1.5 * iqr) & (t0_detected <= q3 + 1.5 * iqr)
                wl_clean = wl_valid[inlier_mask]
                t0_clean = t0_detected[inlier_mask]
                
                coeffs = fit_chirp_polynomial(wl_clean, t0_clean, poly_order)
                
                if coeffs is not None:
                    # Plot chirp curve
                    wl_plot = np.linspace(wavelengths[0], wavelengths[-1], 200)
                    t0_fit = np.polyval(coeffs, wl_plot)
                    
                    fig_chirp = go.Figure()
                    fig_chirp.add_trace(go.Scatter(
                        x=wl_valid, y=t0_detected,
                        mode="markers", name="Detected t₀",
                        marker=dict(size=3, color="gray", opacity=0.4),
                    ))
                    fig_chirp.add_trace(go.Scatter(
                        x=wl_clean, y=t0_clean,
                        mode="markers", name="Inliers",
                        marker=dict(size=5, color="#2196F3"),
                    ))
                    fig_chirp.add_trace(go.Scatter(
                        x=wl_plot, y=t0_fit,
                        mode="lines", name=f"Poly fit (n={poly_order})",
                        line=dict(color="#FF5722", width=2),
                    ))
                    fig_chirp.update_layout(
                        title="Chirp Detection: t₀(λ)",
                        xaxis_title="Wavelength (nm)",
                        yaxis_title="t₀ (ps)",
                        height=350,
                        margin=dict(t=40, b=40),
                    )
                    st.plotly_chart(fig_chirp, use_container_width=True)
                    
                    working_data = apply_chirp_correction(wavelengths, time_delays, working_data, coeffs)
                    log.append(f"Chirp correction: poly_order={poly_order}")
                else:
                    st.warning("Chirp polynomial fit failed.")
            else:
                st.warning(f"Not enough valid points detected ({len(wl_valid)}) for chirp correction.")
        
        with col_c2:
            if len(wl_valid) > 0 and 'coeffs' in dir() and coeffs is not None:
                st.markdown("**Chirp fit coefficients:**")
                for i, c in enumerate(coeffs):
                    st.code(f"c{len(coeffs)-1-i} = {c:.6e}")
                st.caption(f"t₀(λ) = Σ cₙ·λⁿ")
    
    st.divider()
    
    # Apply and store
    if st.button("✅ Apply Preprocessing", type="primary", use_container_width=True):
        st.session_state.processed_data = working_data.copy()
        st.session_state.processing_log = log
        st.success("Preprocessing applied! Proceed to Visualization tab.")
    
    if st.session_state.processing_log:
        st.caption("**Applied steps:** " + " → ".join(st.session_state.processing_log))
    
    # Quick preview: before vs after
    st.subheader("Preview: Raw vs Processed")
    preview_wl = st.select_slider(
        "Preview wavelength (nm)",
        options=[f"{w:.1f}" for w in wavelengths],
        value=f"{wavelengths[len(wavelengths)//2]:.1f}",
    )
    preview_idx = np.argmin(np.abs(wavelengths - float(preview_wl)))
    
    fig_preview = go.Figure()
    fig_preview.add_trace(go.Scatter(
        x=time_delays, y=raw_data[preview_idx, :],
        mode="lines", name="Raw",
        line=dict(color="gray", width=1),
    ))
    fig_preview.add_trace(go.Scatter(
        x=time_delays, y=working_data[preview_idx, :],
        mode="lines", name="Processed",
        line=dict(color="#2196F3", width=2),
    ))
    fig_preview.update_layout(
        xaxis_title="Time Delay (ps)",
        yaxis_title="ΔOD",
        xaxis_type="log",
        xaxis=dict(range=[np.log10(0.05), np.log10(time_delays[-1])]),
        height=300,
        margin=dict(t=20, b=40),
        legend=dict(x=0.02, y=0.98),
    )
    fig_preview.update_xaxes(
        tickvals=[0.1, 1, 10, 100, 1000],
        ticktext=["0.1", "1", "10", "100", "1000"],
    )
    st.plotly_chart(fig_preview, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 2: VISUALIZATION
# ══════════════════════════════════════════════
with tab2:
    st.header("Data Visualization")
    
    data = st.session_state.processed_data
    
    # ── Wavelength & Time Range Selection ──
    with st.expander("🔍 Display Range", expanded=False):
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            wl_range = st.slider(
                "Wavelength range (nm)",
                float(wavelengths[0]), float(wavelengths[-1]),
                (float(wavelengths[0]), float(wavelengths[-1])),
            )
        with col_r2:
            pos_times = time_delays[time_delays > 0]
            t_min_log = float(np.log10(max(pos_times[0], 0.01)))
            t_max_log = float(np.log10(pos_times[-1]))
            t_log_range = st.slider(
                "Time range (log₁₀ ps)",
                t_min_log, t_max_log,
                (t_min_log, t_max_log),
                step=0.1,
            )
    
    wl_mask = (wavelengths >= wl_range[0]) & (wavelengths <= wl_range[1])
    t_mask = (time_delays > 0) & (time_delays >= 10**t_log_range[0]) & (time_delays <= 10**t_log_range[1])
    
    wl_sub = wavelengths[wl_mask]
    t_sub = time_delays[t_mask]
    data_sub = data[np.ix_(wl_mask, t_mask)]
    
    # ── 2D Heatmap ──
    st.subheader("2D ΔOD Heatmap")
    
    col_h1, col_h2 = st.columns([3, 1])
    with col_h2:
        colorscale = st.selectbox("Colorscale", ["RdBu_r", "Spectral_r", "Viridis", "Plasma", "Turbo", "PiYG_r"], index=0)
        symmetric = st.checkbox("Symmetric color range", value=True)
        
        vmax = np.nanmax(np.abs(data_sub))
        if symmetric:
            clim_val = st.slider("Color limit (±ΔOD)", 0.001, float(vmax), float(vmax * 0.8), step=0.001, format="%.3f")
            zmin, zmax = -clim_val, clim_val
        else:
            zmin = float(np.nanmin(data_sub))
            zmax = float(np.nanmax(data_sub))
    
    with col_h1:
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=data_sub,
            x=np.log10(t_sub),
            y=wl_sub,
            colorscale=colorscale,
            zmin=zmin, zmax=zmax,
            colorbar=dict(title="ΔOD", titleside="right"),
            hovertemplate="λ: %{y:.1f} nm<br>t: %{customdata:.2f} ps<br>ΔOD: %{z:.5f}<extra></extra>",
            customdata=np.tile(t_sub, (len(wl_sub), 1)),
        ))
        fig_heatmap.update_layout(
            xaxis_title="Time Delay (ps)",
            yaxis_title="Wavelength (nm)",
            height=500,
            margin=dict(t=20, b=50, l=60, r=20),
            xaxis=dict(
                tickvals=np.log10([0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5000]),
                ticktext=["0.1", "0.5", "1", "5", "10", "50", "100", "500", "1k", "5k"],
            ),
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    st.divider()
    
    # ── Spectral Slices ──
    col_s1, col_s2 = st.columns([1, 1])
    
    with col_s1:
        st.subheader("Spectral Slices (ΔOD vs λ)")
        
        # Default time delays for slices
        default_times = [0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
        available_times = time_delays[time_delays > 0]
        
        time_options = st.multiselect(
            "Select time delays (ps)",
            options=[f"{t:.2f}" for t in available_times],
            default=[f"{available_times[np.argmin(np.abs(available_times - dt))]:.2f}" for dt in default_times if dt <= available_times[-1]],
            max_selections=12,
        )
        
        normalize_spectra = st.checkbox("Normalize spectra", value=False, key="norm_spec")
        
        if time_options:
            fig_spectra = go.Figure()
            colors = [
                "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
                "#bcbd22", "#17becf", "#ff6347", "#4682b4",
            ]
            
            for k, t_str in enumerate(time_options):
                t_val = float(t_str)
                t_idx = np.argmin(np.abs(time_delays - t_val))
                spectrum = data[wl_mask, t_idx]
                
                if normalize_spectra and np.max(np.abs(spectrum)) > 0:
                    spectrum = spectrum / np.max(np.abs(spectrum))
                
                fig_spectra.add_trace(go.Scatter(
                    x=wl_sub, y=spectrum,
                    mode="lines",
                    name=f"{time_delays[t_idx]:.2f} ps",
                    line=dict(color=colors[k % len(colors)], width=1.5),
                ))
            
            fig_spectra.add_hline(y=0, line_dash="dash", line_color="gray", line_width=0.5)
            fig_spectra.update_layout(
                xaxis_title="Wavelength (nm)",
                yaxis_title="Normalized ΔOD" if normalize_spectra else "ΔOD",
                height=420,
                margin=dict(t=20, b=50),
                legend=dict(font=dict(size=10)),
            )
            st.plotly_chart(fig_spectra, use_container_width=True)
    
    # ── Kinetic Traces ──
    with col_s2:
        st.subheader("Kinetic Traces (ΔOD vs t)")
        
        default_wls = [550.0, 610.0, 660.0, 700.0, 750.0]
        wl_options = st.multiselect(
            "Select wavelengths (nm)",
            options=[f"{w:.1f}" for w in wavelengths],
            default=[f"{wavelengths[np.argmin(np.abs(wavelengths - dw))]:.1f}" for dw in default_wls if wavelengths[0] <= dw <= wavelengths[-1]],
            max_selections=10,
        )
        
        time_scale = st.radio("Time axis", ["Log", "Linear"], horizontal=True, key="t_scale_viz")
        normalize_kinetics = st.checkbox("Normalize kinetics", value=False, key="norm_kin")
        
        if wl_options:
            fig_kinetics = go.Figure()
            
            for k, wl_str in enumerate(wl_options):
                wl_val = float(wl_str)
                wl_idx = np.argmin(np.abs(wavelengths - wl_val))
                trace = data[wl_idx, t_mask]
                
                if normalize_kinetics and np.max(np.abs(trace)) > 0:
                    trace = trace / np.max(np.abs(trace))
                
                fig_kinetics.add_trace(go.Scatter(
                    x=t_sub, y=trace,
                    mode="lines",
                    name=f"{wavelengths[wl_idx]:.1f} nm",
                    line=dict(color=colors[k % len(colors)], width=1.5),
                ))
            
            fig_kinetics.add_hline(y=0, line_dash="dash", line_color="gray", line_width=0.5)
            
            layout_kwargs = dict(
                xaxis_title="Time Delay (ps)",
                yaxis_title="Normalized ΔOD" if normalize_kinetics else "ΔOD",
                height=420,
                margin=dict(t=20, b=50),
                legend=dict(font=dict(size=10)),
            )
            if time_scale == "Log":
                layout_kwargs["xaxis_type"] = "log"
                layout_kwargs["xaxis"] = dict(
                    tickvals=[0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5000],
                    ticktext=["0.1", "0.5", "1", "5", "10", "50", "100", "500", "1k", "5k"],
                )
            
            fig_kinetics.update_layout(**layout_kwargs)
            st.plotly_chart(fig_kinetics, use_container_width=True)
    
    # ── Export ──
    st.divider()
    with st.expander("📥 Export Processed Data"):
        export_df = pd.DataFrame(
            data,
            index=pd.Index(wavelengths, name="Wavelength (nm)"),
            columns=[f"{t:.4f}" for t in time_delays],
        )
        
        buffer = io.BytesIO()
        export_df.to_excel(buffer, sheet_name="TA Data")
        buffer.seek(0)
        st.download_button(
            "Download as Excel",
            data=buffer,
            file_name="TA_processed.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


# ══════════════════════════════════════════════
# TAB 3: KINETIC FITTING
# ══════════════════════════════════════════════
with tab3:
    st.header("Kinetic Fitting")
    
    data = st.session_state.processed_data
    
    col_f1, col_f2 = st.columns([1, 2])
    
    with col_f1:
        st.subheader("Fitting Parameters")
        
        fit_wl_str = st.selectbox(
            "Wavelength (nm)",
            options=[f"{w:.1f}" for w in wavelengths],
            index=int(np.argmin(np.abs(wavelengths - 660))),
        )
        fit_wl_idx = np.argmin(np.abs(wavelengths - float(fit_wl_str)))
        
        model_type = st.radio(
            "Decay model",
            ["mono", "bi", "tri"],
            format_func=lambda x: {"mono": "Mono-exponential", "bi": "Bi-exponential", "tri": "Tri-exponential"}[x],
            index=1,
        )
        
        t_start = st.number_input("Fit start time (ps)", value=0.3, min_value=0.0, step=0.1,
                                   help="Exclude early-time coherent artifacts")
        
        run_fit = st.button("🚀 Run Fit", type="primary", use_container_width=True)
        
        # Batch fitting
        st.divider()
        st.subheader("Batch Fitting")
        batch_wls_str = st.multiselect(
            "Select wavelengths for batch fit",
            options=[f"{w:.1f}" for w in wavelengths],
            default=[],
            help="Fit the same model to multiple wavelengths"
        )
        run_batch = st.button("🔄 Run Batch Fit", use_container_width=True)
    
    with col_f2:
        if run_fit:
            trace = data[fit_wl_idx, :]
            result, popt, param_names = fit_kinetic_trace(
                time_delays, trace, model_type=model_type, t_start=t_start
            )
            
            if result is not None:
                # Results table
                st.subheader(f"Results @ {wavelengths[fit_wl_idx]:.1f} nm")
                
                res_cols = st.columns(len(result["params"]) + 1)
                for i, (name, val) in enumerate(result["params"].items()):
                    with res_cols[i]:
                        if "τ" in name:
                            st.metric(name, f"{val:.2f}")
                        elif name == "y₀":
                            st.metric(name, f"{val:.5f}")
                        else:
                            st.metric(name, f"{val:.4f}")
                with res_cols[-1]:
                    st.metric("R²", f"{result['r_squared']:.6f}")
                
                # Plot fit
                fig_fit = make_subplots(
                    rows=2, cols=1, shared_xaxes=True,
                    row_heights=[0.75, 0.25],
                    vertical_spacing=0.05,
                )
                
                # Data
                pos_mask_plot = time_delays > 0
                fig_fit.add_trace(go.Scatter(
                    x=time_delays[pos_mask_plot], y=trace[pos_mask_plot],
                    mode="markers", name="Data",
                    marker=dict(size=3, color="gray", opacity=0.5),
                ), row=1, col=1)
                
                # Fit
                fig_fit.add_trace(go.Scatter(
                    x=result["t_fit"], y=result["y_calc"],
                    mode="lines", name="Fit",
                    line=dict(color="#FF5722", width=2),
                ), row=1, col=1)
                
                # Residuals
                fig_fit.add_trace(go.Scatter(
                    x=result["t_fit"], y=result["residuals"],
                    mode="markers", name="Residuals",
                    marker=dict(size=2, color="#2196F3"),
                ), row=2, col=1)
                fig_fit.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
                
                fig_fit.update_xaxes(type="log", row=1, col=1)
                fig_fit.update_xaxes(
                    type="log", title_text="Time Delay (ps)", row=2, col=1,
                    tickvals=[0.1, 1, 10, 100, 1000],
                    ticktext=["0.1", "1", "10", "100", "1k"],
                )
                fig_fit.update_yaxes(title_text="ΔOD", row=1, col=1)
                fig_fit.update_yaxes(title_text="Residual", row=2, col=1)
                fig_fit.update_layout(height=500, margin=dict(t=20, b=50))
                
                st.plotly_chart(fig_fit, use_container_width=True)
                
                # Component breakdown for multi-exponential
                if model_type in ["bi", "tri"]:
                    st.subheader("Component Breakdown")
                    fig_comp = go.Figure()
                    t_plot = result["t_fit"]
                    
                    p = result["params"]
                    y0 = p["y₀"]
                    
                    comp_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
                    
                    if model_type == "bi":
                        components = [
                            (p["A₁"], p["τ₁ (ps)"], "τ₁"),
                            (p["A₂"], p["τ₂ (ps)"], "τ₂"),
                        ]
                    else:
                        components = [
                            (p["A₁"], p["τ₁ (ps)"], "τ₁"),
                            (p["A₂"], p["τ₂ (ps)"], "τ₂"),
                            (p["A₃"], p["τ₃ (ps)"], "τ₃"),
                        ]
                    
                    for k, (A, tau, label) in enumerate(components):
                        comp = A * np.exp(-t_plot / tau)
                        fig_comp.add_trace(go.Scatter(
                            x=t_plot, y=comp,
                            mode="lines",
                            name=f"{label} = {tau:.2f} ps (A={A:.4f})",
                            line=dict(color=comp_colors[k], width=1.5, dash="dash"),
                        ))
                    
                    fig_comp.add_hline(y=0, line_dash="dot", line_color="gray")
                    fig_comp.update_layout(
                        xaxis_title="Time Delay (ps)",
                        yaxis_title="ΔOD (component)",
                        xaxis_type="log",
                        height=300,
                        margin=dict(t=20, b=50),
                    )
                    st.plotly_chart(fig_comp, use_container_width=True)
            else:
                st.error(f"Fitting failed. {param_names if isinstance(param_names, str) else ''}")
        
        # Batch fitting results
        if run_batch and batch_wls_str:
            st.subheader("Batch Fitting Results")
            
            batch_results = []
            for wl_str in batch_wls_str:
                wl_val = float(wl_str)
                wl_idx = np.argmin(np.abs(wavelengths - wl_val))
                trace = data[wl_idx, :]
                result, popt, pnames = fit_kinetic_trace(
                    time_delays, trace, model_type=model_type, t_start=t_start
                )
                if result is not None:
                    row_data = {"λ (nm)": f"{wavelengths[wl_idx]:.1f}"}
                    row_data.update(result["params"])
                    row_data["R²"] = result["r_squared"]
                    batch_results.append(row_data)
            
            if batch_results:
                batch_df = pd.DataFrame(batch_results)
                st.dataframe(batch_df, use_container_width=True, hide_index=True)
                
                # Plot τ values vs wavelength
                fig_tau = go.Figure()
                tau_cols = [c for c in batch_df.columns if "τ" in c]
                for tc in tau_cols:
                    fig_tau.add_trace(go.Scatter(
                        x=batch_df["λ (nm)"].astype(float),
                        y=batch_df[tc],
                        mode="lines+markers",
                        name=tc,
                        line=dict(width=2),
                    ))
                fig_tau.update_layout(
                    xaxis_title="Wavelength (nm)",
                    yaxis_title="τ (ps)",
                    yaxis_type="log",
                    height=350,
                    margin=dict(t=20, b=50),
                )
                st.plotly_chart(fig_tau, use_container_width=True)
                
                # Download
                buf = io.BytesIO()
                batch_df.to_excel(buf, index=False, sheet_name="Batch Fit")
                buf.seek(0)
                st.download_button("📥 Download batch results", buf,
                                   file_name="TA_batch_fit.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# ══════════════════════════════════════════════
# TAB 4: SVD ANALYSIS
# ══════════════════════════════════════════════
with tab4:
    st.header("SVD / Global Analysis")
    
    data = st.session_state.processed_data
    
    # Use only positive time data for SVD
    pos_mask = time_delays > 0
    data_pos = data[:, pos_mask]
    t_pos = time_delays[pos_mask]
    
    # Handle NaN/zero columns
    valid_cols = ~np.all(data_pos == 0, axis=0) & ~np.any(np.isnan(data_pos), axis=0)
    data_svd = data_pos[:, valid_cols]
    t_svd = t_pos[valid_cols]
    
    st.subheader("1️⃣ Singular Value Decomposition")
    
    if st.button("🧮 Run SVD", type="primary"):
        U, s, Vt = svd(data_svd, full_matrices=False)
        
        st.session_state.svd_U = U
        st.session_state.svd_s = s
        st.session_state.svd_Vt = Vt
        st.session_state.svd_t = t_svd
        st.success("SVD complete!")
    
    if "svd_s" in st.session_state:
        s = st.session_state.svd_s
        U = st.session_state.svd_U
        Vt = st.session_state.svd_Vt
        t_svd = st.session_state.svd_t
        
        col_sv1, col_sv2 = st.columns([1, 1])
        
        with col_sv1:
            # Scree plot
            n_show = min(20, len(s))
            fig_scree = go.Figure()
            fig_scree.add_trace(go.Scatter(
                x=list(range(1, n_show + 1)),
                y=s[:n_show],
                mode="lines+markers",
                marker=dict(size=8, color="#2196F3"),
                line=dict(width=2),
            ))
            fig_scree.update_layout(
                title="Scree Plot (Singular Values)",
                xaxis_title="Component",
                yaxis_title="Singular Value",
                yaxis_type="log",
                height=350,
                margin=dict(t=40, b=50),
                xaxis=dict(dtick=1),
            )
            st.plotly_chart(fig_scree, use_container_width=True)
            
            # Variance explained
            variance = s**2
            var_ratio = variance / np.sum(variance) * 100
            cumvar = np.cumsum(var_ratio)
            
            st.markdown("**Variance explained:**")
            var_df = pd.DataFrame({
                "Component": range(1, min(8, len(s)) + 1),
                "Singular Value": [f"{s[i]:.4f}" for i in range(min(8, len(s)))],
                "Variance (%)": [f"{var_ratio[i]:.2f}" for i in range(min(8, len(s)))],
                "Cumulative (%)": [f"{cumvar[i]:.2f}" for i in range(min(8, len(s)))],
            })
            st.dataframe(var_df, use_container_width=True, hide_index=True)
        
        with col_sv2:
            n_components = st.slider("Number of components to display", 1, min(8, len(s)), 3)
            
            # Component spectra (left singular vectors × singular values)
            fig_uspec = go.Figure()
            comp_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                           "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
            
            for k in range(n_components):
                spectrum = U[:, k] * s[k]
                fig_uspec.add_trace(go.Scatter(
                    x=wavelengths, y=spectrum,
                    mode="lines",
                    name=f"Component {k+1}",
                    line=dict(color=comp_colors[k % len(comp_colors)], width=1.5),
                ))
            fig_uspec.add_hline(y=0, line_dash="dash", line_color="gray", line_width=0.5)
            fig_uspec.update_layout(
                title="Component Spectra (U × S)",
                xaxis_title="Wavelength (nm)",
                yaxis_title="Amplitude",
                height=300,
                margin=dict(t=40, b=50),
            )
            st.plotly_chart(fig_uspec, use_container_width=True)
            
            # Component kinetics (right singular vectors)
            fig_vkin = go.Figure()
            for k in range(n_components):
                kinetic = Vt[k, :]
                fig_vkin.add_trace(go.Scatter(
                    x=t_svd, y=kinetic,
                    mode="lines",
                    name=f"Component {k+1}",
                    line=dict(color=comp_colors[k % len(comp_colors)], width=1.5),
                ))
            fig_vkin.add_hline(y=0, line_dash="dash", line_color="gray", line_width=0.5)
            fig_vkin.update_layout(
                title="Component Kinetics (Vᵀ)",
                xaxis_title="Time Delay (ps)",
                yaxis_title="Amplitude",
                xaxis_type="log",
                height=300,
                margin=dict(t=40, b=50),
                xaxis=dict(
                    tickvals=[0.1, 1, 10, 100, 1000],
                    ticktext=["0.1", "1", "10", "100", "1k"],
                ),
            )
            st.plotly_chart(fig_vkin, use_container_width=True)
        
        st.divider()
        st.subheader("2️⃣ Data Reconstruction")
        
        n_recon = st.slider("Reconstruct with N components", 1, min(10, len(s)), 3, key="n_recon")
        
        # Reconstruct
        recon = U[:, :n_recon] @ np.diag(s[:n_recon]) @ Vt[:n_recon, :]
        residual = data_svd - recon
        
        col_rc1, col_rc2 = st.columns(2)
        
        with col_rc1:
            fig_recon = go.Figure(data=go.Heatmap(
                z=recon,
                x=np.log10(t_svd),
                y=wavelengths,
                colorscale="RdBu_r",
                zmin=-np.nanmax(np.abs(data_svd)) * 0.8,
                zmax=np.nanmax(np.abs(data_svd)) * 0.8,
                colorbar=dict(title="ΔOD"),
            ))
            fig_recon.update_layout(
                title=f"Reconstructed (N={n_recon})",
                xaxis_title="Time Delay (ps)",
                yaxis_title="Wavelength (nm)",
                height=380,
                margin=dict(t=40, b=50),
                xaxis=dict(
                    tickvals=np.log10([0.1, 1, 10, 100, 1000]),
                    ticktext=["0.1", "1", "10", "100", "1k"],
                ),
            )
            st.plotly_chart(fig_recon, use_container_width=True)
        
        with col_rc2:
            fig_resid = go.Figure(data=go.Heatmap(
                z=residual,
                x=np.log10(t_svd),
                y=wavelengths,
                colorscale="RdBu_r",
                zmin=-np.nanmax(np.abs(residual)),
                zmax=np.nanmax(np.abs(residual)),
                colorbar=dict(title="ΔOD"),
            ))
            fig_resid.update_layout(
                title=f"Residual (RMSE: {np.sqrt(np.mean(residual**2)):.6f})",
                xaxis_title="Time Delay (ps)",
                yaxis_title="Wavelength (nm)",
                height=380,
                margin=dict(t=40, b=50),
                xaxis=dict(
                    tickvals=np.log10([0.1, 1, 10, 100, 1000]),
                    ticktext=["0.1", "1", "10", "100", "1k"],
                ),
            )
            st.plotly_chart(fig_resid, use_container_width=True)


# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.divider()
st.caption("**TA Data Analyzer** v1.0 | Built with Streamlit & Plotly")
