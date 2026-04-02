import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit, least_squares
from scipy.signal import savgol_filter
from scipy.linalg import svd
from scipy.interpolate import interp1d
import io

st.set_page_config(page_title="TA Data Analyzer", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>
.block-container{padding-top:1.5rem}
.stTabs [data-baseweb="tab-list"]{gap:8px}
.stTabs [data-baseweb="tab"]{padding:8px 20px;font-weight:500}
div[data-testid="stMetric"]{background-color:#f0f2f6;padding:12px;border-radius:8px}
</style>""", unsafe_allow_html=True)
st.title("⚡ Transient Absorption Data Analyzer")

# ── Helpers ──
@st.cache_data
def parse_ta_data(uf):
    c = uf.read().decode("utf-8"); uf.seek(0)
    lines = c.strip().replace("\r\n","\n").replace("\r","\n").split("\n")
    r = [l.split(",") for l in lines]
    times = np.array([float(x) for x in r[0]])
    wl, dr = [], []
    for row in r[1:]:
        if len(row)<2: continue
        wl.append(float(row[0])); dr.append([float(x) for x in row[1:]])
    return np.array(wl), times[1:], np.array(dr)

def bg_sub(wl, td, d, tr):
    m = (td>=tr[0])&(td<=tr[1])
    if np.sum(m)==0: return d
    return d - np.nanmean(d[:,m], axis=1, keepdims=True)

def fit_chirp_poly(wp, tp, n=3):
    if len(wp)<n+1: return None
    return np.polyfit(wp, tp, n)

def apply_chirp(wl, td, d, co):
    t0r = np.polyval(co, np.median(wl)); out = np.copy(d)
    for i,w in enumerate(wl):
        sh = np.polyval(co,w)-t0r; st2 = td-sh
        v = np.isfinite(d[i,:])
        if np.sum(v)<3: continue
        try:
            f = interp1d(st2[v], d[i,v], kind='linear', bounds_error=False, fill_value=0.0)
            out[i,:] = f(td)
        except: pass
    return out

def smooth_d(d, w=5, p=2, ax=0):
    if w<p+2: w=p+2
    if w%2==0: w+=1
    s = np.copy(d)
    if ax==0:
        for j in range(d.shape[1]):
            try: s[:,j]=savgol_filter(d[:,j],w,p)
            except: pass
    else:
        for i in range(d.shape[0]):
            try: s[i,:]=savgol_filter(d[i,:],w,p)
            except: pass
    return s

def build_asym_zw(dmin, dmax):
    if dmin>=0: return [[0,"rgb(255,255,255)"],[1,"rgb(178,24,43)"]]
    if dmax<=0: return [[0,"rgb(33,102,172)"],[1,"rgb(255,255,255)"]]
    t = dmax-dmin; zf = abs(dmin)/t
    bs = [[0,"rgb(5,48,97)"],[zf*.25,"rgb(33,102,172)"],[zf*.5,"rgb(67,147,195)"],
          [zf*.75,"rgb(146,197,222)"],[zf*.9,"rgb(209,229,240)"],[zf,"rgb(255,255,255)"]]
    rs = [[zf,"rgb(255,255,255)"],[zf+(1-zf)*.1,"rgb(253,219,199)"],
          [zf+(1-zf)*.25,"rgb(244,165,130)"],[zf+(1-zf)*.5,"rgb(214,96,77)"],
          [zf+(1-zf)*.75,"rgb(178,24,43)"],[1.0,"rgb(103,0,31)"]]
    return bs + rs[1:]

def build_c11(cc):
    return [[i/10.0,c] for i,c in enumerate(cc)]

def mono_exp(t,A,tau,y0): return y0+A*np.exp(-t/tau)
def bi_exp(t,A1,t1,A2,t2,y0): return y0+A1*np.exp(-t/t1)+A2*np.exp(-t/t2)
def tri_exp(t,A1,t1,A2,t2,A3,t3,y0): return y0+A1*np.exp(-t/t1)+A2*np.exp(-t/t2)+A3*np.exp(-t/t3)

def fit_kin(t, y, mt="bi", ts=0.0):
    m = (t>=ts)&np.isfinite(y); tf,yf = t[m],y[m]
    if len(tf)<5: return None,None,None
    y0g=yf[-1]; Ag=yf[0]-y0g; tr=tf[-1]-tf[0]
    try:
        if mt=="mono":
            po,_=curve_fit(mono_exp,tf,yf,p0=[Ag,tr/5,y0g],bounds=([-np.inf,.01,-np.inf],[np.inf,tr*10,np.inf]),maxfev=10000)
            yc=mono_exp(tf,*po); pn=["A₁","τ₁ (ps)","y₀"]
        elif mt=="bi":
            po,_=curve_fit(bi_exp,tf,yf,p0=[Ag*.6,tr/20,Ag*.4,tr/2,y0g],bounds=([-np.inf,.01,-np.inf,.1,-np.inf],[np.inf,tr*5,np.inf,tr*10,np.inf]),maxfev=10000)
            yc=bi_exp(tf,*po); pn=["A₁","τ₁ (ps)","A₂","τ₂ (ps)","y₀"]
        elif mt=="tri":
            po,_=curve_fit(tri_exp,tf,yf,p0=[Ag*.4,tr/50,Ag*.35,tr/5,Ag*.25,tr,y0g],bounds=([-np.inf,.01,-np.inf,.1,-np.inf,1,-np.inf],[np.inf,tr*2,np.inf,tr*5,np.inf,tr*10,np.inf]),maxfev=10000)
            yc=tri_exp(tf,*po); pn=["A₁","τ₁ (ps)","A₂","τ₂ (ps)","A₃","τ₃ (ps)","y₀"]
        else: return None,None,None
        sr=np.sum((yf-yc)**2); st2=np.sum((yf-np.mean(yf))**2)
        r2=1-sr/st2 if st2>0 else 0
        return {"params":dict(zip(pn,po)),"r_squared":r2,"t_fit":tf,"y_fit":yf,"y_calc":yc,"residuals":yf-yc},po,pn
    except Exception as e: return None,None,str(e)

# ── Global Analysis helpers (scipy-based) ──
def _ga_parallel_C(t, taus):
    """Concentration matrix for parallel (independent) decay: C_i(t) = exp(-t/tau_i)"""
    n = len(taus)
    C = np.zeros((len(t), n))
    for i, tau in enumerate(taus):
        C[:, i] = np.exp(-t / tau)
    return C

def _ga_sequential_C(t, taus):
    """Concentration matrix for sequential decay: S1->S2->...->Sn.
    Uses analytic solution for first-order sequential kinetics."""
    n = len(taus)
    rates = [1.0 / tau for tau in taus]
    C = np.zeros((len(t), n))
    C[:, 0] = np.exp(-rates[0] * t)
    for j in range(1, n):
        for i in range(j + 1):
            prod_coeff = 1.0
            for m in range(j):
                prod_coeff *= rates[m]
            denom = 1.0
            for m in range(j + 1):
                if m != i:
                    diff = rates[m] - rates[i]
                    if abs(diff) < 1e-12:
                        diff = np.sign(diff) * 1e-12 if diff != 0 else 1e-12
                    denom *= diff
            C[:, j] += (prod_coeff / denom) * np.exp(-rates[i] * t)
    return C

def _ga_convolve_irf(C, t, irf_sigma):
    """Approximate IRF convolution via Gaussian filter."""
    from scipy.ndimage import gaussian_filter1d
    if irf_sigma <= 0 or len(t) < 3:
        return C
    dt = np.median(np.diff(t))
    if dt <= 0:
        return C
    sigma_pts = irf_sigma / dt
    if sigma_pts < 0.5:
        return C
    C_conv = np.copy(C)
    for j in range(C.shape[1]):
        C_conv[:, j] = gaussian_filter1d(C[:, j], sigma_pts, mode='constant', cval=0.0)
    return C_conv

def _ga_build_C(t, taus, model_type, use_irf, irf_center, irf_sigma):
    """Build concentration matrix with optional IRF."""
    t_shifted = t - irf_center if use_irf else t
    t_shifted = np.maximum(t_shifted, 0.0)
    if model_type == "parallel":
        C = _ga_parallel_C(t_shifted, taus)
    else:
        C = _ga_sequential_C(t_shifted, taus)
    if use_irf and irf_sigma > 0:
        C = _ga_convolve_irf(C, t, irf_sigma)
    return C

def _ga_residuals_flat(params, t, D, n_comp, model_type, use_irf):
    """Compute flattened residuals for least_squares."""
    taus = np.abs(params[:n_comp])
    taus = np.clip(taus, 0.01, 1e8)
    irf_center = params[n_comp] if use_irf else 0.0
    irf_sigma = abs(params[n_comp + 1]) if use_irf else 0.0
    C = _ga_build_C(t, taus, model_type, use_irf, irf_center, irf_sigma)
    try:
        SAS = np.linalg.lstsq(C, D.T, rcond=None)[0]
    except Exception:
        return D.ravel()
    fit = C @ SAS
    return (D.T - fit).ravel()

def run_global_analysis(wl, t, D, n_comp, tau_guesses, model_type="parallel",
                        use_irf=False, irf_center=0.3, irf_width=0.1):
    """Run global analysis and return results dict."""
    p0 = list(tau_guesses[:n_comp])
    lb = [0.01] * n_comp
    ub = [1e8] * n_comp
    if use_irf:
        p0 += [irf_center, irf_width]
        lb += [-10.0, 0.01]
        ub += [10.0, 5.0]

    result = least_squares(
        _ga_residuals_flat, p0,
        args=(t, D, n_comp, model_type, use_irf),
        bounds=(lb, ub), method='trf',
        max_nfev=500 * len(p0),
        x_scale='jac', verbose=0
    )

    opt_taus = np.abs(result.x[:n_comp])
    opt_taus = np.clip(opt_taus, 0.01, 1e8)
    irf_c_opt = result.x[n_comp] if use_irf else None
    irf_w_opt = abs(result.x[n_comp + 1]) if use_irf else None

    C_final = _ga_build_C(t, opt_taus, model_type, use_irf,
                          irf_c_opt if irf_c_opt is not None else 0.0,
                          irf_w_opt if irf_w_opt is not None else 0.0)

    SAS_final = np.linalg.lstsq(C_final, D.T, rcond=None)[0]
    D_fit = (C_final @ SAS_final).T
    D_res = D - D_fit

    ss_res = np.sum(D_res ** 2)
    ss_tot = np.sum((D - np.mean(D)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    sort_idx = np.argsort(opt_taus)
    opt_taus = opt_taus[sort_idx]
    SAS_final = SAS_final[sort_idx, :]
    C_final = C_final[:, sort_idx]

    return {
        "taus": opt_taus,
        "SAS": SAS_final,
        "C": C_final,
        "D_fit": D_fit,
        "D_res": D_res,
        "r2": r2,
        "rmse": np.sqrt(np.mean(D_res ** 2)),
        "nfev": result.nfev,
        "cost": result.cost,
        "irf_center": irf_c_opt,
        "irf_width": irf_w_opt,
        "model_type": model_type,
    }

CS_OPTS=["RdBu_r","RdBu","Spectral_r","Spectral","Viridis","Plasma","Inferno","Magma","Cividis","Turbo","PiYG_r","PiYG","BrBG_r","BrBG","PRGn_r","PRGn","RdYlBu_r","RdYlBu","RdYlGn_r","RdYlGn","Hot","Jet","Rainbow","Greys","Blues","Reds","Greens","Portland","Picnic","Earth","Electric","Blackbody"]
LC=["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf","#ff6347","#4682b4"]

# ── Sidebar ──
with st.sidebar:
    st.header("📂 Data Upload")
    uf = st.file_uploader("Upload TA CSV", type=["csv"])
    if uf:
        wavelengths,time_delays,raw_data = parse_ta_data(uf)
        st.success(f"✅ {uf.name}")
        st.caption(f"λ: {wavelengths[0]:.1f}–{wavelengths[-1]:.1f} nm ({len(wavelengths)})")
        st.caption(f"t: {time_delays[0]:.3f}–{time_delays[-1]:.1f} ps ({len(time_delays)})")
        st.caption(f"ΔOD: {np.nanmin(raw_data):.4f} – {np.nanmax(raw_data):.4f}")
        if "pd" not in st.session_state: st.session_state.pd = raw_data.copy()
        if "pl" not in st.session_state: st.session_state.pl = []
        if "cp" not in st.session_state: st.session_state.cp = []

if not uf:
    st.info("👈 Upload CSV to begin.")
    st.stop()

tab1,tab2,tab3,tab4,tab5 = st.tabs(["🔧 Preprocessing","📊 Visualization","📈 Kinetic Fitting","🧩 SVD Analysis","🌐 Global Analysis"])

# ═══ TAB 1 ═══
with tab1:
    st.header("Preprocessing Pipeline")
    wd = raw_data.copy(); lg = []
    ca,cb = st.columns(2)
    with ca:
        st.subheader("1️⃣ Background Subtraction")
        do_bg = st.checkbox("Enable BG subtraction", value=True)
        nt = time_delays[time_delays<0]
        if len(nt)>0 and do_bg:
            bgr = st.slider("BG range (ps)", float(nt[0]), 0.0, (float(nt[0]),-0.05), step=0.01)
            wd = bg_sub(wavelengths, time_delays, wd, bgr)
            lg.append(f"BG:[{bgr[0]:.3f},{bgr[1]:.3f}]")
    with cb:
        st.subheader("2️⃣ Smoothing")
        do_sm = st.checkbox("Enable smoothing", value=False)
        if do_sm:
            sax = st.radio("Along:", ["Wavelength","Time","Both"], horizontal=True)
            sw = st.slider("Window", 3, 21, 5, step=2)
            sp = st.slider("Poly", 1, 5, 2)
            if sax in ["Wavelength","Both"]: wd=smooth_d(wd,sw,sp,0); lg.append(f"Sm(λ):{sw}")
            if sax in ["Time","Both"]: wd=smooth_d(wd,sw,sp,1); lg.append(f"Sm(t):{sw}")
    st.divider()
    st.subheader("3️⃣ Chirp Correction — Interactive")
    do_ch = st.checkbox("Enable chirp correction", value=False)
    if do_ch:
        st.markdown("**Select points** on the heatmap or add manually to define t₀(λ).")
        cm1,cm2 = st.columns([3,1])
        with cm2:
            pord = st.slider("Poly order", 2, 6, 3, key="cpoly")
            st.markdown("**Manual entry:**")
            mw = st.number_input("λ", value=float(np.median(wavelengths)), min_value=float(wavelengths[0]), max_value=float(wavelengths[-1]), step=1.0, key="mw")
            mt2 = st.number_input("t₀ (ps)", value=0.3, min_value=-1.0, max_value=10.0, step=0.05, key="mt2")
            if st.button("➕ Add", key="acp"): st.session_state.cp.append((mw,mt2)); st.rerun()
            c1,c2 = st.columns(2)
            with c1:
                if st.button("↩️ Undo", key="ucp"):
                    if st.session_state.cp: st.session_state.cp.pop(); st.rerun()
            with c2:
                if st.button("🗑️ Clear", key="ccp"): st.session_state.cp=[]; st.rerun()
            st.markdown(f"**{len(st.session_state.cp)} pts**")
            if st.session_state.cp:
                st.dataframe(pd.DataFrame(st.session_state.cp, columns=["λ","t₀"]), hide_index=True, height=180)
        with cm1:
            em = (time_delays>-0.5)&(time_delays<10); te=time_delays[em]; de=wd[:,em]
            fcm = go.Figure(data=go.Heatmap(z=de.T, x=wavelengths, y=te, colorscale="RdBu_r",
                zmin=-np.nanmax(np.abs(de))*.8, zmax=np.nanmax(np.abs(de))*.8,
                colorbar_title="ΔOD",
                hovertemplate="λ:%{x:.1f}nm<br>t:%{y:.3f}ps<br>ΔOD:%{z:.5f}<extra></extra>"))
            if st.session_state.cp:
                pw=[p[0] for p in st.session_state.cp]; pt=[p[1] for p in st.session_state.cp]
                fcm.add_trace(go.Scatter(x=pw, y=pt, mode="markers",
                    marker=dict(size=10, color="#00FF00", symbol="x", line=dict(width=2,color="black")), name="Points"))
                if len(pw)>=pord+1:
                    co = fit_chirp_poly(np.array(pw), np.array(pt), pord)
                    if co is not None:
                        wf=np.linspace(wavelengths[0],wavelengths[-1],200)
                        fcm.add_trace(go.Scatter(x=wf, y=np.polyval(co,wf), mode="lines",
                            name=f"Fit(n={pord})", line=dict(color="#FF5722",width=2)))
            fcm.update_layout(xaxis_title="λ (nm)", yaxis_title="Time (ps)", height=450,
                margin=dict(t=20,b=50), dragmode="lasso", hovermode="closest")
            ck = st.plotly_chart(fcm, use_container_width=True, on_select="rerun", key="cch")
            if ck and ck.get("selection") and ck["selection"].get("points"):
                for p in ck["selection"]["points"]:
                    nx,ny = p.get("x"),p.get("y")
                    if nx is not None and ny is not None:
                        if not any(abs(ew-nx)<1 and abs(et-ny)<.05 for ew,et in st.session_state.cp):
                            st.session_state.cp.append((round(nx,1),round(ny,4)))
                st.rerun()
        if len(st.session_state.cp)>=pord+1:
            pw=np.array([p[0] for p in st.session_state.cp]); pt=np.array([p[1] for p in st.session_state.cp])
            co = fit_chirp_poly(pw, pt, pord)
            if co is not None:
                wd = apply_chirp(wavelengths, time_delays, wd, co)
                lg.append(f"Chirp:{len(pw)}pts,n={pord}")
                st.success(f"Chirp applied ({len(pw)} pts).")
        elif len(st.session_state.cp)>0:
            st.info(f"Need ≥{pord+1} pts (have {len(st.session_state.cp)}).")
    st.divider()
    if st.button("✅ Apply Preprocessing", type="primary", use_container_width=True):
        st.session_state.pd = wd.copy(); st.session_state.pl = lg
        st.success("Applied!")
    if st.session_state.pl: st.caption("**Steps:** "+" → ".join(st.session_state.pl))
    st.subheader("Preview")
    pvw = st.select_slider("λ (nm)", options=[f"{w:.1f}" for w in wavelengths], value=f"{wavelengths[len(wavelengths)//2]:.1f}")
    pi = np.argmin(np.abs(wavelengths-float(pvw)))
    fp = go.Figure()
    fp.add_trace(go.Scatter(x=time_delays,y=raw_data[pi,:],mode="lines",name="Raw",line=dict(color="gray",width=1)))
    fp.add_trace(go.Scatter(x=time_delays,y=wd[pi,:],mode="lines",name="Processed",line=dict(color="#2196F3",width=2)))
    fp.update_layout(xaxis_title="Time(ps)",yaxis_title="ΔOD",xaxis_type="log",
        xaxis=dict(range=[np.log10(.05),np.log10(time_delays[-1])]),height=280,margin=dict(t=20,b=40),legend=dict(x=.02,y=.98))
    fp.update_xaxes(tickvals=[.1,1,10,100,1000],ticktext=["0.1","1","10","100","1k"])
    st.plotly_chart(fp, use_container_width=True)

# ═══ TAB 2 ═══
with tab2:
    st.header("Data Visualization")
    data = st.session_state.pd
    with st.expander("🔍 Range", expanded=False):
        r1,r2 = st.columns(2)
        with r1: wlr = st.slider("λ (nm)", float(wavelengths[0]),float(wavelengths[-1]),(float(wavelengths[0]),float(wavelengths[-1])))
        with r2:
            pt2=time_delays[time_delays>0]; tlm=float(np.log10(max(pt2[0],.01))); tlx=float(np.log10(pt2[-1]))
            tlr=st.slider("Time (log₁₀ ps)",tlm,tlx,(tlm,tlx),step=0.1)
    wm=(wavelengths>=wlr[0])&(wavelengths<=wlr[1])
    tm2=(time_delays>0)&(time_delays>=10**tlr[0])&(time_delays<=10**tlr[1])
    ws,ts2,ds = wavelengths[wm], time_delays[tm2], data[np.ix_(wm,tm2)]

    st.subheader("2D ΔOD Heatmap")
    h1,h2 = st.columns([3,1])
    with h2:
        swap = st.checkbox("Swap axes", value=False, key="swap")
        st.markdown("**Color**")
        cmod = st.radio("Mode", ["Preset","Custom 11-stop","Asymmetric zero-white"], key="cm")
        if cmod=="Preset":
            cs = st.selectbox("Scale", CS_OPTS, index=0)
            rv = st.checkbox("Reverse", value=False, key="r1")
            if rv: cs = cs[:-2] if cs.endswith("_r") else cs+"_r"
        elif cmod=="Custom 11-stop":
            st.caption("11 colors: 0%→100%")
            d11=["#053061","#2166ac","#4393c3","#92c5de","#d1e5f0","#ffffff","#fddbc7","#f4a582","#d6604d","#b2182b","#67001f"]
            cc=[]; ccl2=st.columns(6)
            for i in range(11):
                with ccl2[i%6]: cc.append(st.color_picker(f"{i*10}%",value=d11[i],key=f"c{i}"))
            rv = st.checkbox("Reverse", value=False, key="r2")
            if rv: cc=cc[::-1]
            cs = build_c11(cc)
        else:
            st.info("ΔOD=0→white, asymmetric range.")
            rv = st.checkbox("Reverse", value=False, key="r3")
        st.markdown("**Range**")
        vmx = float(np.nanmax(np.abs(ds)))
        if cmod=="Asymmetric zero-white":
            dmn,dmx = float(np.nanmin(ds)), float(np.nanmax(ds))
            amn = st.slider("Min ΔOD", float(dmn*1.5), 0.0, float(dmn), step=.001, format="%.4f", key="amn")
            amx = st.slider("Max ΔOD", 0.0, float(dmx*1.5), float(dmx), step=.001, format="%.4f", key="amx")
            zmn,zmx = amn,amx
            cs = build_asym_zw(zmn,zmx)
            if rv:
                cs=[[1-c[0],c[1]] for c in reversed(cs)]; cs[0][0]=0.0; cs[-1][0]=1.0
        else:
            sym = st.checkbox("Symmetric", value=True, key="sy")
            if sym:
                cl = st.slider("±ΔOD", .001, float(vmx), float(vmx*.8), step=.001, format="%.3f")
                zmn,zmx = -cl,cl
            else: zmn,zmx = float(np.nanmin(ds)),float(np.nanmax(ds))
    with h1:
        lt = np.log10(ts2)
        tv=[float(v) for v in np.log10([.1,.5,1,5,10,50,100,500,1000,5000])]
        tt=["0.1","0.5","1","5","10","50","100","500","1k","5k"]
        if swap:
            fh = go.Figure(data=go.Heatmap(z=ds.T,x=ws,y=lt,colorscale=cs,zmin=zmn,zmax=zmx,
                colorbar_title="ΔOD",
                hovertemplate="λ:%{x:.1f}nm<br>t:%{customdata:.2f}ps<br>ΔOD:%{z:.5f}<extra></extra>",
                customdata=np.tile(ts2,(len(ws),1)).T))
            fh.update_layout(xaxis_title="λ (nm)",yaxis_title="Time (ps)",height=500,margin=dict(t=20,b=50,l=60,r=20),yaxis=dict(tickvals=tv,ticktext=tt))
        else:
            fh = go.Figure(data=go.Heatmap(z=ds,x=lt,y=ws,colorscale=cs,zmin=zmn,zmax=zmx,
                colorbar_title="ΔOD",
                hovertemplate="λ:%{y:.1f}nm<br>t:%{customdata:.2f}ps<br>ΔOD:%{z:.5f}<extra></extra>",
                customdata=np.tile(ts2,(len(ws),1))))
            fh.update_layout(xaxis_title="Time (ps)",yaxis_title="λ (nm)",height=500,margin=dict(t=20,b=50,l=60,r=20),xaxis=dict(tickvals=tv,ticktext=tt))
        st.plotly_chart(fh, use_container_width=True)
    st.divider()

    s1,s2 = st.columns(2)
    with s1:
        st.subheader("Spectral Slices")
        avt=time_delays[time_delays>0]; dft=[.5,1,5,10,50,100,500,1000]
        to = st.multiselect("Time (ps)", options=[f"{t:.2f}" for t in avt],
            default=[f"{avt[np.argmin(np.abs(avt-d))]:.2f}" for d in dft if d<=avt[-1]], max_selections=12)
        nsm = st.radio("Normalize", ["None","Max |ΔOD|","At specific λ"], horizontal=True, key="ns")
        nwl = None
        if nsm=="At specific λ":
            nwl = st.number_input("Norm λ (nm)", min_value=float(wavelengths[0]),max_value=float(wavelengths[-1]),value=float(np.median(wavelengths)),step=1.0,key="nwl")
        if to:
            fs = go.Figure()
            for k,tstr in enumerate(to):
                tv2=float(tstr); ti=np.argmin(np.abs(time_delays-tv2)); sp=data[wm,ti].copy()
                if nsm=="Max |ΔOD|" and np.max(np.abs(sp))>0: sp=sp/np.max(np.abs(sp))
                elif nsm=="At specific λ" and nwl:
                    ni=np.argmin(np.abs(ws-nwl)); nv=sp[ni]
                    if abs(nv)>1e-10: sp=sp/nv
                fs.add_trace(go.Scatter(x=ws,y=sp,mode="lines",name=f"{time_delays[ti]:.2f}ps",line=dict(color=LC[k%len(LC)],width=1.5)))
            fs.add_hline(y=0,line_dash="dash",line_color="gray",line_width=.5)
            if nsm=="At specific λ" and nwl:
                fs.add_vline(x=nwl,line_dash="dot",line_color="red",line_width=1,annotation_text=f"norm@{nwl:.0f}nm")
            yl2 = "ΔOD" if nsm=="None" else ("Norm. ΔOD" if nsm=="Max |ΔOD|" else f"ΔOD/ΔOD({nwl:.0f}nm)")
            fs.update_layout(xaxis_title="λ (nm)",yaxis_title=yl2,height=420,margin=dict(t=20,b=50),legend=dict(font=dict(size=10)))
            st.plotly_chart(fs, use_container_width=True)
    with s2:
        st.subheader("Kinetic Traces")
        dw=[550,610,660,700,750]
        wo = st.multiselect("λ (nm)", options=[f"{w:.1f}" for w in wavelengths],
            default=[f"{wavelengths[np.argmin(np.abs(wavelengths-d))]:.1f}" for d in dw if wavelengths[0]<=d<=wavelengths[-1]], max_selections=10)
        tsc = st.radio("Time axis", ["Log","Linear"], horizontal=True, key="tsc")
        nkm = st.radio("Normalize", ["None","Max |ΔOD|","At specific time"], horizontal=True, key="nk")
        nkt = None
        if nkm=="At specific time":
            nkt = st.number_input("Norm t (ps)", min_value=float(ts2[0]) if len(ts2)>0 else .1, max_value=float(ts2[-1]) if len(ts2)>0 else 5000., value=1.0, step=.1, key="nkt")
        if wo:
            fk = go.Figure()
            for k,wstr in enumerate(wo):
                wv=float(wstr); wi=np.argmin(np.abs(wavelengths-wv)); tr=data[wi,tm2].copy()
                if nkm=="Max |ΔOD|" and np.max(np.abs(tr))>0: tr=tr/np.max(np.abs(tr))
                elif nkm=="At specific time" and nkt:
                    tni=np.argmin(np.abs(ts2-nkt)); nv=tr[tni]
                    if abs(nv)>1e-10: tr=tr/nv
                fk.add_trace(go.Scatter(x=ts2,y=tr,mode="lines",name=f"{wavelengths[wi]:.1f}nm",line=dict(color=LC[k%len(LC)],width=1.5)))
            fk.add_hline(y=0,line_dash="dash",line_color="gray",line_width=.5)
            ykl = "ΔOD" if nkm=="None" else ("Norm. ΔOD" if nkm=="Max |ΔOD|" else f"ΔOD/ΔOD(t={nkt:.1f}ps)")
            lk=dict(xaxis_title="Time(ps)",yaxis_title=ykl,height=420,margin=dict(t=20,b=50),legend=dict(font=dict(size=10)))
            if tsc=="Log":
                lk["xaxis_type"]="log"
                lk["xaxis"]=dict(tickvals=[.1,.5,1,5,10,50,100,500,1000,5000],ticktext=["0.1","0.5","1","5","10","50","100","500","1k","5k"])
            fk.update_layout(**lk)
            st.plotly_chart(fk, use_container_width=True)
    st.divider()
    with st.expander("📥 Export"):
        edf=pd.DataFrame(data,index=pd.Index(wavelengths,name="λ"),columns=[f"{t:.4f}" for t in time_delays])
        buf=io.BytesIO(); edf.to_excel(buf,sheet_name="TA"); buf.seek(0)
        st.download_button("Download Excel",data=buf,file_name="TA_processed.xlsx",mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ═══ TAB 3 ═══
with tab3:
    st.header("Kinetic Fitting")
    data = st.session_state.pd
    f1,f2 = st.columns([1,2])
    with f1:
        fws = st.selectbox("λ (nm)", options=[f"{w:.1f}" for w in wavelengths], index=int(np.argmin(np.abs(wavelengths-660))))
        fwi = np.argmin(np.abs(wavelengths-float(fws)))
        mtp = st.radio("Model", ["mono","bi","tri"], format_func=lambda x:{"mono":"Mono","bi":"Bi","tri":"Tri"}[x]+"-exp", index=1)
        tst = st.number_input("Fit start (ps)", value=0.3, min_value=0.0, step=0.1)
        rf = st.button("🚀 Fit", type="primary", use_container_width=True)
        st.divider()
        bws = st.multiselect("Batch λ", options=[f"{w:.1f}" for w in wavelengths], default=[])
        rb = st.button("🔄 Batch Fit", use_container_width=True)
    with f2:
        if rf:
            tr=data[fwi,:]; res,po,pn=fit_kin(time_delays,tr,mt=mtp,ts=tst)
            if res:
                st.subheader(f"@ {wavelengths[fwi]:.1f} nm")
                rc=st.columns(len(res["params"])+1)
                for i,(n,v) in enumerate(res["params"].items()):
                    with rc[i]:
                        fm = f"{v:.2f}" if "τ" in n else (f"{v:.5f}" if n=="y₀" else f"{v:.4f}")
                        st.metric(n,fm)
                with rc[-1]: st.metric("R²",f"{res['r_squared']:.6f}")
                ff=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[.75,.25],vertical_spacing=.05)
                pm2=time_delays>0
                ff.add_trace(go.Scatter(x=time_delays[pm2],y=tr[pm2],mode="markers",name="Data",marker=dict(size=3,color="gray",opacity=.5)),row=1,col=1)
                ff.add_trace(go.Scatter(x=res["t_fit"],y=res["y_calc"],mode="lines",name="Fit",line=dict(color="#FF5722",width=2)),row=1,col=1)
                ff.add_trace(go.Scatter(x=res["t_fit"],y=res["residuals"],mode="markers",name="Res",marker=dict(size=2,color="#2196F3")),row=2,col=1)
                ff.add_hline(y=0,line_dash="dash",line_color="gray",row=2,col=1)
                ff.update_xaxes(type="log",row=1,col=1)
                ff.update_xaxes(type="log",title_text="Time(ps)",row=2,col=1,tickvals=[.1,1,10,100,1000],ticktext=["0.1","1","10","100","1k"])
                ff.update_yaxes(title_text="ΔOD",row=1,col=1); ff.update_yaxes(title_text="Res",row=2,col=1)
                ff.update_layout(height=500,margin=dict(t=20,b=50))
                st.plotly_chart(ff, use_container_width=True)
                if mtp in ["bi","tri"]:
                    st.subheader("Components")
                    fc=go.Figure(); tp=res["t_fit"]; p=res["params"]; ccl3=["#1f77b4","#ff7f0e","#2ca02c"]
                    comps=[(p["A₁"],p["τ₁ (ps)"],"τ₁"),(p["A₂"],p["τ₂ (ps)"],"τ₂")]
                    if mtp=="tri": comps.append((p["A₃"],p["τ₃ (ps)"],"τ₃"))
                    for k,(A,tau,lb) in enumerate(comps):
                        fc.add_trace(go.Scatter(x=tp,y=A*np.exp(-tp/tau),mode="lines",name=f"{lb}={tau:.2f}ps",line=dict(color=ccl3[k],width=1.5,dash="dash")))
                    fc.add_hline(y=0,line_dash="dot",line_color="gray")
                    fc.update_layout(xaxis_title="Time(ps)",yaxis_title="ΔOD",xaxis_type="log",height=300,margin=dict(t=20,b=50))
                    st.plotly_chart(fc, use_container_width=True)
            else: st.error(f"Failed. {pn if isinstance(pn,str) else ''}")
        if rb and bws:
            st.subheader("Batch Results")
            br=[]
            for ws_b in bws:
                wi=np.argmin(np.abs(wavelengths-float(ws_b))); res,_,_=fit_kin(time_delays,data[wi,:],mt=mtp,ts=tst)
                if res: rd={"λ":f"{wavelengths[wi]:.1f}"}; rd.update(res["params"]); rd["R²"]=res["r_squared"]; br.append(rd)
            if br:
                bdf=pd.DataFrame(br); st.dataframe(bdf,use_container_width=True,hide_index=True)
                ft=go.Figure()
                for tc in [c for c in bdf.columns if "τ" in c]:
                    ft.add_trace(go.Scatter(x=bdf["λ"].astype(float),y=bdf[tc],mode="lines+markers",name=tc,line=dict(width=2)))
                ft.update_layout(xaxis_title="λ(nm)",yaxis_title="τ(ps)",yaxis_type="log",height=350,margin=dict(t=20,b=50))
                st.plotly_chart(ft, use_container_width=True)
                buf=io.BytesIO(); bdf.to_excel(buf,index=False,sheet_name="Batch"); buf.seek(0)
                st.download_button("📥 Download",buf,file_name="TA_batch.xlsx",mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ═══ TAB 4 ═══
with tab4:
    st.header("SVD / Global Analysis")
    data = st.session_state.pd
    pm3=time_delays>0; dp=data[:,pm3]; tp3=time_delays[pm3]
    vc=~np.all(dp==0,axis=0)&~np.any(np.isnan(dp),axis=0); ds2,ts3=dp[:,vc],tp3[vc]
    if st.button("🧮 Run SVD", type="primary"):
        U,s,Vt=svd(ds2,full_matrices=False)
        st.session_state.sU=U; st.session_state.ss=s; st.session_state.sV=Vt; st.session_state.st=ts3
        st.success("Done!")
    if "ss" in st.session_state:
        s=st.session_state.ss; U=st.session_state.sU; Vt=st.session_state.sV; ts3=st.session_state.st
        v1,v2=st.columns(2)
        with v1:
            ns2=min(20,len(s)); fsc=go.Figure()
            fsc.add_trace(go.Scatter(x=list(range(1,ns2+1)),y=s[:ns2],mode="lines+markers",marker=dict(size=8,color="#2196F3"),line=dict(width=2)))
            fsc.update_layout(title="Scree Plot",xaxis_title="Comp",yaxis_title="SV",yaxis_type="log",height=350,margin=dict(t=40,b=50),xaxis=dict(dtick=1))
            st.plotly_chart(fsc, use_container_width=True)
            vr=s**2; vp=vr/np.sum(vr)*100; cv=np.cumsum(vp); nm=min(8,len(s))
            st.dataframe(pd.DataFrame({"#":range(1,nm+1),"SV":[f"{s[i]:.4f}" for i in range(nm)],"Var%":[f"{vp[i]:.2f}" for i in range(nm)],"Cum%":[f"{cv[i]:.2f}" for i in range(nm)]}),use_container_width=True,hide_index=True)
        with v2:
            nc=st.slider("Components",1,min(8,len(s)),3); ccl4=["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f"]
            fu=go.Figure()
            for k in range(nc): fu.add_trace(go.Scatter(x=wavelengths,y=U[:,k]*s[k],mode="lines",name=f"C{k+1}",line=dict(color=ccl4[k%8],width=1.5)))
            fu.add_hline(y=0,line_dash="dash",line_color="gray",line_width=.5)
            fu.update_layout(title="Spectra(U×S)",xaxis_title="λ(nm)",yaxis_title="Amp",height=300,margin=dict(t=40,b=50))
            st.plotly_chart(fu, use_container_width=True)
            fv=go.Figure()
            for k in range(nc): fv.add_trace(go.Scatter(x=ts3,y=Vt[k,:],mode="lines",name=f"C{k+1}",line=dict(color=ccl4[k%8],width=1.5)))
            fv.add_hline(y=0,line_dash="dash",line_color="gray",line_width=.5)
            fv.update_layout(title="Kinetics(Vᵀ)",xaxis_title="Time(ps)",yaxis_title="Amp",xaxis_type="log",height=300,margin=dict(t=40,b=50),xaxis=dict(tickvals=[.1,1,10,100,1000],ticktext=["0.1","1","10","100","1k"]))
            st.plotly_chart(fv, use_container_width=True)
        st.divider(); st.subheader("Reconstruction")
        nr=st.slider("N comp",1,min(10,len(s)),3,key="nr")
        rec=U[:,:nr]@np.diag(s[:nr])@Vt[:nr,:]; res2=ds2-rec
        r1,r2=st.columns(2)
        with r1:
            fr=go.Figure(data=go.Heatmap(z=rec,x=np.log10(ts3),y=wavelengths,colorscale="RdBu_r",zmin=-np.nanmax(np.abs(ds2))*.8,zmax=np.nanmax(np.abs(ds2))*.8,colorbar_title="ΔOD"))
            fr.update_layout(title=f"Recon(N={nr})",xaxis_title="Time(ps)",yaxis_title="λ(nm)",height=380,margin=dict(t=40,b=50),xaxis=dict(tickvals=[float(v) for v in np.log10([.1,1,10,100,1000])],ticktext=["0.1","1","10","100","1k"]))
            st.plotly_chart(fr, use_container_width=True)
        with r2:
            frs=go.Figure(data=go.Heatmap(z=res2,x=np.log10(ts3),y=wavelengths,colorscale="RdBu_r",zmin=-np.nanmax(np.abs(res2)),zmax=np.nanmax(np.abs(res2)),colorbar_title="ΔOD"))
            frs.update_layout(title=f"Residual(RMSE:{np.sqrt(np.mean(res2**2)):.6f})",xaxis_title="Time(ps)",yaxis_title="λ(nm)",height=380,margin=dict(t=40,b=50),xaxis=dict(tickvals=[float(v) for v in np.log10([.1,1,10,100,1000])],ticktext=["0.1","1","10","100","1k"]))
            st.plotly_chart(frs, use_container_width=True)

# ═══ TAB 5: Global Analysis (scipy-based) ═══
with tab5:
    st.header("Global / Target Analysis")
    st.caption("Scipy-based global fitting — parallel (DADS) or sequential (EADS) decay models")
    data = st.session_state.pd

    gc1, gc2 = st.columns([1, 2])
    with gc1:
        st.subheader("Model Setup")
        ga_model = st.radio("Model type", ["Parallel → DADS", "Sequential → EADS"], index=0, key="ga_model")
        n_comp_ga = st.slider("Number of components", 1, 6, 3, key="ga_ncomp")
        st.markdown("**Initial τ guesses (ps):**")
        tau_guesses = []
        default_taus = [0.5, 5.0, 50.0, 500.0, 5000.0, 50000.0]
        tau_cols = st.columns(min(n_comp_ga, 3))
        for i in range(n_comp_ga):
            with tau_cols[i % len(tau_cols)]:
                tg = st.number_input(f"τ{i+1}", value=default_taus[i] if i < len(default_taus) else 10.0**(i+1),
                                     min_value=0.01, step=0.1, format="%.2f", key=f"ga_tau{i}")
                tau_guesses.append(tg)

        st.divider()
        use_irf = st.checkbox("Enable IRF (Gaussian)", value=True, key="ga_irf_on")
        irf_c, irf_w = 0.3, 0.1
        if use_irf:
            irf_c = st.number_input("IRF center (ps)", value=0.3, step=0.05, format="%.3f", key="ga_irf_c")
            irf_w = st.number_input("IRF width σ (ps)", value=0.10, min_value=0.01, step=0.01, format="%.3f", key="ga_irf_w")

        st.divider()
        ga_tstart = st.number_input("Fit start (ps)", value=0.0, step=0.1, key="ga_tstart")

        st.divider()
        run_ga = st.button("🚀 Run Global Analysis", type="primary", use_container_width=True, key="run_ga")

    with gc2:
        if run_ga:
            with st.spinner("Running global optimization..."):
                try:
                    pm_ga = time_delays >= ga_tstart
                    t_ga = time_delays[pm_ga]
                    D_ga = data[:, pm_ga]
                    valid = ~np.any(np.isnan(D_ga), axis=0)
                    t_ga = t_ga[valid]
                    D_ga = D_ga[:, valid]

                    mtype = "parallel" if "Parallel" in ga_model else "sequential"
                    ga_res = run_global_analysis(
                        wavelengths, t_ga, D_ga, n_comp_ga, tau_guesses,
                        model_type=mtype, use_irf=use_irf,
                        irf_center=irf_c, irf_width=irf_w
                    )
                    st.session_state.ga_result = ga_res
                    st.session_state.ga_t = t_ga
                    st.session_state.ga_D = D_ga
                    st.success(f"✅ Done! (nfev: {ga_res['nfev']}, R²: {ga_res['r2']:.6f})")
                except Exception as e:
                    st.error(f"❌ Failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())

        if "ga_result" in st.session_state:
            ga_res = st.session_state.ga_result
            t_ga = st.session_state.ga_t
            D_ga = st.session_state.ga_D
            n_comp_res = len(ga_res["taus"])
            ccl5 = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b"]
            is_dads = ga_res["model_type"] == "parallel"

            # ── Lifetimes ──
            st.subheader("Optimized Lifetimes")
            tc_cols = st.columns(n_comp_res + 2)
            for j, tau in enumerate(ga_res["taus"]):
                with tc_cols[j]:
                    if tau < 1:
                        st.metric(f"τ{j+1}", f"{tau*1000:.1f} fs")
                    elif tau < 1000:
                        st.metric(f"τ{j+1}", f"{tau:.2f} ps")
                    else:
                        st.metric(f"τ{j+1}", f"{tau/1000:.2f} ns")
            with tc_cols[n_comp_res]:
                st.metric("R²", f"{ga_res['r2']:.6f}")
            with tc_cols[n_comp_res + 1]:
                st.metric("RMSE", f"{ga_res['rmse']:.2e}")
            if ga_res["irf_center"] is not None:
                ic1, ic2 = st.columns(2)
                with ic1: st.metric("IRF center", f"{ga_res['irf_center']:.3f} ps")
                with ic2: st.metric("IRF width", f"{ga_res['irf_width']:.3f} ps")

            st.divider()

            # ── DADS / EADS ──
            sas_label = "DADS" if is_dads else "EADS"
            st.subheader(f"Species-Associated Spectra ({sas_label})")
            fsas = go.Figure()
            for k in range(n_comp_res):
                tau = ga_res["taus"][k]
                if tau < 1:
                    lbl = f"τ{k+1}={tau*1000:.0f}fs"
                elif tau < 1000:
                    lbl = f"τ{k+1}={tau:.1f}ps"
                else:
                    lbl = f"τ{k+1}={tau/1000:.1f}ns"
                fsas.add_trace(go.Scatter(
                    x=wavelengths, y=ga_res["SAS"][k, :],
                    mode="lines", name=lbl,
                    line=dict(color=ccl5[k % len(ccl5)], width=2),
                ))
            fsas.add_hline(y=0, line_dash="dash", line_color="gray", line_width=0.5)
            fsas.update_layout(
                xaxis_title="Wavelength (nm)", yaxis_title=f"{sas_label} (ΔOD)",
                height=400, margin=dict(t=20, b=50),
                legend=dict(font=dict(size=11)),
            )
            st.plotly_chart(fsas, use_container_width=True)

            # ── Concentration profiles ──
            st.subheader("Concentration Profiles")
            fconc = go.Figure()
            for k in range(n_comp_res):
                tau = ga_res["taus"][k]
                if tau < 1:
                    lbl = f"S{k+1} (τ={tau*1000:.0f}fs)"
                elif tau < 1000:
                    lbl = f"S{k+1} (τ={tau:.1f}ps)"
                else:
                    lbl = f"S{k+1} (τ={tau/1000:.1f}ns)"
                fconc.add_trace(go.Scatter(
                    x=t_ga, y=ga_res["C"][:, k],
                    mode="lines", name=lbl,
                    line=dict(color=ccl5[k % len(ccl5)], width=2),
                ))
            fconc.update_layout(
                xaxis_title="Time (ps)", yaxis_title="Population",
                xaxis_type="log", height=350, margin=dict(t=20, b=50),
                xaxis=dict(tickvals=[.1, 1, 10, 100, 1000, 5000],
                           ticktext=["0.1", "1", "10", "100", "1k", "5k"]),
            )
            st.plotly_chart(fconc, use_container_width=True)

            st.divider()

            # ── Fitted vs Residual heatmaps ──
            st.subheader("Fit Quality")
            fq1, fq2 = st.columns(2)
            lt_ga = np.log10(np.maximum(t_ga, 1e-6))
            tv_ga = [float(v) for v in np.log10([.1, 1, 10, 100, 1000])]
            tt_ga = ["0.1", "1", "10", "100", "1k"]
            zmx_ga = float(np.nanmax(np.abs(D_ga))) * 0.8

            with fq1:
                f_fit = go.Figure(data=go.Heatmap(
                    z=ga_res["D_fit"], x=lt_ga, y=wavelengths,
                    colorscale="RdBu_r", zmin=-zmx_ga, zmax=zmx_ga,
                    colorbar_title="ΔOD",
                ))
                f_fit.update_layout(
                    title="Fitted", xaxis_title="Time (ps)", yaxis_title="λ (nm)",
                    height=380, margin=dict(t=40, b=50),
                    xaxis=dict(tickvals=tv_ga, ticktext=tt_ga),
                )
                st.plotly_chart(f_fit, use_container_width=True)

            with fq2:
                zmx_res = float(np.nanmax(np.abs(ga_res["D_res"])))
                f_res = go.Figure(data=go.Heatmap(
                    z=ga_res["D_res"], x=lt_ga, y=wavelengths,
                    colorscale="RdBu_r", zmin=-zmx_res, zmax=zmx_res,
                    colorbar_title="ΔOD",
                ))
                f_res.update_layout(
                    title=f"Residual (RMSE: {ga_res['rmse']:.2e})",
                    xaxis_title="Time (ps)", yaxis_title="λ (nm)",
                    height=380, margin=dict(t=40, b=50),
                    xaxis=dict(tickvals=tv_ga, ticktext=tt_ga),
                )
                st.plotly_chart(f_res, use_container_width=True)

            st.divider()

            # ── Kinetic trace comparison ──
            st.subheader("Kinetic Trace Comparison (Data vs Fit)")
            ga_wl_sel = st.multiselect(
                "Select wavelengths (nm)",
                options=[f"{w:.1f}" for w in wavelengths],
                default=[f"{wavelengths[np.argmin(np.abs(wavelengths - d))]:.1f}"
                         for d in [500, 600, 700] if wavelengths[0] <= d <= wavelengths[-1]],
                max_selections=6, key="ga_wl_comp"
            )
            if ga_wl_sel:
                n_sel = len(ga_wl_sel)
                fcomp = make_subplots(
                    rows=n_sel, cols=1, shared_xaxes=True,
                    vertical_spacing=0.04,
                    subplot_titles=[f"λ = {w} nm" for w in ga_wl_sel],
                )
                for idx, wstr in enumerate(ga_wl_sel):
                    wv = float(wstr)
                    wi = np.argmin(np.abs(wavelengths - wv))
                    fcomp.add_trace(go.Scatter(
                        x=t_ga, y=D_ga[wi, :],
                        mode="markers", name="Data" if idx == 0 else None,
                        marker=dict(size=3, color="gray", opacity=0.4),
                        showlegend=(idx == 0), legendgroup="data",
                    ), row=idx + 1, col=1)
                    fcomp.add_trace(go.Scatter(
                        x=t_ga, y=ga_res["D_fit"][wi, :],
                        mode="lines", name="Fit" if idx == 0 else None,
                        line=dict(color="#FF5722", width=1.5),
                        showlegend=(idx == 0), legendgroup="fit",
                    ), row=idx + 1, col=1)
                    fcomp.update_xaxes(type="log", row=idx + 1, col=1)
                    fcomp.update_yaxes(title_text="ΔOD", row=idx + 1, col=1)
                fcomp.update_xaxes(
                    title_text="Time (ps)", row=n_sel, col=1,
                    tickvals=[.1, 1, 10, 100, 1000], ticktext=["0.1", "1", "10", "100", "1k"],
                )
                fcomp.update_layout(height=200 * n_sel + 80, margin=dict(t=40, b=50))
                st.plotly_chart(fcomp, use_container_width=True)

            st.divider()

            # ── Export ──
            with st.expander("📥 Export Global Analysis Results"):
                ex1, ex2, ex3 = st.columns(3)
                sas_df = pd.DataFrame(ga_res["SAS"].T, index=pd.Index(wavelengths, name="λ"),
                                      columns=[f"τ{k+1}={ga_res['taus'][k]:.2f}ps" for k in range(n_comp_res)])
                buf_sas = io.BytesIO(); sas_df.to_excel(buf_sas, sheet_name=sas_label); buf_sas.seek(0)
                with ex1:
                    st.download_button(f"📥 {sas_label}", buf_sas, file_name=f"TA_{sas_label}.xlsx",
                                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                conc_df = pd.DataFrame(ga_res["C"], index=pd.Index(t_ga, name="t(ps)"),
                                       columns=[f"S{k+1}" for k in range(n_comp_res)])
                buf_c = io.BytesIO(); conc_df.to_excel(buf_c, sheet_name="Conc"); buf_c.seek(0)
                with ex2:
                    st.download_button("📥 Concentrations", buf_c, file_name="TA_GA_conc.xlsx",
                                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                tau_df = pd.DataFrame({
                    "Component": [f"S{k+1}" for k in range(n_comp_res)],
                    "τ (ps)": ga_res["taus"],
                    "k (ps⁻¹)": [1.0 / t for t in ga_res["taus"]],
                })
                buf_tau = io.BytesIO(); tau_df.to_excel(buf_tau, index=False, sheet_name="Lifetimes"); buf_tau.seek(0)
                with ex3:
                    st.download_button("📥 Lifetimes", buf_tau, file_name="TA_GA_lifetimes.xlsx",
                                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.divider()
st.caption("**TA Data Analyzer** v3.0 | Built with Streamlit & Plotly")
