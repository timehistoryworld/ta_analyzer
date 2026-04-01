import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit
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

tab1,tab2,tab3,tab4 = st.tabs(["🔧 Preprocessing","📊 Visualization","📈 Kinetic Fitting","🧩 SVD Analysis"])

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
                st.dataframe(pd.DataFrame(st.session_state.cp, columns=["λ","t₀"]), use_container_width=True, hide_index=True, height=180)
        with cm1:
            em = (time_delays>-0.5)&(time_delays<10); te=time_delays[em]; de=wd[:,em]
            fcm = go.Figure(data=go.Heatmap(z=de.T, x=wavelengths, y=te, colorscale="RdBu_r",
                zmin=-np.nanmax(np.abs(de))*.8, zmax=np.nanmax(np.abs(de))*.8,
                colorbar=dict(title=dict(text="ΔOD")),
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
                colorbar=dict(title=dict(text="ΔOD",side="right")),
                hovertemplate="λ:%{x:.1f}nm<br>t:%{customdata:.2f}ps<br>ΔOD:%{z:.5f}<extra></extra>",
                customdata=np.tile(ts2,(len(ws),1)).T))
            fh.update_layout(xaxis_title="λ (nm)",yaxis_title="Time (ps)",height=500,margin=dict(t=20,b=50,l=60,r=20),yaxis=dict(tickvals=tv,ticktext=tt))
        else:
            fh = go.Figure(data=go.Heatmap(z=ds,x=lt,y=ws,colorscale=cs,zmin=zmn,zmax=zmx,
                colorbar=dict(title=dict(text="ΔOD",side="right")),
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
            for ws in bws:
                wi=np.argmin(np.abs(wavelengths-float(ws))); res,_,_=fit_kin(time_delays,data[wi,:],mt=mtp,ts=tst)
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
            fr=go.Figure(data=go.Heatmap(z=rec,x=np.log10(ts3),y=wavelengths,colorscale="RdBu_r",zmin=-np.nanmax(np.abs(ds2))*.8,zmax=np.nanmax(np.abs(ds2))*.8,colorbar=dict(title=dict(text="ΔOD"))))
            fr.update_layout(title=f"Recon(N={nr})",xaxis_title="Time(ps)",yaxis_title="λ(nm)",height=380,margin=dict(t=40,b=50),xaxis=dict(tickvals=[float(v) for v in np.log10([.1,1,10,100,1000])],ticktext=["0.1","1","10","100","1k"]))
            st.plotly_chart(fr, use_container_width=True)
        with r2:
            frs=go.Figure(data=go.Heatmap(z=res2,x=np.log10(ts3),y=wavelengths,colorscale="RdBu_r",zmin=-np.nanmax(np.abs(res2)),zmax=np.nanmax(np.abs(res2)),colorbar=dict(title=dict(text="ΔOD"))))
            frs.update_layout(title=f"Residual(RMSE:{np.sqrt(np.mean(res2**2)):.6f})",xaxis_title="Time(ps)",yaxis_title="λ(nm)",height=380,margin=dict(t=40,b=50),xaxis=dict(tickvals=[float(v) for v in np.log10([.1,1,10,100,1000])],ticktext=["0.1","1","10","100","1k"]))
            st.plotly_chart(frs, use_container_width=True)

st.divider()
st.caption("**TA Data Analyzer** v2.0 | Built with Streamlit & Plotly")
