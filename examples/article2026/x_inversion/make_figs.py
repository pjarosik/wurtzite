"""Figures for 'Analytic inversion of the x-coordinate' note."""
import numpy as np, matplotlib as mpl, matplotlib.pyplot as plt
from scipy.special import lambertw
from scipy.optimize import brentq
mpl.rcParams.update({"font.size":9,"axes.titlesize":9.5,"pdf.fonttype":42})

nu, b, r0 = 0.35, 1.0, 0.5
A = (1-2*nu)/(4*np.pi*(1-nu)); B = b/(4*np.pi*(1-nu))
INK,MUTED,GRID = "#1a1a18","#6b6b66","#dcdbd6"
BLUE,ORANGE,RED = "#256abf","#c1440e","#c1440e"
RAMP = ["#9ec5f4","#6da7ec","#3987e5","#256abf","#184f95","#0d366b"]

def fx(x,y): return b/(2*np.pi)*(np.arctan2(y,x)+x*y/(2*(1-nu)*(x*x+y*y)))
def fy(x,y): return -b/(8*np.pi*(1-nu))*((1-2*nu)*np.log((x*x+y*y)/r0**2)-2*y*y/(x*x+y*y))
def X_of(x,y): return x-fx(x,y)
def Y_of(x,y): return y-fy(x,y)
def xX(X,th): return X + b/(2*np.pi)*(th + np.sin(2*th)/(4*(1-nu)))
def rX(X,th):
    c=np.cos(th); return np.where(np.abs(c)>1e-12, xX(X,th)/np.where(np.abs(c)>1e-12,c,1), np.nan)
def rY(Y,th,k=0):
    m=np.sin(th); ok=np.abs(m)>1e-12
    z=np.where(ok,(r0*m/(A*b))*np.exp((Y+B*m*m)/(A*b)),1.0)
    w=lambertw(z.astype(complex),k)
    r=np.where(ok,(A*b/np.where(ok,m,1))*w.real, r0*np.exp(Y/(A*b)))
    return np.where(np.abs(w.imag)<1e-12, r, np.nan)

def style(ax):
    ax.grid(True,color=GRID,lw=0.5); ax.set_axisbelow(True)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    for s in ("left","bottom"): ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED,labelsize=8)

# ---------------------------------------------------------------- fig 1
def plane(X,n=40000):
    segs,cur=[],[]
    for th in np.linspace(-np.pi+1e-9,np.pi-1e-9,n):
        c=np.cos(th); x=xX(X,th); r=x/c if abs(c)>1e-9 else np.nan
        if np.isfinite(r) and r>0: cur.append((r*c,r*np.sin(th)))
        elif cur: segs.append(np.array(cur)); cur=[]
    if cur: segs.append(np.array(cur))
    return segs
def panel(ax,Xs,lim,title):
    for j,(X,col) in enumerate(zip(Xs,RAMP)):
        for s in plane(X): ax.plot(s[:,0],s[:,1],color=col,lw=1.6,solid_capstyle="round",zorder=3)
        frac=(0.86,0.64)[j%2]          # stagger heights so labels never collide
        top=[s for s in plane(X) if s[:,1].max()>frac*lim]
        if top:
            s=max(top,key=lambda a:a[:,1].max()); i=np.argmin(np.abs(s[:,1]-frac*lim))
            ax.annotate(f"{X:+.2f}",(s[i,0],s[i,1]),color=col,fontsize=7,fontweight="bold",
                        ha="center",va="center",zorder=4,
                        bbox=dict(boxstyle="round,pad=0.12",fc="white",ec="none"))
    ax.plot([-lim,0],[0,0],color=INK,lw=1.0,ls=(0,(4,3)),zorder=2)
    ax.plot([0,0],[0,lim],color=ORANGE,lw=1.8,zorder=6)
    ax.plot([0,0],[-lim,0],color=ORANGE,lw=1.8,ls=(0,(1,1.6)),zorder=6)
    ax.plot(0,0,"o",ms=4,mfc="white",mec=INK,mew=1.2,zorder=7)
    ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_aspect("equal")
    ax.set_title(title,color=INK,loc="left",pad=6); ax.set_xlabel(r"$x\ [\hat b^{\perp}]$",color=MUTED)
    style(ax)
fig,axes=plt.subplots(1,2,figsize=(6.6,3.5))
panel(axes[0],[-1.20,-0.80,-0.45,-0.05,0.40,1.00],1.6,r"(a) $\pm1.6\,\hat b^{\perp}$; labels give $X$")
panel(axes[1],[-0.30,-0.20,-0.10,0.00,0.10,0.20],0.30,r"(b) core, $\pm0.30\,\hat b^{\perp}$")
axes[0].set_ylabel(r"$y\ [\hat b^{\perp}]$",color=MUTED)
fig.tight_layout(); fig.savefig("fig_planes.pdf"); plt.close(fig)

# ---------------------------------------------------------------- fig 2
TH=np.linspace(-np.pi+1e-6,np.pi-1e-6,4000)
fig,axes=plt.subplots(1,2,figsize=(6.6,3.0))
for ax,(px,py),ttl in zip(axes,[(1.20,0.70),(0.06,-0.03)],
                          [r"(a) far field: one intersection",r"(b) core: two intersections"]):
    X,Y=X_of(px,py),Y_of(px,py)
    a=rX(X,TH); ax.plot(TH,np.where(a>0,a,np.nan),color=BLUE,lw=1.8,label=r"$r_X(\theta;X)$",zorder=3)
    for k,ls in ((0,"-"),(-1,(0,(4,2)))):
        c=rY(Y,TH,k)
        ax.plot(TH,np.where(c>0,c,np.nan),color=ORANGE,lw=1.6,ls=ls,zorder=3,
                label=(r"$r_Y(\theta;Y)$, $W_0$" if k==0 else r"$r_Y(\theta;Y)$, $W_{-1}$"))
    with np.errstate(all="ignore"):
        for k in (0,-1):
            f=np.where((a>0)&(rY(Y,TH,k)>0),a-rY(Y,TH,k),np.nan); s=np.sign(f)
            for i in np.where(np.isfinite(f[:-1])&np.isfinite(f[1:])&(s[:-1]*s[1:]<0))[0]:
                t=brentq(lambda u: float(rX(X,np.array([u]))[0]-rY(Y,np.array([u]),k)[0]),TH[i],TH[i+1])
                ax.plot(t,float(rX(X,np.array([t]))[0]),"o",ms=7,mfc="white",mec=INK,mew=1.6,zorder=6)
    ax.set_ylim(0,max(0.35,2.2*np.hypot(px,py))); ax.set_xlim(-np.pi,np.pi)
    ax.set_xticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi]); ax.set_xticklabels([r"$-\pi$",r"$-\pi/2$","0",r"$\pi/2$",r"$\pi$"])
    ax.set_title(ttl,color=INK,loc="left",pad=6); ax.set_xlabel(r"$\theta$",color=MUTED); style(ax)
axes[0].set_ylabel(r"$r\ [\hat b^{\perp}]$",color=MUTED)
axes[0].legend(frameon=False,fontsize=7,labelcolor=INK,loc="upper left")
fig.tight_layout(); fig.savefig("fig_scalar.pdf"); plt.close(fig)

# ---------------------------------------------------------------- fig 3
def roots(X,Y):
    out=[]
    for k in (0,-1):
        a,c=rX(X,TH),rY(Y,TH,k)
        with np.errstate(all="ignore"): f=np.where((a>0)&(c>0),a-c,np.nan)
        s=np.sign(f)
        for i in np.where(np.isfinite(f[:-1])&np.isfinite(f[1:])&(s[:-1]*s[1:]<0))[0]:
            try: t=brentq(lambda u: float(rX(X,np.array([u]))[0]-rY(Y,np.array([u]),k)[0]),TH[i],TH[i+1],xtol=1e-14)
            except Exception: continue
            r=float(rX(X,np.array([t]))[0]); p=(r*np.cos(t),r*np.sin(t))
            if not any(abs(p[0]-q[0])<1e-8 and abs(p[1]-q[1])<1e-8 for q in out): out.append(p)
    return out
rng=np.random.default_rng(1); Rs=np.array([0.02,0.05,0.1,0.2,0.4,0.8,1.5,3.0]); frac=[]
for R in Rs:
    n2=0
    for _ in range(150):
        t=rng.uniform(-np.pi,np.pi); x,y=R*np.cos(t),R*np.sin(t)
        if len(roots(X_of(x,y),Y_of(x,y)))>=2: n2+=1
    frac.append(100*n2/150)
fig,ax=plt.subplots(figsize=(4.4,2.9))
ax.plot(Rs,frac,"-o",color=BLUE,lw=1.8,ms=6,mfc="white",mec=BLUE,mew=1.8)
ax.axvline(0.05,color=MUTED,lw=1.0,ls=(0,(4,3)))
ax.annotate(r"$r=\hat b^{\perp}/20$ cutoff (Sec. 5.2)",(0.05,52),xytext=(6,0),
            textcoords="offset points",color=MUTED,fontsize=7.5,va="center")
ax.set_xscale("log"); ax.set_xlabel(r"$r\ [\hat b^{\perp}]$",color=MUTED)
ax.set_ylabel(r"share with $\geq 2$ preimages [%]",color=MUTED)
ax.set_title("Non-injectivity of the deformation map",color=INK,loc="left",pad=6)
ax.set_ylim(-4,104); style(ax)
fig.tight_layout(); fig.savefig("fig_preimages.pdf"); plt.close(fig)
print("figures written:", frac)
