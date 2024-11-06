# IN: diff, comm, srcL, vsrmL, vsrmT, itmT <- params
# Set printing=True and plotting=True for interactive graphs in browser (False by default)
# OUT: Dictionary of dictionaries listed below as KEY1 (KEY2)
# xaxis
# prmT <- Optimised to maximise arm cavity power
# power_data (armX, armY, laser, onBS, PRC, SRC)
# peak_data (sens, freq, fwhm, lhm_f, rhm_f) <- l/rhm = left/right half max
# curve_data (NSR_with_RP, NSR_without_RP, signal)

import finesse
import numpy as np
from scipy.signal import find_peaks, peak_widths
from scipy import interpolate
import plotly.graph_objects as go


def nemo_sensitivity(diff, comm, srcL, vsrmL, vsrmT, itmT, printing=False, plotting=False):
    # Adapts from SRC_tunability_DCreadout_changedBW_fin3exp
    fsig = np.geomspace(1e3, 10e3, 151)
    peak_sens = np.nan
    peak_f = np.nan
    peak_bw = np.nan
    left_f = np.nan
    right_f = np.nan

    diff = float(diff)
    comm = float(comm)
    srcL = float(srcL)
    vsrmL = float(vsrmL)
    vsrmT = float(vsrmT)
    itmT = float(itmT)
    prmT = find_optimal_prmT(diff, comm, srcL, vsrmL, vsrmT, itmT)

    kat = finesse.Model()
    kat.parse(
        f"""
    ###########################################################################
    ###   Variables
    ###########################################################################
    var Larm 4000
    var Mtm  74.1
    var itmT {itmT}
    var lmichx 4.5
    var lmichy 4.45

    ###########################################################################
    ###   Input optics
    ###########################################################################
    l L0 500

    # CHECK Input laser power
    pd P_in L0.p1.o

    s l_in L0.p1 prm.p1
    # Power recycling mirror
    m prm T={prmT} L=2e-05 phi=90
    s prc prm.p2 bs.p1 L=53

    # Central beamsplitter
    bs bs T=0.5 L=0 alpha=45

    # CHECK Laser power incident on BS
    pd P_BS bs.p1.i
    # CHECK PRC Power
    pd P_PRC bs.p1.o

    ###########################################################################
    ###   X arm
    ###########################################################################
    s lx bs.p3 itmxar.p1 L=lmichx

    m itmxar T=1 L=0 phi=180
    s ar_thick itmxar.p2 itmx.p1 L=0
    m itmx T=itmT L=20u phi=180
    s LX itmx.p2 etmx.p1 L=Larm

    m etmx T=5u L=40u phi=179.99999

    # CHECK X-arm cavity power
    pd P_armX etmx.p1.i

    pendulum itmx_sus itmx.mech mass=Mtm fz=1 Qz=1M
    pendulum etmx_sus etmx.mech mass=Mtm fz=1 Qz=1M

    ###########################################################################
    ###   Y arm
    ###########################################################################
    s ly bs.p2 itmyar.p1 L=lmichy

    m itmyar T=1 L=0 phi=90
    s ar_thicky itmyar.p2 itmy.p1 L=0
    m itmy T=itmT L=20u phi=90
    s LY itmy.p2 etmy.p1 L=Larm

    m etmy T=5u L=40u phi=90.00001

    # CHECK Y-arm cavity power
    pd P_armY etmy.p1.i

    pendulum itmy_sus itmy.mech mass=Mtm fz=1 Qz=1M
    pendulum etmy_sus etmy.mech mass=Mtm fz=1 Qz=1M

    ###########################################################################
    ###   vSRM
    ###########################################################################

    #m srm T=0.2 L=37.5u phi=-90

    s src bs.p4 SRC_BS.p1 L={srcL}
    bs SRC_BS T=0.5 L=0 alpha=45
    s vSRC1 SRC_BS.p2 vSRM1.p1 L={vsrmL}
    m vSRM1 T={vsrmT} L=0 phi={-90 + comm + diff}
    s vSRC2  SRC_BS.p3 vSRM2.p1 L={vsrmL}
    m vSRM2 T={vsrmT} L=0 phi={0 + comm - diff}

    # CHECK SRC power
    pd P_SRC SRC_BS.p1.i

    ###########################################################################
    ###   Output & squeezing
    ###########################################################################
    dbs OFI 
    link(SRC_BS.p4, OFI.p1)
    readout_dc AS OFI.p3.o

    # A squeezed source could be injected into the dark port
    #sq sqz db=-10 angle=90
    #link(sqz, OFI.p2)

    # ------------------------------------------------------------------------------
    # Degrees of Freedom
    # ------------------------------------------------------------------------------
    dof STRAIN LX.dofs.h +1  LY.dofs.h -1

    # signal generator
    sgen sig STRAIN

    qnoised NSR_with_RP AS.p1.i nsr=True
    qshot NSR_without_RP AS.p1.i nsr=True
    pd1 signal AS.p1.i f=fsig

    fsig(1)
    xaxis(fsig, log, 1k, 10k, 150)

    """
    )
    out = kat.run()
    neg_sens = -np.abs(out['NSR_with_RP'])  # Take negative sensitivity to use findpeaks
    peak_idxs, _ = find_peaks(neg_sens)

    NSR_with_RP = np.abs(out['NSR_with_RP'])
    NSR_without_RP = np.abs(out['NSR_without_RP'])
    signal = np.abs(out['signal'])

    if peak_idxs.size != 0:  # If a peak is found (first peak taken)
        fwhm_idxs = peak_widths(neg_sens, peak_idxs, rel_height=0.5)
        left_idx = fwhm_idxs[2][0]
        right_idx = fwhm_idxs[3][0]
        interp_fsig = interpolate.interp1d(np.arange(151), fsig)

        left_f = interp_fsig(left_idx)
        right_f = interp_fsig(right_idx)
        peak_sens = NSR_with_RP[peak_idxs[0]]
        peak_f = fsig[peak_idxs[0]]
        peak_bw = right_f - left_f

    peak_dict = {"sens": peak_sens, "freq": peak_f, "fwhm": peak_bw, "lhm_f": left_f, "rhm_f": right_f}
    power_dict = {"armX": np.max(np.abs(out['P_armX'])),
                  "armY": np.max(np.abs(out['P_armY'])), "laser": np.max(np.abs(out['P_in'])),
                  "onBS": np.max(np.abs(out['P_BS'])), "PRC": np.max(np.abs(out['P_PRC'])),
                  "SRC": np.max(np.abs(out['P_SRC']))}
    curve_dict = {"NSR_with_RP": NSR_with_RP, "NSR_without_RP": NSR_without_RP,
                  "signal": signal}
    output = {"xaxis": fsig, "prmT": prmT, "peak_data": peak_dict, "power_data": power_dict, "curve_data": curve_dict}

    if plotting:
        plot_sensitivity(fsig, NSR_with_RP, NSR_without_RP, signal, peak_dict)
    if printing:
        print(f"Peak Sensitivity: {peak_sens} 1/rt Hz, Peak Frequency: {peak_f}Hz, Peak FWHM: {peak_bw}Hz")
        print(f"Input laser power: {np.max(np.abs(out['P_in']))}W")
        print(f"Laser power incident on BS: {np.max(np.abs(out['P_BS'])) * 1e-3}kW")
        print(f"PRC power: {np.max(np.abs(out['P_PRC'])) * 1e-3}kW")
        print(f"X-arm cavity power: {np.max(np.abs(out['P_armX'])) * 1e-6}MW")
        print(f"Y-arm cavity power: {np.max(np.abs(out['P_armY'])) * 1e-6}MW")
        print(f"SRC power: {np.max(np.abs(out['P_SRC']))}W")

    return output


# Auto-tune prmT to maximise arm cavity power (impedance matching)
def find_optimal_prmT(diff, comm, srcL, vsrmL, vsrmT, itmT):
    vary_prmT = np.linspace(0.01, 0.99, 99)
    circX = np.zeros((99,))

    for i, prmT in enumerate(vary_prmT):
        kat = finesse.Model()
        kat.parse(
            f"""
        ###########################################################################
        ###   Variables
        ###########################################################################
        var Larm 4000
        var Mtm  74.1
        var itmT {itmT}
        var lmichx 4.5
        var lmichy 4.45

        ###########################################################################
        ###   Input optics
        ###########################################################################
        l L0 500

        # CHECK Input laser power
        pd P_in L0.p1.o

        s l_in L0.p1 prm.p1
        # Power recycling mirror
        m prm T={prmT} L=2e-05 phi=90
        s prc prm.p2 bs.p1 L=53

        # Central beamsplitter
        bs bs T=0.5 L=0 alpha=45

        ###########################################################################
        ###   X arm
        ###########################################################################
        s lx bs.p3 itmxar.p1 L=lmichx

        m itmxar T=1 L=0 phi=180
        s ar_thick itmxar.p2 itmx.p1 L=0
        m itmx T=itmT L=20u phi=180
        s LX itmx.p2 etmx.p1 L=Larm

        m etmx T=5u L=40u phi=179.99999

        # CHECK X-arm cavity power
        pd P_armX etmx.p1.i

        pendulum itmx_sus itmx.mech mass=Mtm fz=1 Qz=1M
        pendulum etmx_sus etmx.mech mass=Mtm fz=1 Qz=1M

        ###########################################################################
        ###   Y arm
        ###########################################################################
        s ly bs.p2 itmyar.p1 L=lmichy

        m itmyar T=1 L=0 phi=90
        s ar_thicky itmyar.p2 itmy.p1 L=0
        m itmy T=itmT L=20u phi=90
        s LY itmy.p2 etmy.p1 L=Larm

        m etmy T=5u L=40u phi=90.00001

        pendulum itmy_sus itmy.mech mass=Mtm fz=1 Qz=1M
        pendulum etmy_sus etmy.mech mass=Mtm fz=1 Qz=1M

        ###########################################################################
        ###   vSRM
        ###########################################################################

        #m srm T=0.2 L=37.5u phi=-90

        s src bs.p4 SRC_BS.p1 L={srcL}
        bs SRC_BS T=0.5 L=0 alpha=45
        s vSRC1 SRC_BS.p2 vSRM1.p1 L={vsrmL}
        m vSRM1 T={vsrmT} L=0 phi={-90 + comm + diff}
        s vSRC2  SRC_BS.p3 vSRM2.p1 L={vsrmL}
        m vSRM2 T={vsrmT} L=0 phi={0 + comm - diff}

        noxaxis()

        """)
        out = kat.run()
        circX[i] = out['P_armX']

    peak_T = vary_prmT[np.argmax(circX)]
    return peak_T


def plot_sensitivity(fsig, NSR_with_RP, NSR_without_RP, signal, peak_dict):

    fig_qnoise = go.Figure()
    fig_qnoise.add_trace(go.Scatter(x=fsig, y=NSR_with_RP, mode='lines+markers', name='qnoised NSR'))
    fig_qnoise.add_trace(go.Scatter(x=fsig, y=NSR_without_RP, mode='lines+markers', name='qshot NSR'))
    fig_qnoise.update_xaxes(type="log")
    fig_qnoise.update_yaxes(type="log")
    fig_qnoise.add_vline(x=peak_dict['freq'])
    fig_qnoise.add_vline(x=peak_dict['rhm_f'], line_dash='dash', line_color='green')
    fig_qnoise.add_vline(x=peak_dict['lhm_f'], line_dash='dash', line_color='green')
    fig_qnoise.update_layout(title="Sensitivity (qnoised, qshot)", xaxis_title="Frequency [Hz]",
                             yaxis_title="Sensitivity [1/rt Hz]")
    fig_qnoise.show(renderer='browser')

    fig_signal = go.Figure()
    fig_signal.add_trace(go.Scatter(x=fsig, y=signal, mode='lines+markers'))
    fig_signal.update_xaxes(type="log")
    fig_signal.update_yaxes(type="log")
    fig_signal.add_vline(x=peak_dict['freq'])
    fig_signal.add_vline(x=peak_dict['rhm_f'], line_dash='dash', line_color='green')
    fig_signal.add_vline(x=peak_dict['lhm_f'], line_dash='dash', line_color='green')
    fig_signal.update_layout(title="Signal Gain (pd1)", xaxis_title="Frequency [Hz]", yaxis_title="Power [W]")
    fig_signal.show(renderer='browser')
