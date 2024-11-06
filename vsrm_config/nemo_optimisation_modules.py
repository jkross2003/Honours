import finesse
import numpy as np
from scipy.signal import find_peaks, peak_widths
from scipy import interpolate
import plotly.graph_objects as go


def find_optimal_prmT(phiComm,phiDiff,srcL,itmT):
    # Varies prmT and maximises arm cavity power (impedance matching condition)
    # TODO: Increase resolution of prmT interval (lower limit is still quite high)
    vary_prmT = np.geomspace(0.01,0.5,100)
    circX = np.zeros((100,))
    # Find prmT to maximise arm cavity power
    for i, prmT in enumerate(vary_prmT):
        kat = finesse.Model()
        kat.parse(
        f"""
        # NEMO Base Model (simplified from OzHF_ITMloss_4km.kat and converted to Finesse 3)
        ###########################################################################
        ###   Variables
        ###########################################################################
        var Larm 4000
        var Mtm  74.1 # from NEMO paper (94.4 in OzHF_ITMloss_4km.kat)
        var itmT {itmT}
        var lmichx 4.5
        var lmichy 4.45

        ###########################################################################
        ###   Input optics
        ###########################################################################
        l L0 500

        s l_in L0.p1 prm.p1
        # Power recycling mirror
        m prm T={prmT} L=2e-05 phi=90
        s prc prm.p2 bs.p1 L=53

        # Central beamsplitter
        bs bs R=0.4999625 T=0.4999625 alpha=45

        # CHECK Input laser power
        # pd P_in L0.p1.o
        # CHECK Laser power incident on BS
        # pd P_BS bs.p1.i
        # CHECK PRC Power
        # pd P_PRC bs.p1.o

        ###########################################################################
        ###   X arm
        ###########################################################################
        s lx bs.p3 itmxar.p1 L=lmichx

        m itmxar T=1-265.0e-06 L=265.0e-06 phi=180 # phi from SRC_tunability_DCreadout_changedBW_fin3exp.ipynb (0.0 in OzHF_ITMloss_4km.kat)
        s ar_thick itmxar.p2 itmx.p1 L=0
        m itmx T=itmT L=20u phi=180
        s LX itmx.p2 etmx.p1 L=Larm

        m etmx T=5u L=20u phi=179.99999

        pendulum itmx_sus itmx.mech mass=Mtm fz=1 Qz=1M
        pendulum etmx_sus etmx.mech mass=Mtm fz=1 Qz=1M

        # CHECK X-arm cavity power
        pd P_armX etmx.p1.i

        ###########################################################################
        ###   Y arm
        ###########################################################################
        s ly bs.p2 itmyar.p1 L=lmichy

        m itmyar T=1-265.0e-06 L=265.0e-06 phi=90 # phi from SRC_tunability_DCreadout_changedBW_fin3exp.ipynb (0.0 in OzHF_ITMloss_4km.kat)
        s ar_thicky itmyar.p2 itmy.p1 L=0
        m itmy T=itmT L=20u phi=90
        s LY itmy.p2 etmy.p1 L=Larm

        m etmy T=5u L=20u phi=90.00001

        pendulum itmy_sus itmy.mech mass=Mtm fz=1 Qz=1M
        pendulum etmy_sus etmy.mech mass=Mtm fz=1 Qz=1M

        # CHECK Y-arm cavity power
        # pd P_armY etmy.p1.i

        ###########################################################################
        ###   vSRM
        ###########################################################################
        s src bs.p4 SRC_BS.p1 L={srcL}
        bs SRC_BS T=0.5 L=0 alpha=45
        s vSRC1 SRC_BS.p2 vSRM1.p1 L=4.5
        m vSRM1 T=0 L=0 phi={-90+phiComm+phiDiff}
        s vSRC2 SRC_BS.p3 vSRM2.p1 L=4.5
        m vSRM2 T=0 L=0 phi={0+phiComm-phiDiff}

        # CHECK SRC power
        # pd P_SRC SRC_BS.p1.i

        ###########################################################################
        ###   Output & squeezing (from SRC_tunability_DCreadout_changedBW_fin3exp.ipynb)
        ###########################################################################
        noxaxis()
        """
        )
        out = kat.run()
        circX[i] = out['P_armX']
    
    peakPow = circX[np.argmax(circX)]
    prmT = vary_prmT[np.argmax(circX)]
    return peakPow, prmT

def adjust_lasPow(phiComm,phiDiff,srcL,itmT,peakPow,prmT,start_lasPow=500,armPow=4.5e6):
    # After the arm cavity power is maximised, the input power is varied in steps of 1W to match to nominal value of 4.5MW (maximum value is 550W)
    # TODO: Add extra physical constraints e.g. limit power on beamsplitter
    lasPow = start_lasPow

    if peakPow < armPow:
        while peakPow < armPow and lasPow < start_lasPow*1.10:
            lasPow += 1
            kat = finesse.Model()
            kat.parse(
            f"""
            # NEMO Base Model (simplified from OzHF_ITMloss_4km.kat and converted to Finesse 3)
            ###########################################################################
            ###   Variables
            ###########################################################################
            var Larm 4000
            var Mtm  74.1 # from NEMO paper (94.4 in OzHF_ITMloss_4km.kat)
            var itmT {itmT}
            var lmichx 4.5
            var lmichy 4.45

            ###########################################################################
            ###   Input optics
            ###########################################################################
            l L0 {lasPow}

            s l_in L0.p1 prm.p1
            # Power recycling mirror
            m prm T={prmT} L=2e-05 phi=90
            s prc prm.p2 bs.p1 L=53

            # Central beamsplitter
            bs bs R=0.4999625 T=0.4999625 alpha=45

            # CHECK Input laser power
            # pd P_in L0.p1.o
            # CHECK Laser power incident on BS
            # pd P_BS bs.p1.i
            # CHECK PRC Power
            # pd P_PRC bs.p1.o

            ###########################################################################
            ###   X arm
            ###########################################################################
            s lx bs.p3 itmxar.p1 L=lmichx

            m itmxar T=1-265.0e-06 L=265.0e-06 phi=180 # phi from SRC_tunability_DCreadout_changedBW_fin3exp.ipynb (0.0 in OzHF_ITMloss_4km.kat)
            s ar_thick itmxar.p2 itmx.p1 L=0
            m itmx T=itmT L=20u phi=180
            s LX itmx.p2 etmx.p1 L=Larm

            m etmx T=5u L=20u phi=179.99999

            pendulum itmx_sus itmx.mech mass=Mtm fz=1 Qz=1M
            pendulum etmx_sus etmx.mech mass=Mtm fz=1 Qz=1M

            # CHECK X-arm cavity power
            pd P_armX etmx.p1.i

            ###########################################################################
            ###   Y arm
            ###########################################################################
            s ly bs.p2 itmyar.p1 L=lmichy

            m itmyar T=1-265.0e-06 L=265.0e-06 phi=90 # phi from SRC_tunability_DCreadout_changedBW_fin3exp.ipynb (0.0 in OzHF_ITMloss_4km.kat)
            s ar_thicky itmyar.p2 itmy.p1 L=0
            m itmy T=itmT L=20u phi=90
            s LY itmy.p2 etmy.p1 L=Larm

            m etmy T=5u L=20u phi=90.00001

            pendulum itmy_sus itmy.mech mass=Mtm fz=1 Qz=1M
            pendulum etmy_sus etmy.mech mass=Mtm fz=1 Qz=1M

            # CHECK Y-arm cavity power
            # pd P_armY etmy.p1.i

            ###########################################################################
            ###   vSRM
            ###########################################################################
            s src bs.p4 SRC_BS.p1 L={srcL}
            bs SRC_BS T=0.5 L=0 alpha=45
            s vSRC1 SRC_BS.p2 vSRM1.p1 L=4.5
            m vSRM1 T=0 L=0 phi={-90+phiComm+phiDiff}
            s vSRC2 SRC_BS.p3 vSRM2.p1 L=4.5
            m vSRM2 T=0 L=0 phi={0+phiComm-phiDiff}

            # CHECK SRC power
            # pd P_SRC SRC_BS.p1.i

            ###########################################################################
            ###   Output & squeezing (from SRC_tunability_DCreadout_changedBW_fin3exp.ipynb)
            ###########################################################################
            noxaxis()
            """
            )
            out = kat.run()
            peakPow = out['P_armX']
    else: 
        while peakPow > armPow:
            lasPow -= 1
            kat = finesse.Model()
            kat.parse(
            f"""
            # NEMO Base Model (simplified from OzHF_ITMloss_4km.kat and converted to Finesse 3)
            ###########################################################################
            ###   Variables
            ###########################################################################
            var Larm 4000
            var Mtm  74.1 # from NEMO paper (94.4 in OzHF_ITMloss_4km.kat)
            var itmT {itmT}
            var lmichx 4.5
            var lmichy 4.45

            ###########################################################################
            ###   Input optics
            ###########################################################################
            l L0 {lasPow}

            s l_in L0.p1 prm.p1
            # Power recycling mirror
            m prm T={prmT} L=2e-05 phi=90
            s prc prm.p2 bs.p1 L=53

            # Central beamsplitter
            bs bs R=0.4999625 T=0.4999625 alpha=45

            # CHECK Input laser power
            # pd P_in L0.p1.o
            # CHECK Laser power incident on BS
            # pd P_BS bs.p1.i
            # CHECK PRC Power
            # pd P_PRC bs.p1.o

            ###########################################################################
            ###   X arm
            ###########################################################################
            s lx bs.p3 itmxar.p1 L=lmichx

            m itmxar T=1-265.0e-06 L=265.0e-06 phi=180 # phi from SRC_tunability_DCreadout_changedBW_fin3exp.ipynb (0.0 in OzHF_ITMloss_4km.kat)
            s ar_thick itmxar.p2 itmx.p1 L=0
            m itmx T=itmT L=20u phi=180
            s LX itmx.p2 etmx.p1 L=Larm

            m etmx T=5u L=20u phi=179.99999

            pendulum itmx_sus itmx.mech mass=Mtm fz=1 Qz=1M
            pendulum etmx_sus etmx.mech mass=Mtm fz=1 Qz=1M

            # CHECK X-arm cavity power
            pd P_armX etmx.p1.i

            ###########################################################################
            ###   Y arm
            ###########################################################################
            s ly bs.p2 itmyar.p1 L=lmichy

            m itmyar T=1-265.0e-06 L=265.0e-06 phi=90 # phi from SRC_tunability_DCreadout_changedBW_fin3exp.ipynb (0.0 in OzHF_ITMloss_4km.kat)
            s ar_thicky itmyar.p2 itmy.p1 L=0
            m itmy T=itmT L=20u phi=90
            s LY itmy.p2 etmy.p1 L=Larm

            m etmy T=5u L=20u phi=90.00001

            pendulum itmy_sus itmy.mech mass=Mtm fz=1 Qz=1M
            pendulum etmy_sus etmy.mech mass=Mtm fz=1 Qz=1M

            # CHECK Y-arm cavity power
            # pd P_armY etmy.p1.i

            ###########################################################################
            ###   vSRM
            ###########################################################################
            s src bs.p4 SRC_BS.p1 L={srcL}
            bs SRC_BS T=0.5 L=0 alpha=45
            s vSRC1 SRC_BS.p2 vSRM1.p1 L=4.5
            m vSRM1 T=0 L=0 phi={-90+phiComm+phiDiff}
            s vSRC2 SRC_BS.p3 vSRM2.p1 L=4.5
            m vSRM2 T=0 L=0 phi={0+phiComm-phiDiff}

            # CHECK SRC power
            # pd P_SRC SRC_BS.p1.i

            ###########################################################################
            ###   Output & squeezing (from SRC_tunability_DCreadout_changedBW_fin3exp.ipynb)
            ###########################################################################
            noxaxis()
            """
            )
            out = kat.run()
            peakPow = out['P_armX']
        lasPow += 1
    
    return lasPow

def Finesse_sensitivity_into_Bilby(phiComm,phiDiff,srcL,itmT,prmT=0,lasPow=0,optimise_prmT=True,export_peaks=False,squeezing=True):  # Only if itmT changes does prmT need to be recalculated
    # Calculates ASD ARRAY (quantum shot and radiation pressure noise) given parameter values (after optional impedance-matching and fixing arm cavity power to 4.5MW)
    # Squeezing is optional and exporting information about the lowest-frequency sensitivity peak (dictionary) is optional
    # ASD curve can be used for plotting or into Bilby
    fsig = np.geomspace(100,10e3,201)
    peak_sens = 0
    peak_f = 0
    peak_bw = 0
    left_f = 0
    right_f = 0
    
    phiComm = float(phiComm)
    phiDiff = float(phiDiff)
    srcL = float(srcL)
    itmT = float(itmT)
    if not squeezing:
        add_squeezing = '#'
    else:
        add_squeezing = ''
    
    # Maximise arm
    if optimise_prmT:
        peakPow, prmT = find_optimal_prmT(phiComm,phiDiff,srcL,itmT)
        lasPow = adjust_lasPow(phiComm,phiDiff,srcL,itmT,peakPow,prmT)
    else:
        prmT = float(prmT)
        lasPow = float(lasPow)
    
    kat = finesse.Model()
    # From SRC_tunability_DCreadout_changedBW_fin3exp.ipynb
    kat.parse(
    f"""
    # NEMO Base Model (simplified from OzHF_ITMloss_4km.kat and converted to Finesse 3)
    ###########################################################################
    ###   Variables
    ###########################################################################
    var Larm 4000
    var Mtm  74.1 # from NEMO paper (94.4 in OzHF_ITMloss_4km.kat)
    var itmT {itmT}
    var lmichx 4.5
    var lmichy 4.45

    ###########################################################################
    ###   Input optics
    ###########################################################################
    l L0 {lasPow}

    s l_in L0.p1 prm.p1
    # Power recycling mirror
    m prm T={prmT} L=2e-05 phi=90
    s prc prm.p2 bs.p1 L=53

    # Central beamsplitter
    bs bs R=0.4999625 T=0.4999625 alpha=45

    # CHECK Input laser power
    # pd P_in L0.p1.o
    # CHECK Laser power incident on BS
    # pd P_BS bs.p1.i
    # CHECK PRC Power
    # pd P_PRC bs.p1.o

    ###########################################################################
    ###   X arm
    ###########################################################################
    s lx bs.p3 itmxar.p1 L=lmichx

    m itmxar T=1-265.0e-06 L=265.0e-06 phi=180 # phi from SRC_tunability_DCreadout_changedBW_fin3exp.ipynb (0.0 in OzHF_ITMloss_4km.kat)
    s ar_thick itmxar.p2 itmx.p1 L=0
    m itmx T=itmT L=20u phi=180
    s LX itmx.p2 etmx.p1 L=Larm

    m etmx T=5u L=20u phi=179.99999

    pendulum itmx_sus itmx.mech mass=Mtm fz=1 Qz=1M
    pendulum etmx_sus etmx.mech mass=Mtm fz=1 Qz=1M

    # CHECK X-arm cavity power
    # pd P_armX etmx.p1.i

    ###########################################################################
    ###   Y arm
    ###########################################################################
    s ly bs.p2 itmyar.p1 L=lmichy

    m itmyar T=1-265.0e-06 L=265.0e-06 phi=90 # phi from SRC_tunability_DCreadout_changedBW_fin3exp.ipynb (0.0 in OzHF_ITMloss_4km.kat)
    s ar_thicky itmyar.p2 itmy.p1 L=0
    m itmy T=itmT L=20u phi=90
    s LY itmy.p2 etmy.p1 L=Larm

    m etmy T=5u L=20u phi=90.00001

    pendulum itmy_sus itmy.mech mass=Mtm fz=1 Qz=1M
    pendulum etmy_sus etmy.mech mass=Mtm fz=1 Qz=1M

    # CHECK Y-arm cavity power
    # pd P_armY etmy.p1.i

    ###########################################################################
    ###   vSRM
    ###########################################################################
    s src bs.p4 SRC_BS.p1 L={srcL}
    bs SRC_BS T=0.5 L=0 alpha=45
    s vSRC1 SRC_BS.p2 vSRM1.p1 L=4.5
    m vSRM1 T=0 L=0 phi={-90+phiComm+phiDiff}
    s vSRC2 SRC_BS.p3 vSRM2.p1 L=4.5
    m vSRM2 T=0 L=0 phi={0+phiComm-phiDiff}

    # CHECK SRC power
    # pd P_SRC SRC_BS.p1.i

    ###########################################################################
    ###   Output & squeezing (from SRC_tunability_DCreadout_changedBW_fin3exp.ipynb)
    ###########################################################################
    dbs OFI 
    link(SRC_BS.p4, OFI.p1)
    readout_dc AS OFI.p3.o

    # A squeezed source could be injected into the dark port
    {add_squeezing}sq sqz db=-7 angle=90
    {add_squeezing}link(sqz, OFI.p2)

    # ------------------------------------------------------------------------------
    # Degrees of Freedom
    # ------------------------------------------------------------------------------
    dof STRAIN LX.dofs.h +1  LY.dofs.h -1

    # signal generator
    sgen sig STRAIN

    qnoised NSR_with_RP AS.p1.i nsr=True
    # qshot NSR_without_RP AS.p1.i nsr=True
    # pd1 signal AS.p1.i f=fsig

    fsig(1)
    xaxis(fsig, log, 100, 10k, 200)
    """
    )
    out = kat.run()
    ASDarr = np.abs(out['NSR_with_RP'])  # Ready for bilby

    if not export_peaks:
        return fsig, ASDarr, prmT, lasPow
    else:
        neg_sens = -ASDarr # Take reciprocal sensitivity to use findpeaks i.e. FWHM defined by using reciprocal sensitivity!
        peak_idxs, _ = find_peaks(neg_sens)
        if peak_idxs.size != 0: # If a peak is found (first peak taken)
            fwhm_idxs = peak_widths(neg_sens, peak_idxs, rel_height=0.5)
            left_idx = fwhm_idxs[2][0]
            right_idx = fwhm_idxs[3][0]
            interp_fsig = interpolate.interp1d(np.arange(201), fsig)
            
            left_f = interp_fsig(left_idx)
            right_f = interp_fsig(right_idx)
            peak_sens = ASDarr[peak_idxs[0]]
            peak_f = fsig[peak_idxs[0]]
            peak_bw = right_f - left_f
            peak_dict = {'peak_sens': peak_sens, 'peak_f': peak_f, 'peak_bw': peak_bw}
        else:
            peak_dict = {'peak_sens': np.nan, 'peak_f': np.nan, 'peak_bw': np.nan}
        return fsig, ASDarr, prmT, lasPow, peak_dict

def Finesse_sensitivity_into_txt(params_idx,save_path,phiComm,phiDiff,srcL,itmT,prmT=0,lasPow=0,optimise_prmT=True,squeezing=True):  # Only if itmT changes does prmT need to be recalculated
    # Calculates ASD TXT FILE (quantum shot and radiation pressure noise) given parameter values (after optional impedance-matching and fixing arm cavity power to 4.5MW)
    # Squeezing is optional and exporting information about the lowest-frequency sensitivity peak (dictionary) is optional
    # ASD curve can be used for plotting or into Bilby
    fsig = np.geomspace(100,10e3,201)
    peak_sens = 0
    peak_f = 0
    peak_bw = 0
    left_f = 0
    right_f = 0
    
    phiComm = float(phiComm)
    phiDiff = float(phiDiff)
    srcL = float(srcL)
    itmT = float(itmT)
    if not squeezing:
        add_squeezing = '#'
    else:
        add_squeezing = ''
    
    # Maximise arm
    if optimise_prmT:
        peakPow, prmT = find_optimal_prmT(phiComm,phiDiff,srcL,itmT)
        lasPow = adjust_lasPow(phiComm,phiDiff,srcL,itmT,peakPow,prmT)
    else:
        prmT = float(prmT)
        lasPow = float(lasPow)
    
    kat = finesse.Model()
    # From SRC_tunability_DCreadout_changedBW_fin3exp.ipynb
    kat.parse(
    f"""
    # NEMO Base Model (simplified from OzHF_ITMloss_4km.kat and converted to Finesse 3)
    ###########################################################################
    ###   Variables
    ###########################################################################
    var Larm 4000
    var Mtm  74.1 # from NEMO paper (94.4 in OzHF_ITMloss_4km.kat)
    var itmT {itmT}
    var lmichx 4.5
    var lmichy 4.45

    ###########################################################################
    ###   Input optics
    ###########################################################################
    l L0 {lasPow}

    s l_in L0.p1 prm.p1
    # Power recycling mirror
    m prm T={prmT} L=2e-05 phi=90
    s prc prm.p2 bs.p1 L=53

    # Central beamsplitter
    bs bs R=0.4999625 T=0.4999625 alpha=45

    # CHECK Input laser power
    # pd P_in L0.p1.o
    # CHECK Laser power incident on BS
    # pd P_BS bs.p1.i
    # CHECK PRC Power
    # pd P_PRC bs.p1.o

    ###########################################################################
    ###   X arm
    ###########################################################################
    s lx bs.p3 itmxar.p1 L=lmichx

    m itmxar T=1-265.0e-06 L=265.0e-06 phi=180 # phi from SRC_tunability_DCreadout_changedBW_fin3exp.ipynb (0.0 in OzHF_ITMloss_4km.kat)
    s ar_thick itmxar.p2 itmx.p1 L=0
    m itmx T=itmT L=20u phi=180
    s LX itmx.p2 etmx.p1 L=Larm

    m etmx T=5u L=20u phi=179.99999

    pendulum itmx_sus itmx.mech mass=Mtm fz=1 Qz=1M
    pendulum etmx_sus etmx.mech mass=Mtm fz=1 Qz=1M

    # CHECK X-arm cavity power
    # pd P_armX etmx.p1.i

    ###########################################################################
    ###   Y arm
    ###########################################################################
    s ly bs.p2 itmyar.p1 L=lmichy

    m itmyar T=1-265.0e-06 L=265.0e-06 phi=90 # phi from SRC_tunability_DCreadout_changedBW_fin3exp.ipynb (0.0 in OzHF_ITMloss_4km.kat)
    s ar_thicky itmyar.p2 itmy.p1 L=0
    m itmy T=itmT L=20u phi=90
    s LY itmy.p2 etmy.p1 L=Larm

    m etmy T=5u L=20u phi=90.00001

    pendulum itmy_sus itmy.mech mass=Mtm fz=1 Qz=1M
    pendulum etmy_sus etmy.mech mass=Mtm fz=1 Qz=1M

    # CHECK Y-arm cavity power
    # pd P_armY etmy.p1.i

    ###########################################################################
    ###   vSRM
    ###########################################################################
    s src bs.p4 SRC_BS.p1 L={srcL}
    bs SRC_BS T=0.5 L=0 alpha=45
    s vSRC1 SRC_BS.p2 vSRM1.p1 L=4.5
    m vSRM1 T=0 L=0 phi={-90+phiComm+phiDiff}
    s vSRC2 SRC_BS.p3 vSRM2.p1 L=4.5
    m vSRM2 T=0 L=0 phi={0+phiComm-phiDiff}

    # CHECK SRC power
    # pd P_SRC SRC_BS.p1.i

    ###########################################################################
    ###   Output & squeezing (from SRC_tunability_DCreadout_changedBW_fin3exp.ipynb)
    ###########################################################################
    dbs OFI 
    link(SRC_BS.p4, OFI.p1)
    readout_dc AS OFI.p3.o

    # A squeezed source could be injected into the dark port
    {add_squeezing}sq sqz db=-7 angle=90
    {add_squeezing}link(sqz, OFI.p2)

    # ------------------------------------------------------------------------------
    # Degrees of Freedom
    # ------------------------------------------------------------------------------
    dof STRAIN LX.dofs.h +1  LY.dofs.h -1

    # signal generator
    sgen sig STRAIN

    qnoised NSR_with_RP AS.p1.i nsr=True
    # qshot NSR_without_RP AS.p1.i nsr=True
    # pd1 signal AS.p1.i f=fsig

    fsig(1)
    xaxis(fsig, log, 100, 10k, 200)
    """
    )
    out = kat.run()
    ASDarr = np.abs(out['NSR_with_RP'])  # Ready for bilby
    lines = [f"{ASDarr[i]}\n" for i in range(201)]
    lines[-1].rstrip('\n')
    filename = f"vSRM_{params_idx[0]},{params_idx[1]},{params_idx[2]},{params_idx[3]}_{phiComm}_{phiDiff}_{srcL}_{itmT}_{prmT}_{lasPow}_ASD_with_RP.txt"
    file = open(save_path+filename, "w")
    file.writelines(lines)
    file.close()

    return prmT, lasPow