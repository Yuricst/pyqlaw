"""Object containing perturbation computation scheme"""

from numba import njit
import numpy as np
import spiceypy as spice

from ._elements import kep2sv, mee_with_a2sv


@njit
def perturbation_third_body_battin(r, s, mu_third):
    """Third body perturbation acceleration via Battin's formulation
    
    Args:
        r (np.array): position vector of spacecraft w.r.t. primary body
        s (np.array): position vector of third body w.r.t. primary body
        mu_third (float): GM of third body

    Returns:
        (np.array): third-body perturbation acceleration
    """
    d = r - s
    dnorm = np.sqrt(np.dot(d,d))
    q = np.dot(r, r - 2*s)/np.dot(s,s)
    F = q*((3 + 3*q + q**2)/(1 + np.sqrt(1+q)**3))
    return -mu_third/dnorm**3 * (r + F*s)


@njit
def perturbation_J2(rECI, T_ECI2PA, mu, J2, Re):
    """Compute J2 perturbation in perifocal frame and return vector in ECI frame
    
    Args:
        rECI (np.array): position vector in ECI frame
        T_ECI2PA (np.array): transformation matrix from ECI to principal axes frame
        mu (float): GM of central body
        J2 (float): J2 coefficient of central body
        Re (float): equatorial radius of central body

    Returns:
        (np.array): J2 perturbation acceleration in ECI frame
    """
    r = np.linalg.norm(rECI)
    coeff = -3*mu*J2*Re**2/(2*r**5)
    rPA = T_ECI2PA @ rECI
    a_J2_PA = coeff * np.array([
        (1 - 5*rPA[2]**2) * rPA[0],
        (1 - 5*rPA[2]**2) * rPA[1],
        (3 - 5*rPA[2]**2) * rPA[2]
    ])
    return T_ECI2PA.T @ a_J2_PA


def perturbation_SRP(r_sun2sc, AU, k_srp):
    """Compute solar radiation pressure (SRP) perturbation
    
    Args:
        r_sun2sc (np.array): position vector of spacecraft w.r.t. Sun, in LU
        AU (float): Astronomical unit, in LU
        k_srp (float): acceleration magnitude, LU/TU^2

    Returns:
        (np.array): SRP acceleration
    """
    return k_srp * AU**2  * r_sun2sc / np.linalg.norm(r_sun2sc)**3


@njit
def pxformECI2RTN(rECI, vECI):
    """Compute 3-by-3 transformation matrix from ECI to RTN frame
    
    Args:
        rECI (np.array): position vector in ECI frame
        vECI (np.array): velocity vector in ECI frame

    Returns:
        (np.array): 3-by-3 transformation matrix from ECI to RTN frame
    """
    er = rECI / np.linalg.norm(rECI)
    h = np.cross(rECI, vECI)
    en = h / np.linalg.norm(h)
    et = np.cross(en, er)
    
    ECI2RTN = np.zeros((3,3))
    ECI2RTN[0,:] = er
    ECI2RTN[1,:] = et
    ECI2RTN[2,:] = en
    return ECI2RTN


class SpicePerturbations:
    """Class contains parameters for computing perturbations in RTN frame.
    SPICE is used to query positions of third bodies and transformation matrices.
    Default inputs are for an Earth-centered orbit.
    
    Args:
        et_ref (float): reference time in ET, i.e. ephemeris time corresponding to `t = 0`
        LU (float): length unit in km
        TU (float): time unit in seconds
        obs_id (str): SPICE ID of observer body, i.e. central body
        frame_qlaw (str): SPICE frame in which Q-law is defined
        frame_PA (str): SPICE frame corresponding to principal axes frame of central body
        third_bodies_names (list): list of SPICE IDs of third bodies
        third_bodies_gms (list): list of GM's of third bodies
        J2 (float): J2 coefficient of central body
        Re_km (float): equatorial radius of central body in km
        use_J2 (bool): flag to use J2 perturbation
    """
    def __init__(
        self,
        et_ref,
        LU,
        TU,
        obs_id = "399",
        frame_qlaw = "J2000",
        frame_PA = "ITRF93",
        third_bodies_names = ["301", "10"], 
        third_bodies_gms = None,
        J2 = 0.00108263,
        Re_km = 6378.0,
        P_SRP_SI = 4.56e-6,
        Cr_SRP = 1.2,
        Am_SI = 0.01,
        use_J2 = True,
        use_SRP = True,
    ):
        # store parameters
        self.et_ref = et_ref
        self.LU = LU
        self.TU = TU
        self.obs_id = obs_id
        self.frame_qlaw = frame_qlaw
        self.frame_PA = frame_PA
        self.third_bodies_names = third_bodies_names
        self.J2 = J2
        self.Re = Re_km / LU
        self.k_srp = P_SRP_SI * Cr_SRP * Am_SI / (1e3 * LU/TU**2)

        # flags for turning perturbation terms on/off
        self.use_J2 = use_J2
        self.use_SRP = use_SRP

        # get GM's if not provided
        VU = LU / TU
        MU_REF = LU * VU**2
        self.obs_mu = spice.bodvrd(obs_id, "GM", 1)[1][0] / MU_REF
        if (third_bodies_gms is None) and (third_bodies_names is not None):
            self.third_bodies_gms = [spice.bodvrd(ID, "GM", 1)[1][0]/MU_REF for ID in third_bodies_names]
        else:
            self.third_bodies_gms = third_bodies_gms
        return
    
    def get_perturbations_RTN(self, t, oe, elements_type):
        """Compute perturbation in RTN frame
        
        Args:
            t (float): time in TU
            oe (np.array): osculating elements
            elements_type (str): type of osculating elements

        Returns:
            (np.array): perturbation acceleration in RTN frame
        """
        # get inertial position & velocity
        if elements_type == "keplerian":
            rv_ECI = kep2sv(oe, self.obs_mu)
        elif elements_type == "mee_with_a":
            rv_ECI = mee_with_a2sv(oe, self.obs_mu)
        else:
            raise ValueError(f"Invalid elements type {elements_type}")

        # first compute perturbations in ECI
        acc_ptrb_ECI = np.zeros(3,)

        # third-body perturbation in ECI
        et = self.et_ref + t * self.TU     # time to query spice ephemeris
        for mu3,ID in zip(self.third_bodies_gms, self.third_bodies_names):
            pos3_km,_ = spice.spkpos(ID, et, self.frame_qlaw, "NONE", self.obs_id)
            acc_ptrb_ECI += perturbation_third_body_battin(rv_ECI[0:3], pos3_km/self.LU, mu3)

        # J2 perturbation in ECI
        if self.use_J2:
            T_ECI2PA = spice.pxform(self.frame_qlaw, self.frame_PA, et)
            acc_ptrb_ECI += perturbation_J2(rv_ECI[0:3], T_ECI2PA, self.obs_mu, self.J2, self.Re)

        # SRP perturbation in ECI
        if self.use_SRP:
            pos_sun_km,_ = spice.spkpos("10", et, self.frame_qlaw, "NONE", self.obs_id)
            r_sun2sc = rv_ECI[0:3] - pos_sun_km/self.LU
            acc_ptrb_ECI += perturbation_SRP(r_sun2sc, spice.convrt(1.0, "AU", "km")/self.LU, self.k_srp)

        # transform perturbations to RTN
        T_ECI2RTN = pxformECI2RTN(rv_ECI[0:3], rv_ECI[3:6])
        acc_ptrb_RTN = T_ECI2RTN @ acc_ptrb_ECI
        return acc_ptrb_RTN