"""Reconstruction - The class and functions associated with reconstructing a \
particle from its observables."""


import numpy as np
import os
from abc import abstractmethod
from skhep.pdg import ParticleDataTable



### Data ###
PDT = ParticleDataTable(os.path.join(os.getcwd(),
                                     "Data","mass_width_2018.mcd"))


### Functions ###
def addReco(obj1,obj2):
    """Adds two Reco objects, returns a Reco object."""
    px = obj1.px + obj2.px
    py = obj1.py + obj2.py
    pz = obj1.pz + obj2.pz
    E = obj1.E + obj2.E
    return Reco(px,py,pz,E)
    
    
def stringReco(obj):
    """Converts a Reco object to a string."""
    name = obj.get_name()
    name = obj._pid if (name is None) else name
    return ("pdg: " + name + " E: " + str(obj._E)
            + " px: " + str(obj._px) + " py: " + str(obj._py)
            + " pz: "+ str(obj._pz) + " mass: " + str(obj._m))



### Classes ###
class Reco(object):
    def __init__(self,px,py,pz,E):

        self._px = px
        self._py = py
        self._pz = pz
        self._E = E
        self._m = np.sqrt(max(E*E - px*px - py*py - pz*pz, 0.0)) #Rest mass
        self._perp = np.sqrt(px*px + py*py)  # Transverse momentum
        self._pid = 0
        
        # Calculate Transverse Momentum Squared
        kt2=px*px + py*py
        self._kt2 = kt2  # perpendicular momentum squared

        maxrap = 1e5

        # Calculate Phi Transverse Momentum
        if kt2 == 0.0:
            phi = 0.0
        else:
            phi = np.arctan2(py,px)
        
        # Ensure 0 <= Phi <= 2*Pi
        if phi < 0.0:
            phi += 2*np.pi
        if phi >= 2*np.pi:
            phi -= 2*np.pi
        
        # Calculate Rapidity
        if E == abs(pz) and kt2 == 0:
            maxraphere = maxrap + abs(pz)
            if pz >= 0.0: 
                self._rap = maxraphere
            else: 
                self._rap = -maxraphere
        else:
            effective_m2 = max(0.0,self._m * self._m)
            E_plus_pz = E + abs(pz)
            self._rap = 0.5*np.log((kt2 + effective_m2)
                                   /(E_plus_pz*E_plus_pz))
            if pz > 0:
                self._rap = -self._rap

    def __str__(self):
        return stringReco(self)

    def __repr__(self):
        attributes = ["pdg: {pid} [{name}]".format(pid=self._pid,
                                                   name=self.get_name()),
                      "E: {E}".format(E=self._E),
                      "px: {px}".format(px=self._px),
                      "py: {py}".format(py=self._py),
                      "pz: {pz}".format(pz=self._pz),
                      "mass: {m}".format(m=self._m)]
        return "Reco(" + "; ".join(attributes) +")"

    def get_name(self, no_rtn=None):
        pid = self.pid
        
        if pid == 21:
            return 'g'
        elif pid == 23:
            return 'Z'
        elif pid == 24:
            return 'W+'
        elif pid == -24:
            return 'W-'
        elif pid == 25:
            return 'h'
        
        pid = pid if pid > 0 else -1 * pid
        name = PDT[pid].name if PDT.has_key(pid) else str(None)
        if 1 <= pid <= 6:  # Quarks
            if self._pid > 0:
                return name.replace("bar", '')
            else:
                return name
        elif 11 <= pid <= 16: # Leptons
            if self._pid < 0:
                return name.replace('-', '+')\
                           .replace('_', "bar_")
            else:
                return name
        else:
            return str(no_rtn)


    def setpx(self,px):
        self._px = px

    def setpy(self,py):
        self._py = py

    def setpz(self,pz):
        self._pz = pz

    def setE(self,E):
        self._E = E

    def setpid(self,pid):
        self._pid = pid

    def setpT(self,phat_beam):
        p = self.p
        p -= np.dot(p, phat_beam) * phat_beam
        self._pT = p
        return self.pT


    def is_quark(self):
        return 1 <= abs(self.pid) <= 6

    def is_down_type_quark(self):
        return abs(self.pid) in [1, 3, 5]

    def is_up_type_quark(self):
        return abs(self.pid) in [2, 4, 6]

    def is_lepton(self):
        return 11 <= abs(self.pid) <= 16

    def is_charged_lepton(self):
        return abs(self.pid) in [11, 13, 15]

    def is_neutrino(self):
        return abs(self.pid) in [12, 14, 16]


    @property
    def px(self): return self._px
    @property
    def py(self): return self._py
    @property
    def pz(self): return self._pz
    @property
    def E(self): return self._E
    @property
    def pid(self): return self._pid
    @property
    def m(self): return self._m
    @property
    def perp(self): return self._perp
    @property
    def rap(self): return self._rap
    @property
    def p(self): return np.array([self.px, self.py, self.pz])
    @property
    def mag_p(self): return np.linalg.norm(self.p)
    @property
    def pT(self):
        try:
            return self._pT
        except:
            print("You must execute the method 'setpT' first")
            return None


    @abstractmethod
    def get_init_keys():
        return ['px', 'py', 'pz', 'e']
    
    @classmethod
    def from_dict(cls, particle_dict):
        keys = cls.get_init_keys()
        
        observables = [particle_dict[key] for key in keys]
        temp = cls(*observables)
        temp.setpid(particle_dict['id'])
        
        return temp
    
    @classmethod
    def empty(cls):
        keys = cls.get_init_keys()
        
        empty_observables = [0.0 for key in keys]
        temp = cls(*empty_observables)
        temp.setpid(0)
        
        return temp