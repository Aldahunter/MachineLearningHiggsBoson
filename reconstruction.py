### Imports ###
import numpy as np
import os
from skhep.pdg import ParticleDataTable



### Variables ###
PDT = ParticleDataTable(os.path.join(os.getcwd(),
                                     "Data","mass_width_2018.mcd"))



### Functions ###
def addReco(obj1,obj2):
    px = obj1.px + obj2.px
    py = obj1.py + obj2.py
    pz = obj1.pz + obj2.pz
    E = obj1.E + obj2.E
    return Reco(px,py,pz,E)
    
    
def stringReco(obj1):
    name = obj1.get_name()
    name = obj1._pid if (name is None) else name
    return ("pdg: " + name + " E: " + str(obj1._E)
            + " px: " + str(obj1._px) + " py: " + str(obj1._py)
            + " pz: "+ str(obj1._pz) + " mass: " + str(obj1._m))



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
            self._rap = 0.5*np.log((kt2 + effective_m2)/(E_plus_pz*E_plus_pz))
            if pz > 0:
                self._rap = -self._rap
    
    def __str__(self):
        return stringReco(self)
    
    def __repr__(self):
        attributes = ["pdg: {pid} [{name}]".format(pid=self._pid,name=self.get_name()),
                      "E: {E}".format(E=self._E), "px: {px}".format(px=self._px),
                      "py: {py}".format(py=self._py), "pz: {pz}".format(pz=self._pz),
                      "mass: {m}".format(m=self._m)]
        return "Reco(" + "; ".join(attributes) +")"
    
    def get_name(self):
        pid = self._pid if (self._pid > 0) else -1*self._pid
        name = PDT[pid].name if PDT.has_key(pid) else str(None)
        
        if 1 <= pid <= 6:  # Quarks
            if self._pid > 0:
                return name.replace("bar", '')
            else: return name
        
        elif 11 <= pid <= 16: # Leptons
            if self._pid < 0:
                return name.replace('-', '+')\
                           .replace('_', "bar_")
            else: return name
        
        else: return str(None)
    
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
    def pT(self):
        try:
            return self._pT
        except:
            print("You must execute the method 'setpT' first")
            return None