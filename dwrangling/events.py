"""Events - The classs and functions associated with storing a collision event."""

import numpy as np

import dwrangling.reconstruction as DWR
import dwrangling.lhefiles as DWLHE



### Loading/Saving Functions ###
load_as_events = DWLHE.lhe_to_events



### Classes ###
class Event(object):
    def __init__(self, init_state, inter_state, leptons, hadrons,
                 met, muon, antimuon, neutrino, antineutrino):
        self._init_state = init_state
        self._inter_state = inter_state
        self._leptons = leptons
        self._hadrons = hadrons
        self._met = met
        self._muon = muon
        self._antimuon = antimuon
        self._neutrino = neutrino
        self._antineutrino = antineutrino
    
    def get_pbeam(self):
        pbeam = np.zeros(3)
        
        for particle in self._init_state:
            ppart = np.array([particle.px, particle.py, particle.pz])
            pbeam += ppart
        
        return pbeam
    
    def get_phat_beam(self):
        p_beam = self.get_pbeam()
        return p_beam / np.linalg.norm(p_beam)
    
    
    def get_pT(self, beam_on_z=True):
        leptons_pT = []
        hadrons_pT = []
        
        # If beam lies on z-axis
        if beam_on_z:
            for particle in self.leptons:
                leptons_pT.append(particle.perp)
            for particle in self.hadrons:
                hadrons_pT.append(particle.perp)
        
        # If beam doesn't lie on z-axis, i.e. has been rotated
        else:
            phat_beam = self.get_phat_beam()
            for particle in self.leptons:
                leptons_pT.append(particle.setpT(phat_beam))
            for particle in self.hadrons:
                hadrons_pT.append(particle.setpT(phat_beam))
        
        return {"leptons":leptons_pT, "hadrons":hadrons_pT}
    
    
    def __str__(self):
        attributes = [f"initial: [{ ', '.join([p.get_name(p.pid) for p in self.init_state]) }]",
                      f"intermediate: [{ ', '.join([p.get_name(p.pid) for p in self.inter_state]) }]",
                      f"leptons: [{ ', '.join([p.get_name(p.pid) for p in self.leptons]) }]",
                      f"hadrons: [{ ', '.join([p.get_name(p.pid) for p in self.hadrons]) }]",]
        return "; ".join(attributes)
    
    
    def __repr__(self):
        attributes = [f"met.E:{self.met.E:.2f}",
                      f"mu-.E:{self.muon.E:.2f}",
                      f"mu+.E:{self.antimuon.E:.2f}",
                      f"nu.E:{self.neutrino.E:.2f}",
                      f"nubar.E:{self.antineutrino.E:.2f}",]
        return "Event(" + self.__str__()+"; " + "; ".join(attributes) + ")"
    
    
    @property
    def init_state(self): return self._init_state
    @property
    def inter_state(self): return self._inter_state
    @property
    def leptons(self): return self._leptons
    @property
    def hadrons(self): return self._hadrons
    @property
    def met(self): return self._met
    @property
    def muon(self): return self._muon
    @property
    def antimuon(self): return self._antimuon
    @property
    def neutrino(self): return self._neutrino
    @property
    def antineutrino(self): return self._antineutrino
    
    
    @classmethod
    def from_lhe_event(cls, lhe_event):
        init_state, inter_state, leptons, hadrons = [], [], [], []
        met = DWR.Reco.empty()  # Missing Transverse Energy?
        muon, antimuon = DWR.Reco.empty(), DWR.Reco.empty()
        neutrino, antineutrino = DWR.Reco.empty(), DWR.Reco.empty()
        
        # Iterate through all particles in the event
        for particle in lhe_event['particles']:
            temp = DWR.Reco.from_dict(particle)
            
            # Initial Particles
            if particle['status'] == -1:
                init_state.append( temp )

            # Final Particles
            elif particle['status'] == 1:
                
                if temp.is_neutrino():
                    met = DWR.addReco(met, temp)

                    # Muon Type Neutrinos
                    if temp.pid == 14: 
                        neutrino = DWR.addReco(neutrino, temp)
                        neutrino.setpid(14)
                    elif temp.pid == -14: 
                        antineutrino = DWR.addReco(antineutrino, temp)
                        antineutrino.setpid(-14)

                elif temp.is_charged_lepton():
                    leptons.append(temp)

                    # Muon Type Charged Leptons
                    if temp.pid == 13: 
                        muon = DWR.addReco(muon, temp)
                        muon.setpid(13)
                    elif temp.pid == -13: 
                        antimuon = DWR.addReco(antimuon, temp)
                        antimuon.setpid(-13)

                else:
                    hadrons.append(temp)

            # Intermediate Particles
            else:
                inter_state.append( temp )

        return cls(init_state, inter_state, leptons, hadrons,
                   met, muon, antimuon, neutrino, antineutrino)