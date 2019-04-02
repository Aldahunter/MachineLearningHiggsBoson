import matplotlib.colors as mcolors

class StdCmap(mcolors.LinearSegmentedColormap):
    """Standard ColorMap class for the plotting module."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_middle(True)
    
    def __repr__(self):
        cname = self._middle_color
        cname = "" if (cname == "") else '('+cname+')'
        return f"StdCmap({self.name} | middle: {self._set_middle})"
    
    def __copy__(self):
        copy = super().__copy__()
        return StdCmap(copy.name, copy._segmentdata)
    copy = __copy__


    def set_middle(self, middle_on, color='green'):
        """If `True` sets a white-centre between the orange and blue, \
        otherwise removes the white center."""
        
        # Define primary colors
        pcolors = ["red", "green", "blue"]
        
        # Turn middle color on
        if middle_on:
            # Get RGB of specified color
            color = 'green' if (color is None) else color
            try: color_name, color = color, mcolors.to_rgb(color);
            except: color_name, color = "", color;
            self._middle_color = color_name
            
            # Iterate through the middle's rgb segmentdata 
            for rgb, pcolor in zip(color[:3], pcolors):
                # Set white -> (max(r), max(g), max(b))
                self._segmentdata[pcolor][3] = [0.5, rgb, rgb]
        # Or turn middle color off
        else:
            # Cycle through primary colors
            for pcolor in pcolors:
                # Set to midpoint of rgb's above and below
                mid = ((self._segmentdata[pcolor][2][2]
                        + self._segmentdata[pcolor][4][1]) / 2)
                self._segmentdata[pcolor][3] = [0.5, mid, mid]
        
        self._set_middle = bool(middle_on)
        return self
    
    def reversed(self):
        """Reverse the colormap."""
        self_r = super().reversed(self)
        self._segmentdata = self_r._segmentdata
        
        if self.name.endswith('_r'):
            self.name = self.name[:-2]
        else:
            self.name = self.name + '_r'
        return self
    _r = reversed
    
    
    @classmethod
    def from_sequence(cls, name, seq):
        """Returns a :class:`StdCmap`.

        Parameters:
         - name: A :class:`str` for the name of the color map.
         - seq: A :class:`list` of :class:`float`s and matplotlib color
                names as :class:`str`s (or RGB :class:`tuple`s). The floats
                should be increasing and in the interval (0,1), since they
                give the proportions between the RGB tuples."""
        
        seq = ([(None,) * 3, 0.0]
               + [mcolors.to_rgb(item) if isinstance(item, str) else item
                  for item in seq]
               + [1.0, (None,) * 3])
        
        cdict = {'red': [], 'green': [], 'blue': []}
        for i, item in enumerate(seq):

            if isinstance(item, float):
                r1, g1, b1 = seq[i - 1]
                r2, g2, b2 = seq[i + 1]

                cdict['red'].append([item, r1, r2])
                cdict['green'].append([item, g1, g2])
                cdict['blue'].append([item, b1, b2])

        return cls(name, cdict)