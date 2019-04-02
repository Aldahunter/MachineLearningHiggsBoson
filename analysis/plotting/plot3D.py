"""A Module to wrap the Mayavi.mlab module for plotting 3d figures."""
import matplotlib
matplotlib.use('Qt5Agg')
matplotlib.interactive(True)
from warnings import warn as raiseWarn
raiseWarn("Ensure '%gui qt' is the backend for iPython.", ImportWarning)

import numpy as np
import mayavi
from mayavi import mlab
from copy import deepcopy
import analysis.plotting as AP

class plot3D(object):
    """A wrapper object for simply producing mayavi.mlab 3D plots."""
    
    ## Crucial Private Class Attributes ##
    _extent = [0, 100, 0, 100, 0, 100]
    _drawing = []
    _actors = []
    _drawn = False
    _fig = None
    
    
    ## Define Class Default Values ##
    _fig_kwargs = {'fgcolor':(0,0,0), 'bgcolor':(1,1,1)}
    _view_kwargs = {'azimuth':155, 'elevation':55, 'distance':350,
                    'focalpoint':np.array([50, 50, 50])}
    
    _cmap = AP.std_cmap.copy().set_middle(False)
    _cmap_props = {'vmin':None, 'vmax':None, 'alpha':0.8}
    
    _axes_keys = ['color', 'line_width', 'opacity',
                  'xvisible', 'yvisible', 'zvisible']
    _axes_props = {'color':(0,0,0), 'line_width':2.0, 'opacity':1.0,
                   'xvisible':True, 'yvisible':True, 'zvisible':True}
    _outline_keys = ['color', 'line_width', 'opacity',
                     'cornered', 'corner_factor']
    _outline = {'cornered':True, 'corner_factor':1/6}
    
    ## Setup Tick Paramaters Backend ##
    _tick_params = {'fly_mode': 'outer_edges', 'n_ticks':5,
                    'label_format': '%.0f', 'bold': True,
                    'italic': True, 'shadow': False,
                    'labelpad':3, 'frame_label':True,
                    'frame_color':(1,1,1), 'frame_width':0,
                    'fontsize':12, 'font':'times',
                    'fontcolor': (0,0,0), 'tight': True,
                    'ha': 'left', 'va': 'bottom',
                    'corner_offset': 0.025}
    def _fly_mode(ax, value): ax.axes.fly_mode = value;
    def _n_ticks(ax, value): ax.axes.number_of_labels = value;
    def _corner_offset(ax, value): ax.axes.corner_offset = value;
    def _tick_format(ax, value): ax.axes.label_format = value;
    def _tick_bold(ax, value): ax.label_text_property.bold = value;
    def _tick_italic(ax, value): ax.label_text_property.italic = value;
    def _tick_shadow(ax, value): ax.label_text_property.shadow = value;
    def _tick_frame(ax, value): ax.label_text_property.frame = value;
    def _tick_frame_width(ax, value):
        ax.label_text_property.frame_width = value;
    def _tick_frame_color(ax, value):
        ax.label_text_property.frame_color = value;
    def _tick_fontsize(ax, value):
        ax.label_text_property.font_size = value;
    def _tick_fontfamily(ax, value):
        ax.label_text_property.font_family = value;
    def _tick_fontcolor(ax, value):
        ax.label_text_property.color  = value;
    def _tick_tbox(ax, value):
        ax.label_text_property.use_tight_bounding_box = value;
    def _tick_ha(ax, value):
        ax.label_text_property.justification = value;
    def _tick_va(ax, value):
        ax.label_text_property.vertical_justification = value;
    def _tick_offset(ax, value):
        ax.label_text_property.line_offset = value;
    _tick_funcs = {'fly_mode': _fly_mode, 'n_ticks':_n_ticks,
                   'label_format':_tick_format, 'bold': _tick_bold,
                   'italic':_tick_italic, 'shadow':_tick_shadow,
                   'labelpad':_tick_offset, 'frame_label':_tick_frame,
                   'frame_color':_tick_frame_color, 'ha':_tick_ha, 
                   'frame_width':_tick_frame_width, 'va':_tick_va,
                   'fontsize':_tick_fontsize, 'font':_tick_fontfamily,
                   'fontcolor':_tick_fontcolor, 'tight':_tick_tbox,
                   'corner_offset':_corner_offset}
    
    ## Setup Label Paramaters Backend ##
    _label_params = {'bold': True, 'italic': False,
                     'shadow': False, 'labelpad':0,
                     'frame_label':True, 'frame_width':0,
                     'frame_color':(1,1,1), 'fontsize':13,
                     'font':'times', 'fontcolor': (0,0,0),
                     'tight': True, 'ha': 'left', 'va': 'bottom'}
    def _x_label(ax, value): ax.axes.x_label = value;
    def _y_label(ax, value): ax.axes.y_label = value;
    def _z_label(ax, value): ax.axes.z_label = value;
    def _label_bold(ax, value): ax.title_text_property.bold = value;
    def _label_italic(ax, value): ax.title_text_property.italic= value;
    def _label_shadow(ax, value): ax.title_text_property.shadow= value;
    def _label_frame(ax, value): ax.title_text_property.frame = value;
    def _label_frame_width(ax, value):
        ax.title_text_property.frame_width = value;
    def _label_frame_color(ax, value):
        ax.title_text_property.frame_color = value;
    def _label_fontsize(ax, value):
        ax.title_text_property.font_size = value;
    def _label_fontfamily(ax, value):
        ax.title_text_property.font_family = value;
    def _label_fontcolor(ax, value):
        ax.title_text_property.color  = value;
    def _label_tbox(ax, value):
        ax.title_text_property.use_tight_bounding_box = value;
    def _label_ha(ax, value):
        ax.title_text_property.justification = value;
    def _label_va(ax, value):
        ax.title_text_property.vertical_justification = value;
    def _label_offset(ax, value):
        ax.title_text_property.line_offset = value;
    _label_funcs = {'x_label':_x_label, 'y_label':_y_label,
                    'z_label':_z_label, 'bold':_label_bold,
                    'italic':_label_italic, 'shadow':_label_shadow,
                    'labelpad':_label_offset, 'font':_label_fontfamily,
                    'frame_label':_label_frame,  'tight':_label_tbox,
                    'frame_color':_label_frame_color,
                    'frame_width':_label_frame_width,
                    'fontsize':_label_fontsize, 'ha':_label_ha, 
                    'fontcolor':_label_fontcolor, 'va':_label_va}
    
    
    def __init__(self, x0=None, x1=None, y0=None, y1=None,
                 z0=None, z1=None, **fig_kwargs):
        """Init Funct"""
        
        # Create figure instance with supplied kwargs
        self._fig_kwargs.update(fig_kwargs)
        self._fig = mlab.figure(**self._fig_kwargs)
        
        # Get axes limits
        self._axes_lims = np.array([[x0, x1], [y0, y1], [z0, z1]])
        self._find_lims = self._axes_lims == None
        
        # If limits not specified set as infs for comparision
        self._axes_lims[:,0] = np.where(self._find_lims[:,0], 
                                        +np.inf, self._axes_lims[:,0])
        self._axes_lims[:,1] = np.where(self._find_lims[:,1],
                                        -np.inf, self._axes_lims[:,1])
    
    
    def _get_lims(self, x, y, z):
        
        # Get the actor's (object's) x-, y-, z-limits
        actor_lims = np.array([[np.min(x), np.max(x)],
                               [np.min(y), np.max(y)],
                               [np.min(z), np.max(z)]])
        
        # Compare with axes limits for masks
        min_mask = actor_lims[:,0] < self._axes_lims[:,0]
        max_mask = actor_lims[:,1] > self._axes_lims[:,1]
        
        # Change axes limits for new mins/maxs
        self._axes_lims[:,0] = np.where(min_mask &self._find_lims[:,0],
                                        actor_lims[:,0],
                                        self._axes_lims[:,0])
        self._axes_lims[:,1] = np.where(max_mask &self._find_lims[:,1],
                                        actor_lims[:,1],
                                        self._axes_lims[:,1])
        
        # Return actor's limits
        return actor_lims
    
    
    @staticmethod
    def _get_extent(i, i_min, i_max):
        return ((i - i_min) * 10) / ((i_max - i_min) * 0.1)
    
    
    def _get_extents(self, actor_lims):
        actor_range = []
        for actor_lim, axis_lims in zip(actor_lims, self._axes_lims):
            actor_range += [*self._get_extent(np.array(actor_lim),
                                              *axis_lims)]
        return np.array(actor_range)    
    
    
    def _draw_surf(self, x, y, z, surf_lims, contours, kwargs):
        
        # Take out local cmap properties
        surf_cmap_props = self.cmap_props
        if 'cmap_props' in kwargs:
            surf_cmap_props = kwargs['cmap_props']
            del kwargs['cmap_props']
        
        # Update surface kwargs
        surf_kwargs = {'colormap':self.cmap, 'color':None,
                       'vmin': surf_cmap_props.get('vmin', None),
                       'vmax': surf_cmap_props.get('vmax', None)}
        surf_kwargs.update(**kwargs)
        
        # Get the range (extent) of surf over the x-, y-, z-axis
        surf_ranges = self._get_extents(surf_lims)
        
        # If custom colormap remove from kwargs, to add after
        cmap = None
        if not isinstance(surf_kwargs['colormap'], str):
            cmap = surf_kwargs['colormap']
            del surf_kwargs['colormap']
        
        # Plot the surface
        surf = mlab.surf(x.T, y.T, z, #figure=self._fig,
                         extent=surf_ranges, **surf_kwargs)
        
        # Add custom colormap
        if (cmap is not None) or ('alpha' in surf_cmap_props):
            if cmap is None:
                cmap = surf.module_manager.scalar_lut_manager \
                           .lut.table.to_array()
            
            lut = self._colormap(surf, cmap, surf_cmap_props)
            
            # Apply new cmap to actor
            surf.module_manager.scalar_lut_manager.lut.table = lut
        
        # Maybe set surf's scale?
        # surf.actor.actor.scale = self._ax_scale

        # If contours wanted, apply contour properties
        if bool(contours['n_contours']):
            surf.enable_contours = True
            surf.contour.auto_contours = True
            surf.actor.mapper.scalar_mode = 'use_cell_data'
            surf.contour.filled_contours = contours['filled']
            surf.contour.number_of_contours = contours['n_contours']
        
        # Add surface actor to internal list
        self._actors.append(surf)
    
    
    def surf(self, x, y, z, contours={}, **kwargs):
        for kwarg in ['figure', 'extent']:
            if kwarg in kwargs:
                raise ValueError(f"Cannot set kwarg '{kwarg}'.")
                
        # Update contour properties
        if contours is None:
            surf_contours = None
        else:
            surf_contours = {'n_contours':10, 'filled':True}
            surf_contours.update(contours)
        
        # Obtain surf's limits and update axes' limits
        surf_lims = self._get_lims(x, y, z)
        
        # Add surface to drawring list
        self._drawing.append(((deepcopy(x), deepcopy(y), deepcopy(z),
                               surf_lims, surf_contours, kwargs),
                              self._draw_surf))
    
    
    def _draw_axes(self, actor):
        
        # Get and remove axis visibility from dict 
        xvisible = self.axes_props.get('xvisible', None)
        if xvisible is not None: del self.axes_props['xvisible'];
        yvisible = self.axes_props.get('yvisible', None)
        if yvisible is not None: del self.axes_props['yvisible'];
        zvisible = self.axes_props.get('zvisible', None)
        if zvisible is not None: del self.axes_props['zvisible'];

        # Create axes instance
        ax = mlab.axes(actor, #figure=self._fig,
                       extent=self._extent,
                       ranges=self._axes_lims.ravel(),
                       **self.axes_props)
        
        # Set axes visibilities
        if xvisible is not None:
            ax.axes.x_axis_visibility = xvisible
            self.axes_props = {'xvisible': xvisible}
        if yvisible is not None:
            ax.axes.y_axis_visibility = yvisible
            self.axes_props = {'yvisible': yvisible}
        if zvisible is not None:
            ax.axes.z_axis_visibility = zvisible
            self.axes_props = {'zvisible': zvisible}
        
        # Set ticks and ticklabels
        for param, value in self.tick_params.items():
            self._tick_funcs[param](ax, value)
        
        # Set axes labels
        for param, value in self.label_params.items():
            self._label_funcs[param](ax, value)
        
        # Add surface actor to internal list
        self._actors.append(ax)

        # Setup axis outline [optional]
        if self._outline:
            # Use axes' properties if properties not specified
            outline_props= {'color':self.axes_props['color'],
                            'line_width':self.axes_props['line_width'],
                            'opacity':self.axes_props['opacity']}
            
            # If outline is a dictionary, update proerties
            if isinstance(self._outline, dict):
                for key in outline_props.keys():
                    if key in self._outline:
                        outline_props[key] = self._outline[key]
            
            # Create outline instance
            outline = mlab.outline(actor, #figure=self._fig,
                                   extent=self._extent,
                                   **outline_props)
            
            # Apply cornered outline properties
            if self._outline['cornered']:
                outline.outline_mode = 'cornered'
                outline.outline_filter \
                       .corner_factor = self._outline['corner_factor']
            else:
                outline.outline_mode = 'full'
            
            # Add outline actor to internal list
            self._actors.append(outline)
    
    
    def _colormap(self, actor, cmap, cmap_props):
        
        # Get Colormap Look-Up-Table
        if not isinstance(cmap, np.ndarray):            
            # Get a list of 256 RGBA tuples
            lut = cmap(np.arange(256))
        else:
            # Already a LUT
            lut = cmap
            lut = lut.astype(float) / lut.max()
            
        # If alpha is set apply properties
        if cmap_props['alpha']:
            
            calpha = cmap_props['alpha']
            
            # If alpha is a single number between (0,1] set all RGBA
            # tuples to this alpha
            if isinstance(calpha, (int, float)) and (0 < calpha <= 1):
                lut[:,-1] = calpha
            # Else get alpha gradient across tuple (a0, a1)
            elif isinstance(calpha, tuple) and (len(calpha) == 2):
                lut[:,-1] = np.linspace(*calpha, 256)
            # Otherwise it is an incorrect format so inform user
            else:
                raise ValueError("'cmap_props['alpha']' must be a " +
                                 "single value between (0,1] or a " +
                                 "tuple of (alpha0, alpha1).")
                                        
        # Convert LUT to array of ints between [0,255]
        lut *= 255
        lut = lut.round().astype('uint8')
        return lut
    

    def clear_drawings(self):
        for obj in self._drawing:
            del obj
        self._actors = []
    
    
    def clear_actors(self):
        for actor in self._actors:
            del actor
        self._actors = []
    
    def _redraw(self):
        self.mlab.draw(figure=self._fig)
    
    def _draw(self):
        
        # If already drawn re-draw figure
        if self._drawn:
            self._redraw()
        
        # Remove all previous actors
        self.clear_actors()
        
        # Draw actors onto figure
        for args, actor_fn in self._drawing:
            actor_fn(*deepcopy(args))
        
        # Draw axes (attached to first actor)
        self._draw_axes(self._actors[0])
        
        # Render transparent images correctly
        self._fig.scene.renderer.use_depth_peeling = 1
        
        # Setup view of 3d plot
        mlab.view(**self._view_kwargs)
        
        self._drawn = True
    
    def savefig(self, filepath):
        self._draw()
        mlab.savefig(filename=filepath)
    
    def show(self):
        self._draw()
        return self._fig
    
    
    
    ## Public Attributes ##
    @property
    def cmap(self): return self._cmap
    @cmap.setter
    def cmap(self, value):
        if not isinstance(value, matplotlib.colors.Colormap):
            raise("'cmap' must be a 'Colormap' object.")
        self._cmap = value
            
            
    @property
    def cmap_props(self): return self._cmap_props
    @cmap_props.setter
    def cmap_props(self, values):
        if not isinstance(values, dict):
            raise ValueError("'cmap_props' must be a 'dict'.")
        for kwarg in values.keys():
            if kwarg not in self._cmap_props:
                raise ValueError(f"Invalid argument '{kwarg}', " +
                    f"valid arguments are: {self._cmap_props.keys()}")
        self._cmap_props.update(values)
    
    
    @property
    def axes_props(self): return self._axes_props
    @axes_props.setter
    def axes_props(self, values):
        if not isinstance(values, dict):
            raise ValueError("'axes_props' must be a 'dict'.")
        for kwarg in values.keys():
            if kwarg not in self._axes_keys:
                raise ValueError(f"Invalid argument '{kwarg}', " +
                    f"valid arguments are: {self._axes_keys.keys()}")
        self._axes_props.update(values)
    
    
    @property
    def outline(self): return self._outline
    @outline.setter
    def outline(self, values):
        if not isinstance(values, (dict, type(None))):
            raise ValueError("'outline' must be either a 'dict' " +
                             "or 'None'.")
        if values is None: self._outline = None
        else:
            for kwarg in values.keys():
                if kwarg not in self._outline_keys:
                    raise ValueError(f"Invalid argument '{kwarg}', " +
                         f"valid arguments are: {self._outline_keys}")
            if self._outline is None: self._outline = values
            else: self._outline.update(values)
    
    
    @property
    def tick_params(self): return self._tick_params
    @tick_params.setter
    def tick_params(self, values):
        if not isinstance(values, dict):
            raise ValueError("'tick_params' must be a 'dict'.")
        for kwarg in values.keys():
            if kwarg not in self._tick_funcs:
                raise ValueError(f"Invalid argument '{kwarg}', " +
                    f"valid arguments are: {self._tick_funcs.keys()}")
        self._tick_params.update(values)
    
    
    @property
    def label_params(self): return self._label_params
    @label_params.setter
    def label_params(self, values):
        if not isinstance(values, dict):
            raise ValueError("'label_params' must be a 'dict'.")
        for kwarg in values.keys():
            if kwarg not in self._label_funcs:
                raise ValueError(f"Invalid argument '{kwarg}', " +
                    f"valid arguments are: {self._label_funcs.keys()}")
        self._label_params.update(values)
    
    
    @property
    def x_label(self): return self._label_params.get('x_label', None)
    @x_label.setter
    def x_label(self, value):
        if not isinstance(value, str):
            raise ValueError("'x_label' must be a 'str'.")
        self._label_params['x_label'] = value
    
    
    @property
    def y_label(self): return self._label_params.get('y_label', None)
    @y_label.setter
    def y_label(self, value):
        if not isinstance(value, str):
            raise ValueError("'y_label' must be a 'str'.")
        self._label_params['y_label'] = value
    
    
    @property
    def z_label(self): return self._label_params.get('z_label', None)
    @z_label.setter
    def z_label(self, value):
        if not isinstance(value, str):
            raise ValueError("'z_label' must be a 'str'.")
        self._label_params['z_label'] = value
    
    @property
    def view(self): return self._view_kwargs
    @view.setter
    def view(self, values):
        if not isinstance(values, dict):
             raise ValueError("'view' must be a 'dict'.")
        for kwarg in values.keys():
            if kwarg not in self._view_kwargs.keys():
                raise ValueError(f"Invalid argument '{kwarg}', " +
                    f"valid arguments are: {self._view_kwargs.keys()}")
        self._view_kwargs.update(values)
            
        