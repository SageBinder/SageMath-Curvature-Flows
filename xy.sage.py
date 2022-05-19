

# This file was *autogenerated* from the file xy.sage
from sage.all_cmdline import *   # import sage library

_sage_const_0p5 = RealNumber('0.5'); _sage_const_3 = Integer(3); _sage_const_0 = Integer(0); _sage_const_1 = Integer(1); _sage_const_50 = Integer(50)
import numpy as np

def get_plot(X, x_0, x_m, x_f, y_0, y_f):
	plot_0 = parametric_plot3d(X(x, y), (x, x_0, x_m), (y, y_0, y_f), frame=False, color='green')		if x_0 != x_m else None

	plot_1 = parametric_plot3d(X(x, y), (x, x_m, x_f), (y, y_0, y_f), frame=False, color='purple', opacity=_sage_const_0p5 )		if x_m != x_f else None

	line = parametric_plot3d(X(x_m, y), (y, y_0, y_f), frame=False, color='black', thickness=_sage_const_3 , axes=True)

	axis = parametric_plot3d(X(x, _sage_const_0 ), (x, x_0, x_f), frame=False, color='red', thickness=_sage_const_3 )
	
	plot = line + axis
	if plot_0 is not None:
		plot += plot_0
	if plot_1 is not None:
		plot += plot_1

	return plot


__tmp__=var("x,y"); X = symbolic_expression((x, y, x*y)).function(x,y)

bounds = (-_sage_const_1 , _sage_const_1 )

a = animate([get_plot(X, bounds[_sage_const_0 ], t, bounds[_sage_const_1 ], bounds[_sage_const_0 ], bounds[_sage_const_1 ]) for t in np.linspace(bounds[_sage_const_0 ], bounds[_sage_const_1 ], _sage_const_50 )])
a.save("test.html", online=True)

