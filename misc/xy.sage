import numpy as np

def get_plot(X, x_0, x_m, x_f, y_0, y_f):
	plot_0 = parametric_plot3d(X(x, y), (x, x_0, x_m), (y, y_0, y_f), frame=False, color='green')\
		if x_0 != x_m else None

	plot_1 = parametric_plot3d(X(x, y), (x, x_m, x_f), (y, y_0, y_f), frame=False, color='purple', opacity=0.5)\
		if x_m != x_f else None

	line = parametric_plot3d(X(x_m, y), (y, y_0, y_f), frame=False, color='black', thickness=3, axes=True)

	axis = parametric_plot3d(X(x, 0), (x, x_0, x_f), frame=False, color='red', thickness=3)
	
	plot = line + axis
	if plot_0 is not None:
		plot += plot_0
	if plot_1 is not None:
		plot += plot_1

	return plot


X(x, y) = (x, y, x*y)

bounds = (-1, 1)

a = animate([get_plot(X, bounds[0], t, bounds[1], bounds[0], bounds[1]) for t in np.linspace(bounds[0], bounds[1], 50)])
a.save("test.html", online=True)