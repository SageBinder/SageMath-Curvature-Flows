

# This file was *autogenerated* from the file curvature_to_plot.sage
from sage.all_cmdline import *   # import sage library

_sage_const_0 = Integer(0); _sage_const_1 = Integer(1); _sage_const_0p1 = RealNumber('0.1'); _sage_const_3 = Integer(3); _sage_const_15 = Integer(15); _sage_const_50 = Integer(50); _sage_const_6 = Integer(6); _sage_const_4 = Integer(4)
import numpy as np
from sage.symbolic.integration.integral import definite_integral

pi = RR.pi()

def integrate_curvature(kappa, s, srange=(_sage_const_0 ,_sage_const_1 ), theta_0=_sage_const_0 , x_0=_sage_const_0 , y_0=_sage_const_0 , step=_sage_const_0p1 ):
    theta, x, y = var('θ, x, y')
    DE0 = kappa
    DE1 = cos(theta)
    DE2 = sin(theta)
    ICs = [srange[_sage_const_0 ], theta_0, x_0, y_0]

    P = desolve_system_rk4([DE0, DE1, DE2], [theta, x, y], ics=ICs, ivar=s, end_points=srange[_sage_const_1 ], step=step)
    return P


def splines_from_curvature(kappa, s, srange=(_sage_const_0 ,_sage_const_1 ), theta_0=_sage_const_0 , x_0=_sage_const_0 , y_0=_sage_const_0 , step=_sage_const_0p1 ):
    P = integrate_curvature(kappa, s, srange, theta_0, x_0, y_0, step)
    
    # x_spline = PolynomialRing(RR, s).lagrange_polynomial([(s, x) for s, theta, x, y in P])
    # y_spline = PolynomialRing(RR, s).lagrange_polynomial([(s, y) for s, theta, x, y in P])
    x_spline = spline([(s, x) for s, theta, x, y in P])
    y_spline = spline([(s, y) for s, theta, x, y in P])

    return (x_spline, y_spline)


def spline_plot_from_curvature(kappa, s, srange=(_sage_const_0 ,_sage_const_1 ), theta_0=_sage_const_0 , x_0=_sage_const_0 , y_0=_sage_const_0 , color='automatic', axes=True, step=_sage_const_0p1 ):
    P = integrate_curvature(kappa, s, srange, theta_0, x_0, y_0, step)
    
    x_points = []
    y_points = []
    for s, theta, x, y in P:
        x_points.append((s, x))
        y_points.append((s, y))

    x_spline = spline(x_points)
    y_spline = spline(y_points)

    return parametric_plot((x_spline, y_spline), srange, color=color, axes=axes)


def spline_avg(f, srange):
    return f.definite_integral(srange[_sage_const_0 ], srange[_sage_const_1 ]) / (srange[_sage_const_1 ] - srange[_sage_const_0 ])


def splines_from_curvature_fix_center(kappa, s, srange=(_sage_const_0 ,_sage_const_1 ), theta_0=_sage_const_0 , x_0=_sage_const_0 , y_0=_sage_const_0 , center=(_sage_const_0 ,_sage_const_0 ), step=_sage_const_0p1 ):
    x, y = splines_from_curvature(kappa, s, srange, theta_0, x_0, y_0, step)
    x_bar = spline_avg(x, srange)
    y_bar = spline_avg(y, srange)
    return (lambda z: x(z) - x_bar, lambda z: y(z) - y_bar)

# c1 = 1
# c2 = 0.2
# N=120

# kappa(s, t) = 1/5 + cos(s)*exp(-c2*t) + 5*cos(3*s)*exp(-9*c2*t)
# plots = [spline_plot_from_curvature(kappa(s, t), s, srange=(0, 10*pi + 0.2), color='black', axes=False) for t in [k*2*pi/N for k in range(N)]]
# a = animate(plots, figsize=[6, 6])
# a.gif(savefile="hotcurve.gif", delay=4, show_path=True)

# kappa(s, t) = 1/5 + cos(s)*cos(c1 * t) + cos(3*s)*cos(3*c1*t)
# plots = [spline_plot_from_curvature(kappa(s, t), s, srange=(0, 10*pi + 0.2), color='black', axes=False) for t in [k*2*pi/N for k in range(N)]]
# a = animate(plots, figsize=[6, 6])
# a.gif(savefile="vibcurve_3.gif", delay=4, show_path=True)

# kappa(s, a) = 1/3 + sin(s) + 3/a * sin(3*s)
# plots = [spline_plot_from_curvature(kappa(s, a), s, srange=(0, 6*pi), color=Color((6-a)/5, 0, a/5)) for a in np.linspace(1, 5, 50)]
# a = animate(plots, xmin=-5, xmax=2, ymin=-4, ymax=4)
# a.gif(savefile="closed_curve.gif", delay=4, show_path=True)

# kappa(s, a) = 1/3 + 1/a*sin(s)
# plots = [spline_plot_from_curvature(kappa(s, a), s, srange=(0, 6*pi), color='red') for a in np.linspace(1, 15, 50)]
# a = animate(plots, xmin=-5, xmax=3, ymin=-2, ymax=6.5)
# a.gif(savefile="to_circle.gif", delay=4, show_path=True)

__tmp__=var("s,a"); kappa = symbolic_expression(_sage_const_1 /_sage_const_3  + sin(s) + _sage_const_3 /a * sin(_sage_const_3 *s)).function(s,a)

plots = []

for a in np.linspace(_sage_const_1 , _sage_const_15 , _sage_const_50 ):
    print(f"a: {a}")
    x, y = splines_from_curvature_fix_center(kappa(s, a), s, srange=(_sage_const_0 , _sage_const_6 *pi))
    plots.append(parametric_plot((x, y), (_sage_const_0 , _sage_const_6 *pi), color='black', axes=True))

    # x_bar = func_avg(x, s, srange=(0, 6*pi))
    # y_bar = func_avg(y, s, srange=(0, 6*pi))
    # f(s) = (x(s) - x_bar, y(s) - y_bar)
    # plots.append(parametric_plot(f, (s, 0, 6*pi), color='black', axes=True))

print("Animating...")
a = animate(plots, xmin=-_sage_const_4 , xmax=_sage_const_4 , ymin=-_sage_const_4 , ymax=_sage_const_4 )
print("Saving...")
a.gif(savefile="closed_curve.gif", delay=_sage_const_4 , show_path=True)


