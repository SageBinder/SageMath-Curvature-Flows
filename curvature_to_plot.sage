import numpy as np
from sage.symbolic.integration.integral import definite_integral

pi = RR.pi()

def integrate_curvature(kappa, s, srange=(0,1), theta_0=0, x_0=0, y_0=0, step=0.1):
    theta, x, y = var('θ, x, y')
    DE0 = kappa
    DE1 = cos(theta)
    DE2 = sin(theta)
    ICs = [srange[0], theta_0, x_0, y_0]

    P = desolve_system_rk4([DE0, DE1, DE2], [theta, x, y], ics=ICs, ivar=s, end_points=srange[1], step=step)
    return P


def splines_from_curvature(kappa, s, srange=(0,1), theta_0=0, x_0=0, y_0=0, step=0.1):
    P = integrate_curvature(kappa, s, srange, theta_0, x_0, y_0, step)

    x_spline = spline([(s, x) for s, theta, x, y in P])
    y_spline = spline([(s, y) for s, theta, x, y in P])

    return (x_spline, y_spline)


def spline_plot_from_curvature(kappa, s, srange=(0,1), theta_0=0, x_0=0, y_0=0, color='automatic', axes=True, step=0.1):
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
    return f.definite_integral(srange[0], srange[1]) / (srange[1] - srange[0])


def splines_from_curvature_fix_center(kappa, s, srange=(0,1), theta_0=0, x_0=0, y_0=0, center=(0,0), step=0.1):
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

kappa(s, a) = 1/3 + sin(s) + 3/a * sin(3*s)

plots = []

for a in np.linspace(1, 15, 50):
    print(f"a: {a}")
    x, y = splines_from_curvature_fix_center(kappa(s, a), s, srange=(0, 6*pi))
    plots.append(parametric_plot((x, y), (0, 6*pi), color='black', axes=True))

print("Animating...")
a = animate(plots, xmin=-4, xmax=4, ymin=-4, ymax=4)
print("Saving...")
a.gif(savefile="closed_curve.gif", delay=4, show_path=True)

