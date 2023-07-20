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
    theta_spline = spline([(s, theta) for s, theta, x, y in P])

    return (x_spline, y_spline, theta_spline)


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


def splines_to_angular_momentum(x_0, y_0, x_1, y_1, dt, srange=(0,1)):
    def theta(a, b): return arccos(a.inner_product(b) / (a.norm() * b.norm()))
    def v_x(z): return (x_0(z) - x_1(z)) / dt
    def v_y(z): return (y_0(z) - y_1(z)) / dt
    def v(z): return vector([v_x(z), v_y(z)])
    def w(z): return v(z).norm() * sin(theta(v(z), vector([x_0(z), y_0(z)]))) / sqrt(x_0(z)^2 + y_0(z)^2) # |v| * sin(theta) / r
    integrand = lambda z: w(z) * (x_0(z)^2 + y_0(z)^2) # w * r^2
    angular_momentum = numerical_integral(integrand, srange[0], srange[1])[0] # int_S w * r^2 ds
    return angular_momentum


def splines_to_moment(x, y, srange=(0,1)):
    return numerical_integral(lambda z: (x(z)^2 + y(z)^2), srange[0], srange[1])[0]


def splines_from_curvature_fix_center(kappa, s, srange=(0,1), theta_0=0, x_0=0, y_0=0, center=(0,0), step=0.1):
    x, y, theta = splines_from_curvature(kappa, s, srange, theta_0, x_0, y_0, step)
    x_bar = spline_avg(x, srange)
    y_bar = spline_avg(y, srange)
    return (lambda z: x(z) - x_bar, lambda z: y(z) - y_bar, theta)

kappa(s, a) = 1/3 + sin(s) + 3/a * sin(3*s)

plots = []
rotation_removed_plots = []
kappa_plots = []
theta_plots = []
splines = []
space, dt = np.linspace(1, 4, 6, retstep=True)
total_counterrotation = 0
for a in space:
    print(f"Calculating curve for a = {a}...")
    x, y, theta = splines_from_curvature_fix_center(kappa(s, a), s, theta_0=0, srange=(0, 6*pi))

    R = matrix.identity(2)
    if len(splines) >= 1:
        angular_momentum = splines_to_angular_momentum(splines[-1][0], splines[-1][1], x, y, dt, srange=(0, 6*pi))
        I = splines_to_moment(x, y, srange=(0, 6*pi))
        angular_velocity = angular_momentum / I
        dtheta = angular_velocity * dt
        total_counterrotation += dtheta
        print(f"dtheta: {dtheta}")
        R = matrix([[cos(total_counterrotation), -sin(total_counterrotation)], [sin(total_counterrotation), cos(total_counterrotation)]])
    f = lambda z: R * vector([x(z), y(z)])
    splines.append((x, y))

    rotation_removed_plots.append(parametric_plot((lambda z: f(z)[0], lambda z: f(z)[1]), (0, 6*pi), color='black', axes=True, ticks=[[], []]))
    plots.append(parametric_plot((x, y), (0, 6*pi), color='red', axes=True, ticks=[[], []]))

    pi_ticks = [i*pi for i in range(-10, 11)]
    pi_tick_labels = [f"{i}π" for i in range(-10, 11)]

    kappa_plots.append(plot(lambda z: kappa(z, a), color='green', axes=True, ticks=[pi_ticks, pi_ticks], tick_formatter=[pi_tick_labels, pi_tick_labels], xmin=0, xmax=6*pi))
    theta_plots.append(plot(theta, (0, 6*pi), color='blue', axes=True, ticks=[pi_ticks, pi_ticks], tick_formatter=[pi_tick_labels, pi_tick_labels], xmin=0, xmax=6*pi))


# print("Animating...")
# a_both = animate([rotation_removed_plot + plot for rotation_removed_plot, plot in zip(rotation_removed_plots, plots)], xmin=-4, xmax=4, ymin=-4, ymax=4)
# a_with_rotation = animate([plot for rotation_removed_plot, plot in zip(rotation_removed_plots, plots)], xmin=-4, xmax=4, ymin=-4, ymax=4)
# a_rotation_removed = animate([rotation_removed_plot for rotation_removed_plot, plot in zip(rotation_removed_plots, plots)], xmin=-4, xmax=4, ymin=-4, ymax=4)

# print("Saving...")

# a_both.gif(savefile="closed_curve_rotation_and_no_rotation.gif", delay=12, show_path=True)
# a_with_rotation.gif(savefile="closed_curve_with_rotation.gif", delay=12, show_path=True)
# a_rotation_removed.gif(savefile="closed_curve_rotation_removed.gif", delay=12, show_path=True)

# a_both.gif(savefile="closed_curve_rotation_and_no_rotation_fast.gif", delay=4, show_path=True)
# a_with_rotation.gif(savefile="closed_curve_with_rotation_fast.gif", delay=4, show_path=True)
# a_rotation_removed.gif(savefile="closed_curve_rotation_removed_fast.gif", delay=4, show_path=True)

for i, p in enumerate(rotation_removed_plots):
    p.save_image(f"curve/curve_{i}.png", xmin=-4, xmax=4, ymin=-4, ymax=4)

for i, p in enumerate(kappa_plots):
    p.save_image(f"kappa/kappa_{i}.png", ymin=-4, ymax=4)

for i, p in enumerate(theta_plots):
    p.save_image(f"theta/theta_{i}.png", ymin=0, ymax=10)

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
