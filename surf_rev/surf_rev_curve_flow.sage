import numpy as np

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


def spline_avg(f):
    a = f.list()[0][0]
    b = f.list()[-1][0]
    return f.definite_integral(a, b) / (b - a)


def splines_to_angular_momentum(x_0, y_0, x_1, y_1, dt, srange=(0,1), center=(0,0)):
    def theta(a, b): return arccos(a.inner_product(b) / (a.norm() * b.norm()))
    def v_x(z): return (x_0(z) - x_1(z)) / dt
    def v_y(z): return (y_0(z) - y_1(z)) / dt
    def v(z): return vector([v_x(z), v_y(z)])
    def r(z): return sqrt((x_0(z) - center[0])^2 + (y_0(z) - center[1])^2)
    def w(z): return v(z).norm() * sin(theta(v(z), vector([x_0(z) - center[0], y_0(z) - center[1]]))) # |v| * sin(theta)
    integrand = lambda z: w(z) * r(z) # w * r
    angular_momentum = numerical_integral(integrand, srange[0], srange[1])[0] # int_S w * r^2 ds
    return angular_momentum


def splines_to_moment(x, y, srange=(0,1), center=(0,0)):
    return numerical_integral(lambda z: ((x(z) - center[0])^2 + (y(z) - center[1])^2), srange[0], srange[1])[0]


def translate_spline(f, dy):
    return spline([(x, y + dy) for x, y in f.list()])


def rotate_splines(x, y, theta):
    x_list = x.list()
    y_list = y.list()
    
    R = matrix([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
    
    completed_x_list = [(s, x, y(s)) for s, x in x_list]
    completed_y_list = [(s, x(s), y) for s, y in y_list]
    
    rotated_x_spline = spline([(s, (R*vector([x, y]))[0]) for s, x, y in completed_x_list])
    rotated_y_spline = spline([(s, (R*vector([x, y]))[1]) for s, x, y in completed_y_list])
    
    return (rotated_x_spline, rotated_y_spline)


def splines_fix_center(x, y, center=(0,0)):
    x_bar = spline_avg(x)
    y_bar = spline_avg(y)
    return (translate_spline(x, -x_bar + center[0]), translate_spline(y, -y_bar + center[1]))

def splines_from_curvature_fix_center(kappa, s, srange=(0,1), theta_0=0, center=(0,0), step=0.1):
    x, y = splines_from_curvature(kappa, s, srange, theta_0, 0, 0, step)
    return splines_fix_center(x, y, center)


def flow_curvature(kappa, srange, arange, acount, theta_0=0, center=(0,0), step=0.1):
    splines = []
    rotated_splines = []
    space, dt = np.linspace(arange[0], arange[1], acount, retstep=True)
    total_counterrotation = 0
    curves = []
    for a in space:
        print(f"Calculating curve for a = {a}...")
        x, y = splines_from_curvature_fix_center(kappa(s, a), s, srange=srange, theta_0=theta_0, center=center, step=step)
        # print(x.list())
        # print(y.list())
        print("Done with curvature integration.")

        if len(splines) >= 1:
            print("Calculating angular momentum...")
            angular_momentum = splines_to_angular_momentum(splines[-1][0], splines[-1][1], x, y, dt, srange=srange, center=center)
            print("Done with angular momentum.")
            print("Calculating moment...")
            I = splines_to_moment(x, y, srange=srange, center=center)
            print("Done with moment.")
            print(f"I = {I}")
            angular_velocity = angular_momentum / I
            dtheta = angular_velocity * dt
            total_counterrotation += dtheta
            print(f"dtheta: {dtheta}\n")
            
        rotated_x, rotated_y = rotate_splines(x, y, total_counterrotation)
        rotated_translated_x, rotated_translated_y = splines_fix_center(rotated_x, rotated_y, center)
        rotated_splines.append((rotated_translated_x, rotated_translated_y))
        splines.append((x, y))
        
    return splines, rotated_splines


c1 = 1
c2 = 0.2
N = 120

kappa(s, t) = 1/5 + cos(s)*exp(-c2*t) + 5*cos(3*s)*exp(-9*c2*t)
# kappa(s, a) = 1/3 + sin(s) + 3/a * sin(3*s)
srange = (0, 10*pi)
arange = (0, 2*pi)
acount = 50
# x, y = splines_from_curvature(kappa(s, 0), s, srange=srange, theta_0=0, step=0.1)
# plot = plot(x, (0, 10*pi))
# print(f"integral: {x.definite_integral(srange[0], srange[1]-0.001)}")
# plot.save("test.png")


_, rotated_splines = flow_curvature(kappa, srange, arange, acount, center=(6,0), step=0.05)

def X(phi, psi): return (lambda u, v: phi(v)*cos(u), lambda u, v: phi(v)*sin(u), lambda u, v: psi(v))
surf_a = animate([parametric_plot3d(X(phi, psi), (0, 2*pi), srange) for phi, psi in rotated_splines])
surf_a.save("surf_rev_hot_curve_flow.html", online=True, show_path=True)
# curve_a.gif(savefile="hot_curve_flow.gif", delay=12, show_path=True)


# kappa(s, a) = 1/3 + sin(s) + 3/a * sin(3*s)
# srange = (0, 6*pi)
# arange = (1, 3)
# acount = 10

# _, rotated_splines = flow_curvature(kappa, srange, arange, acount, center=(6,0))

# def X(phi, psi): return (lambda u, v: phi(v)*cos(u), lambda u, v: phi(v)*sin(u), lambda u, v: psi(v))
# # curve_a = animate([parametric_plot((lambda z: phi(z), lambda z: psi(z)), (0, 6*pi)) for phi, psi in rotated_splines])
# surf_a = animate([parametric_plot3d(X(phi, psi), (0, 2*pi), (0, 6*pi)) for phi, psi in rotated_splines])
# # surf_a.gif(savefile="surf_rev_curve_flow.gif", delay=12, show_path=True)
# surf_a.save("surf_rev_curve_flow.html", online=True, show_path=True)
# # curve_a.gif(savefile="curve_flow.gif", delay=12, show_path=True)