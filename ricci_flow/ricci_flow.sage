import numpy as np
import os
import logging
import time

# logging.disable(logging.INFO)

pi = RR.pi()

plot_initial_curve = True
plot_initial_m = True
plot_initial_h = True
plot_initial_R = True
plot_initial_K = True
plot_initial_tissot = True

animate_curve = False
animate_m = False
animate_h = False
animate_R = False
animate_tissot = True

folder_name = "./Fig3_flow_tissot_test_3"
print(f"Using folder: {folder_name}")
if not os.path.exists(folder_name):
    print("Folder did not exist. Creating...")
    os.mkdir(folder_name)

def path(name):
    return os.path.join(folder_name, name)


def clamp(x, low, high):
    return low if x < low else high if x > high else x


def tissot(g, urange=(0, 2*pi), vrange=(0, pi), ucount=5, vcount=5, sq_len=0.2):
    def angle(a, b):
        return arccos(a.inner_product(b) / (a.norm() * b.norm())) if a.norm() * b.norm() != 0 else 0
    
    squares = []
    transformed_squares = []
    ellipses = []

    uspace = np.linspace(urange[0], urange[1], ucount)
    vspace = np.linspace(vrange[0], vrange[1], vcount)
    space = cartesian_product_iterator([uspace, vspace])
    for u, v in space:
        g_curr = g(u, v)
        eigenvectors = g_curr.eigenvectors_right()
        eigvec = eigenvectors[0][1][0]
        k1 = eigenvectors[0][0]
        if eigenvectors[0][2] == 2:
            k2 = k1
        else:
            k2 = eigenvectors[1][0]
        
        square = matrix([[0, -sq_len/2, 0, sq_len/2], [-sq_len/2, 0, sq_len/2, 0]])
        cross = vector([eigvec[0], eigvec[1], 0]).cross_product(vector([1, 0, 0]))
        theta = arctan2(cross[2], eigvec.dot_product(vector([1, 0])))
        R = matrix([[cos(theta), sin(theta)], [-sin(theta), cos(theta)]])
        
        translate = matrix([[u, u, u, u], [v, v, v, v]])
        transformed = g_curr * R * square
        
        squares.append((R * square) + translate)
        transformed_squares.append(transformed + translate)
        ellipses.append((u, v, abs(k1*sq_len/2), abs(k2*sq_len/2), -theta))

    return squares, transformed_squares, ellipses


def make_g(h, m):
    return lambda theta, rho: matrix([[h(rho), 0], [0, m(rho)]])


def revolve(x, y):
    # x/y are reversed to rotate around z axis
    return (lambda u, v: y(v)*cos(u), lambda u, v: y(v)*sin(u), lambda u, v: x(v))


def xy_splines_from_hm(h, m, srange=(0, pi), step_size=0.1):
    def y(rho):
        if m(rho) < 0:
            logging.debug(f"\tMaking xy splines from hm, encountered negative: ({rho}, {m(rho)}). Returning 0.")
            return 0
        return sqrt(m(rho))
    
    rho_space, drho = np.linspace(srange[0], srange[1], round((srange[1] - srange[0]) / step_size), retstep=True)
    y_spline = spline([(rho, y(rho)) for rho in rho_space])
    
    def x_integral(rho1, rho2):
        def integrand(s):
            d = h(s) - (y_spline.derivative(s))**2
            return sqrt(d) if d > 0 else 0
        return numerical_integral(integrand, rho1, rho2)[0]

    x_spline_list = [(rho_space[0], x_integral(0, rho_space[0]))]
    for rho in rho_space[1:]:
        last_rho = x_spline_list[-1][0]
        last_x = x_spline_list[-1][1]
        x_spline_list.append((rho, last_x + x_integral(last_rho, rho)))
    x_spline = spline(x_spline_list)
    
    return x_spline, y_spline


def sqrt_spline(f):
    return spline([(x, sqrt(y)) for x, y in f])


def to_spline(f, srange=(0, pi), step_size=0.01):
    return spline([(s, f(s)) for s in np.linspace(srange[0], srange[1], round((srange[1] - srange[0]) / step_size))])


def hm_to_ricci_tensor(h, m, return_K=False, eps=0.1):
    sqrt_m = sqrt_spline(m)
    
    def K(rho):
        return -sqrt_m.derivative(rho, order=2) / sqrt_m(rho)

    def R(rho):
        rho = clamp(rho, eps, pi-eps)
        R11 = K(rho) + ((m.derivative(rho) * h.derivative(rho)) / (4 * m(rho) * h(rho)))
        R22 = (R11 / h(rho)) * m(rho)
        return matrix([[R11, 0], [0, R22]])

    return R if not return_K else (R, K)


def reparam(h, m, ds=0.1):
    l_spline_list = [(0, 0)]
    def l_integrand(s): return sqrt(h(s)) if h(s) > 0 else 0
    s_space = np.linspace(0, pi, round(pi / ds))

    curr_int = 0
    for i, curr_s in enumerate(s_space[1:]):
        curr_int += numerical_integral(l_integrand, s_space[i], curr_s)[0]
        l_spline_list.append((curr_s, curr_int))

    l = spline(l_spline_list)
    l_inv = spline([(b, a) for a, b in l.list()])

    tot_s = l(pi)
    h_reparam = to_spline(lambda z: (tot_s / pi)^2)
    m_reparam = spline([(rho, max(0, m(l_inv(rho * tot_s / pi)))) for rho, _ in m.list()])

    return h_reparam, m_reparam


def add_cap(h, m):
    h_capped = spline(h.list())
    m_capped = spline(m.list())

    m_capped.append((0, 0))
    m_capped.append((pi, 0))

    sqrt_m = sqrt_spline(m_capped)

    h_capped.append((0, sqrt_m.derivative(0)**2))
    h_capped.append((pi, sqrt_m.derivative(pi)**2))

    return h_capped, m_capped


def euler_step(h, m, dt, rho_space, eps, k_space=None, cap=True, return_R=False):
    if k_space is None:
        logging.info("\tComputing Ricci tensor")
        R = hm_to_ricci_tensor(h, m, eps=eps)
        def k(rho): return -2*R(rho)
        k_space = [k(rho) for rho in rho_space]
    rho_k_space = list(zip(rho_space, k_space))

    logging.info("\tComputing h spline")
    h_next = spline([(rho, h(rho) + k[0][0]*dt) for rho, k in rho_k_space])
    logging.info("\tComputing m spline")
    m_next = spline([(rho, m(rho) + k[1][1]*dt) for rho, k in rho_k_space])

    if cap:
        h_next, m_next = add_cap(h_next, m_next)

    return (h_next, m_next, k_space, R) if return_R else (h_next, m_next, k_space)


def rk4_step(h1, m1, dt, eps=0.01, drho=0.01):
    rho_space = np.linspace(eps, pi-eps, round((pi-2*eps) / drho))

    logging.info("\tRunning rk4 step 1")
    h2, m2, k1_space, R = euler_step(h1, m1, dt/2, rho_space, eps, return_R=True)

    logging.info("\n\tRunning rk4 step 2")
    _, _, k2_space = euler_step(h2, m2, dt/2, rho_space, eps)
    h3, m3, _ = euler_step(h1, m1, dt/2, rho_space, eps, k_space=k2_space)

    logging.info("\n\tRunning rk4 step 3")
    _, _, k3_space = euler_step(h3, m3, dt/2, rho_space, eps)
    h4, m4, _ = euler_step(h1, m1, dt, rho_space, eps, k_space=k3_space)
    
    logging.info("\n\tRunning rk4 step 4")
    _, _, k4_space = euler_step(h4, m4, dt, rho_space, eps)

    k_space = [(k1 + 2*k2 + 2*k3 + k4)/6 for k1, k2, k3, k4 in zip(k1_space, k2_space, k3_space, k4_space)]
    h_next, m_next, _ = euler_step(h1, m1, dt, rho_space, eps, k_space=k_space)

    logging.info("\n")

    return h_next, m_next, R


tissot_eps = 0.5
tissot_theta_padding = 1
tissot_rho_padding = 0.5
tissot_const = 0.5

# c3 = 0.766
# c5 = -0.091

c3 = 0.021
c5 = 0.598

h(rho) = 1
m(rho) = ((sin(rho) + c3*sin(3*rho) + c5*sin(5*rho))/(1 + 3*c3 + 5*c5))**2

srange = (0, pi)
h = to_spline(h, srange)
m = to_spline(m, srange)
x, y = xy_splines_from_hm(h, m, srange, step_size=0.1)
R, K = hm_to_ricci_tensor(h, m, return_K=True)

if plot_initial_curve:
    xy_plot = parametric_plot((x, y), (x.list()[0][0], x.list()[-1][0]))
    xy_plot.save(path("initial_curve_of_revolution.png"))
if plot_initial_m:
    plot(lambda z: sqrt(m(z)), srange, title="sqrt(m)").save(path("initial_sqrt_m.png"))
if plot_initial_h:
    plot(h, srange, title="h").save(path("initial_h.png"))
if plot_initial_R:
    def R11(rho): return R(rho)[0][0]
    def R22(rho): return R(rho)[1][1]
    plot(R11, srange, title="R11").save(path("initial_R11.png"))
    plot(R22, srange, title="R22").save(path("initial_R22.png"))
if plot_initial_K:
    plot(K, srange, marker=",", linestyle="", title="K").save(path("initial_K.png"))
if plot_initial_tissot:
    _, _, ellipses = tissot(make_g(h, m), vrange=(tissot_eps, pi-tissot_eps), sq_len=1)
    tissot_plot = Graphics()
    tissot_scale = tissot_const / (2 * m(pi/2))
    tissot_plot += sum([ellipse((x, y), k1 * tissot_scale, k2 * tissot_scale, theta, axes=False) for x, y, k1, k2, theta in ellipses])
    tissot_plot.save(path("initial_tissot.png"), xmin=-tissot_theta_padding, xmax=2*pi + tissot_theta_padding, ymin=-tissot_rho_padding, ymax=pi + tissot_rho_padding)


# Ricci flow
dt = 0.0001
N = 3001
plot_gap = 10
reparam_gap = 4
space, dt = np.linspace(0, dt*(N-1), N, retstep=True)
eps = 0.1
drho = 0.1

cm = colormaps.RdYlGn

print("Running ricci flow...")
print(f"c3 = {c3}")
print(f"c5 = {c5}")
print(f"dt = {dt}")
print(f"N = {N}")
print(f"plot_gap = {plot_gap}")

curve_plots = []
revolved_plots = []
m_plots = []
h_plots = []
R11_plots = []
R22_plots = []
tissot_plots = []

for i in range(N):
    try:
        print(f"\nRK4: Iteration {i}/{N-1}, t = {dt*i}")
        h, m, R = rk4_step(h, m, dt, eps=eps, drho=drho)
        
        if i % reparam_gap == 0:
            print("\tReparametrizing...")
            h, m = reparam(h, m)

        if i % plot_gap == 0:
            logging.info("\tGetting x and y splines from h and m splines")
            x, y = xy_splines_from_hm(h, m, srange, step_size=0.1)

            def c(theta, rho, eps=0.1):
                rho = clamp(rho, eps, pi-eps)
                K = y.derivative(rho, order=2) / y(rho)
                sigmoid = 1 / (1 + exp(-K))
                return sigmoid
            
            print("\tAppending plots")
            revolved_plots.append(parametric_plot3d(revolve(x, y), (0, 2*pi), srange, plot_points=[20, 80], color=(c, cm)))
            if animate_curve:
                curve_plots.append(parametric_plot((x, y), srange))
            if animate_m:
                m_plots.append(plot(lambda z: sqrt(m(z)), srange, title="sqrt(m)"))
            if animate_h:
                h_plots.append(plot(h, srange, title="h"))
            if animate_R:
                def R11(rho): return R(rho)[0][0]
                def R22(rho): return R(rho)[1][1]
                R11_plots.append(plot(R11, srange, title="R11"))
                R22_plots.append(plot(R22, srange, title="R22"))
            if animate_tissot:
                tissot_scale = tissot_const / (2 * m(pi/2))
                _, _, ellipses = tissot(make_g(h, m), vrange=(tissot_eps, pi-tissot_eps), sq_len=1)
                tissot_plot = Graphics()
                tissot_plot += sum([ellipse((x, y), k1 * tissot_scale, k2 * tissot_scale, theta, axes=True) for x, y, k1, k2, theta in ellipses])
                tissot_plot.set_axes_range(xmin=-tissot_theta_padding, xmax=2*pi + tissot_theta_padding, ymin=-tissot_rho_padding, ymax=pi + tissot_rho_padding)
                tissot_plots.append(tissot_plot)
    except Exception as e:
        print(f"Encountered exception on iteration {i}/{N-1}, t = {dt*i}:")
        print(repr(e))
        print("Terminating flow.")
        break

if animate_curve:
    print("Animating curve...")
    start = time.time()
    curve_anim = animate(curve_plots)
    end = time.time()
    print(f"Done with animation in {end - start} seconds.")

    print("Saving curve animation...")
    start = time.time()
    curve_anim.save(path("curve_flow.gif"), show_path=True)
    end = time.time()
    print(f"Saved animation in {end - start} seconds.")
if animate_m:
    print("Animating sqrt(m)...")
    start = time.time()
    m_anim = animate(m_plots)
    end = time.time()
    print(f"Done with animation in {end - start} seconds.")

    print("Saving sqrt(m) animation...")
    start = time.time()
    m_anim.save(path("sqrt_m_anim.gif"), show_path=True)
    end = time.time()
    print(f"Saved animation in {end - start} seconds.")
if animate_h:
    print("Animating h...")
    start = time.time()
    h_anim = animate(h_plots)
    end = time.time()
    print(f"Done with animation in {end - start} seconds.")

    print("Saving h animation...")
    start = time.time()
    h_anim.save(path("h_anim.gif"), show_path=True)
    end = time.time()
    print(f"Saved animation in {end - start} seconds.")
if animate_R:
    print("Animating R...")
    start = time.time()
    R11_anim = animate(R11_plots)
    R22_anim = animate(R22_plots)
    end = time.time()
    print(f"Done with animation in {end - start} seconds.")

    print("Saving R animation...")
    start = time.time()
    R11_anim.save(path("R11_anim.gif"), show_path=True)
    R22_anim.save(path("R22_anim.gif"), show_path=True)
    end = time.time()
    print(f"Saved animation in {end - start} seconds.")
if animate_tissot:
    print("Animating Tissot...")
    start = time.time()
    tissot_anim = animate(tissot_plots)
    end = time.time()
    print(f"Done with animation in {end - start} seconds.")

    print("Saving Tissot animation...")
    start = time.time()
    tissot_anim.save(path("tissot_anim.gif"), show_path=True)
    end = time.time()
    print(f"Saved animation in {end - start} seconds.")

print("Animating surface flow...")
start = time.time()
a_surf = animate(revolved_plots)
end = time.time()
print(f"Done with animation in {end - start} seconds.")

print("Saving surface flow...")
start = time.time()
a_surf.save(path("surf_flow.html"), online=True, show_path=True)
end = time.time()
print(f"Saved animation in {end - start} seconds.")
