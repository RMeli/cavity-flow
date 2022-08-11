#include <array>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>

namespace cst{

constexpr size_t Nx = 41;
constexpr size_t Ny = Nx;
constexpr size_t N = Nx * Ny;

constexpr size_t Lx = 2;
constexpr size_t Ly = Lx;

constexpr auto dx = static_cast<double>(Lx) / (Nx - 1);
constexpr auto dy = static_cast<double>(Ly) / (Ny - 1);

constexpr auto Nt = 50;
constexpr auto iters = 700;
constexpr auto dt = 0.001;

constexpr auto rho = 1.0;
constexpr auto nu = 0.1;

auto dx2 = dx * dx;
auto dy2 = dy * dy;
auto dx2dy2 = 2.0 * (dx2 + dy2);
auto dtDdx = dt / dx;
auto dtDdy = dt/ dy;

} // cst

using Grid = std::array<double, cst::Nx * cst::Ny>;

namespace util{

inline size_t index(size_t i, size_t j) {
    return i * cst::Nx + j;
}

inline auto multiindex(size_t i, size_t j){
    return std::make_tuple(
        index(i, j),
        index(i, j + 1),
        index(i, j - 1),
        index(i + 1, j),
        index(i - 1, j)
    );
}

void write(const std::string& fname, const Grid& g){
    std::ofstream outf{fname};
    if(outf.is_open()){
        for(size_t iy = 0; iy < cst::Ny; iy++){
            for(size_t ix = 0; ix < cst::Ny; ix++){
                auto idx = index(iy, ix);
                outf << std::scientific << std::showpos << std::setprecision(5) << g[idx] << ' ';
            }
            outf << '\n';
        }
    }
    outf.close();
}

} // util

namespace serial{

void pressure(Grid& p, const Grid& u, const Grid& v){
    Grid c;

    for(size_t iy = 1; iy < cst::Ny - 1; iy++){ // y runs across rows
        for(size_t ix = 1; ix < cst::Nx - 1; ix++){ // x runs across columns
            auto [idx, idx_right, idx_left, idx_up, idx_down] = util::multiindex(iy, ix);
            
            auto du_x = (u[idx_right] - u[idx_left]) / (2.0 * cst::dx);
            auto du_y = (u[idx_up] - u[idx_down]) / (2.0 * cst::dy);
            auto dv_x = (v[idx_right] - v[idx_left]) / (2.0 * cst::dx);
            auto dv_y = (v[idx_up] - v[idx_down]) / (2.0 * cst::dy); 
            
            c[idx] = - cst::rho * cst::dx2 * cst::dy2 / cst::dx2dy2 *
                    (
                        (du_x + dv_y) / cst::dt
                        - du_x * du_x - 2.0 * du_y * dv_x - dv_y * dv_y
                    );
        }
    }

    for(size_t i = 0; i < cst::Nt; i++){
        Grid pold;

        for(size_t j = 0; j < cst::N; j++){
            pold[j] = p[j];
        }

        for(size_t iy = 1; iy < cst::Ny - 1; iy++){ // y runs across rows
            for(size_t ix = 1; ix < cst::Nx - 1; ix++){ // x runs across columns
                auto [idx, idx_right, idx_left, idx_up, idx_down] = util::multiindex(iy, ix);

                p[idx] = (
                    (pold[idx_right] + pold[idx_left]) * cst::dy2 +
                    (pold[idx_up] + pold[idx_down]) * cst::dx2
                    ) / cst::dx2dy2 + c[idx];
            }
        }

        for (size_t iy = 0; iy < cst::Ny; iy++) {
            p[util::index(iy, 0)] = p[util::index(iy, 1)];
            p[util::index(iy, cst::Nx - 1)] = p[util::index(iy, cst::Nx - 2)];
        }

        for (size_t ix = 0; ix < cst::Nx; ix++) {
            p[util::index(0, ix)] = p[util::index(1, ix)];
            p[util::index(cst::Ny - 1, ix)] = 0.0;
        }
    }
}

void step(Grid& u_new, Grid& v_new, Grid& p, const Grid& u, const Grid& v){
    pressure(p, u, v);

    for (size_t iy = 1; iy < cst::Ny - 1; iy++) {   // y runs across rows
        for (size_t ix = 1; ix < cst::Nx - 1; ix++) { // x runs across columns
            auto [idx, idx_right, idx_left, idx_up, idx_down] =
                util::multiindex(iy, ix);

            u_new[idx] =
                u[idx] - u[idx] * cst::dt / cst::dx * (u[idx] - u[idx_left]) -
                v[idx] * cst::dt / cst::dy * (u[idx] - u[idx_down]) -
                cst::dtDdx / (2.0 * cst::rho) * (p[idx_right] - p[idx_left]) +
                cst::nu * cst::dtDdx / cst::dx *
                    (u[idx_right] - 2.0 * u[idx] + u[idx_left]) +
                cst::nu * cst::dtDdy / cst::dy *
                    (u[idx_up] - 2.0 * u[idx] + u[idx_down]);

            v_new[idx] = v[idx] -
                        u[idx] * cst::dt / cst::dx * (v[idx] - v[idx_left]) -
                        v[idx] * cst::dt / cst::dy * (v[idx] - v[idx_down]) -
                        cst::dtDdy / (2.0 * cst::rho) * (p[idx_up] - p[idx_down]) +
                        cst::nu * cst::dtDdx / cst::dx *
                            (v[idx_right] - 2.0 * v[idx] + v[idx_left]) +
                        cst::nu * cst::dtDdy / cst::dy *
                            (v[idx_up] - 2.0 * v[idx] + v[idx_down]);
        }
    }
}

void boundary(Grid& u, Grid& v){
    for(size_t iy = 0; iy < cst::Ny; iy++){
        auto x_left = util::index(iy, 0);
        auto x_right = util::index(iy, cst::Nx - 1);
        u[x_left] = 0.0;
        u[x_right] = 0.0;
        v[x_left] = 0.0;
        v[x_right] = 0.0;
    }

    for(size_t ix = 0; ix < cst::Nx; ix++){
        auto y_top = util::index(0, ix);
        auto y_bottom = util::index(cst::Ny - 1, ix);
        u[y_bottom] = 1.0;
        u[y_top] = 0.0;
        v[y_bottom] = 0.0;
        v[y_top] = 0.0;
    }
}

void run(){
    Grid u, v, p, u_new, v_new;

    for(size_t idx = 0; idx < cst::N; idx++){
        u[idx] = 0.0;
        v[idx] = 0.0;
        p[idx] = 0.0;
        u_new[idx] = 0.0;
        v_new[idx] = 0.0;
    }

    for(size_t it = 0; it < cst::iters; it++){
        serial::step(u_new, v_new, p, u, v);
        serial::boundary(u_new, v_new);

        std::copy(std::cbegin(u_new), std::cend(u_new), std::begin(u));
        std::copy(std::cbegin(v_new), std::cend(v_new), std::begin(v));
    }

    util::write("p.dat", p);
    util::write("u.dat", u);
    util::write("v.dat", v);
}

} // serial

int main(){
    std::cout << "+++ CFD +++" << '\n';
    std::cout << "Lx = " << cst::Lx << '\n' << "Ly = " << cst::Ly << '\n';
    std::cout << "Nx = " << cst::Nx << '\n' << "Ny = " << cst::Ny << '\n';
    std::cout << "dx = " << cst::dx << '\n' << "dy = " << cst::dy << '\n';
    std::cout << "dt = " << cst::dt << '\n';
    std::cout << "+++ +++ +++\n\n";

    auto ti = std::chrono::high_resolution_clock::now();
    serial::run();
    auto tf = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(tf - ti);
    std::cout << "serial::run\t\t" << duration.count() << " ms\n";

    return 0;
}