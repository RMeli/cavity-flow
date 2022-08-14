#include <array>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>

#ifdef _OPENMP
#include "omp.h"
#endif

namespace cst{

constexpr size_t Nx = 101;
constexpr size_t Ny = Nx;
constexpr size_t N = Nx * Ny;

constexpr size_t Lx = 2;
constexpr size_t Ly = Lx;

constexpr auto dx = static_cast<double>(Lx) / (Nx - 1);
constexpr auto dy = static_cast<double>(Ly) / (Ny - 1);

constexpr auto Nt = 50;
constexpr auto iters = 7000;
constexpr auto dt = 0.0001;

constexpr auto rho = 1.0;
constexpr auto nu = 0.1;

constexpr auto dx2 = dx * dx;
constexpr auto dy2 = dy * dy;
constexpr auto dx2dy2 = 2.0 * (dx2 + dy2);
constexpr auto dtDdx = dt / dx;
constexpr auto dtDdy = dt/ dy;

} // cst

using Grid = std::array<double, cst::Nx * cst::Ny>;

namespace util{

__host__ __device__
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
        outf << "# Lx " << cst::Lx << "\n# Ly " << cst::Ly << '\n';
        outf << "# Nx " << cst::Nx << "\n# Ny " << cst::Ny << '\n'; 

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

inline void cuda_check_status(cudaError_t status){
    if(status != cudaSuccess){
        std::cout << "CUDA Error: " << cudaGetErrorString(status) << std::endl;
        exit(-1);
    }
}

inline void cuda_check_last_error(){
    auto status = cudaGetLastError();
    if(status != cudaSuccess){
        std::cout << "CUDA Error:" << cudaGetErrorString(status) << std::endl;
        exit(-1);
    }
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
    
    for(size_t iy = 0; iy < cst::Ny; iy++){
        auto x_left = util::index(iy, 0);
        auto x_right = util::index(iy, cst::Nx - 1);
        u_new[x_left] = 0.0;
        u_new[x_right] = 0.0;
        v_new[x_left] = 0.0;
        v_new[x_right] = 0.0;
    }

    for(size_t ix = 0; ix < cst::Nx; ix++){
        auto y_top = util::index(0, ix);
        auto y_bottom = util::index(cst::Ny - 1, ix);
        u_new[y_bottom] = 1.0;
        u_new[y_top] = 0.0;
        v_new[y_bottom] = 0.0;
        v_new[y_top] = 0.0;
    }
}

void run(){
    Grid u, v, p, u_new, v_new;

    auto ti = std::chrono::high_resolution_clock::now();
    
    for(size_t idx = 0; idx < cst::N; idx++){
        u[idx] = 0.0;
        v[idx] = 0.0;
        p[idx] = 0.0;
        u_new[idx] = 0.0;
        v_new[idx] = 0.0;
    }

    for(size_t it = 0; it < cst::iters; it++){
        step(u_new, v_new, p, u, v);

        std::copy(std::cbegin(u_new), std::cend(u_new), std::begin(u));
        std::copy(std::cbegin(v_new), std::cend(v_new), std::begin(v));
    }

    auto tf = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(tf - ti);
    std::cout << "serial::run[1]\t\t" << duration.count() << " ms\n";

    util::write("p_serial.dat", p);
    util::write("u_serial.dat", u);
    util::write("v_serial.dat", v);
}

} // serial

namespace omp{

void pressure(Grid& p, const Grid& u, const Grid& v){
    Grid c;

    #pragma omp parallel for
    for(size_t iy = 1; iy < cst::Ny - 1; iy++){ // y runs across rows
        #pragma omp simd
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

        #pragma omp parallel
        {
            #pragma omp for simd
            for(size_t j = 0; j < cst::N; j++){
                pold[j] = p[j];
            }

            #pragma omp for
            for(size_t iy = 1; iy < cst::Ny - 1; iy++){ // y runs across rows
                #pragma omp simd
                for(size_t ix = 1; ix < cst::Nx - 1; ix++){ // x runs across columns
                    auto [idx, idx_right, idx_left, idx_up, idx_down] = util::multiindex(iy, ix);

                    p[idx] = (
                        (pold[idx_right] + pold[idx_left]) * cst::dy2 +
                        (pold[idx_up] + pold[idx_down]) * cst::dx2
                        ) / cst::dx2dy2 + c[idx];
                }
            }

            #pragma omp for simd nowait
            for (size_t iy = 0; iy < cst::Ny; iy++) {
                p[util::index(iy, 0)] = p[util::index(iy, 1)];
                p[util::index(iy, cst::Nx - 1)] = p[util::index(iy, cst::Nx - 2)];
            }

            #pragma omp for simd
            for (size_t ix = 0; ix < cst::Nx; ix++) {
                p[util::index(0, ix)] = p[util::index(1, ix)];
                p[util::index(cst::Ny - 1, ix)] = 0.0;
            }
        }
    }
}

void step(Grid& u_new, Grid& v_new, Grid& p, const Grid& u, const Grid& v){
    pressure(p, u, v);

    #pragma omp parallel
    {
        #pragma omp for
        for (size_t iy = 1; iy < cst::Ny - 1; iy++) {   // y runs across rows
            #pragma omp simd
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

        #pragma omp for simd nowait
        for(size_t iy = 0; iy < cst::Ny; iy++){
            auto x_left = util::index(iy, 0);
            auto x_right = util::index(iy, cst::Nx - 1);
            u_new[x_left] = 0.0;
            u_new[x_right] = 0.0;
            v_new[x_left] = 0.0;
            v_new[x_right] = 0.0;
        }

        #pragma omp for simd nowait
        for(size_t ix = 0; ix < cst::Nx; ix++){
            auto y_top = util::index(0, ix);
            auto y_bottom = util::index(cst::Ny - 1, ix);
            u_new[y_bottom] = 1.0;
            u_new[y_top] = 0.0;
            v_new[y_bottom] = 0.0;
            v_new[y_top] = 0.0;
        }
    }
}

void run(){
    Grid u, v, p, u_new, v_new;

    int n_omp_threads = -1;
#ifdef _OPENMP
    #pragma omp parallel
    {
        n_omp_threads = omp_get_num_threads();
    }
#endif

    auto ti = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for simd
    for(size_t idx = 0; idx < cst::N; idx++){
        u[idx] = 0.0;
        v[idx] = 0.0;
        p[idx] = 0.0;
        u_new[idx] = 0.0;
        v_new[idx] = 0.0;
    }

    for(size_t it = 0; it < cst::iters; it++){
        step(u_new, v_new, p, u, v);

        #pragma omp parallel for simd
        for(size_t i = 0; i < cst::N; i++){
            u[i] = u_new[i];
            v[i] = v_new[i];
        }
    }

    auto tf = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(tf - ti);
    std::cout << "omp::run[" << n_omp_threads << "]\t\t" << duration.count() << " ms\n";
    
    util::write("p_omp.dat", p);
    util::write("u_omp.dat", u);
    util::write("v_omp.dat", v);
}

} // omp

namespace cu{

__device__
void pressure(double* p, const double* u, const double* v, double* pold, double* c){
    auto ix = threadIdx.x + blockDim.x * blockIdx.x;
    auto iy = threadIdx.y + blockDim.y * blockIdx.y;

    auto idx = util::index(iy, ix);
    auto idx_right = util::index(iy, ix + 1);
    auto idx_left = util::index(iy, ix - 1 );
    auto idx_up = util::index(iy + 1, ix);
    auto idx_down = util::index(iy - 1, ix);

    if(iy > 0 and iy < cst::Ny - 1 and ix > 0 and ix < cst::Nx - 1){
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
    __syncthreads();

    for(size_t i = 0; i < cst::Nt; i++){
        if(ix < cst::Nx and iy < cst::Ny){
            pold[idx] = p[idx];
        }
        __syncthreads();

        if(iy > 0 and iy < cst::Ny - 1 and ix > 0 and ix < cst::Nx - 1){
            p[idx] = (
                (pold[idx_right] + pold[idx_left]) * cst::dy2 +
                (pold[idx_up] + pold[idx_down]) * cst::dx2
                ) / cst::dx2dy2 + c[idx];
        }

        if(ix == 0 and iy < cst::Ny){
            p[util::index(iy, 0)] = p[util::index(iy, 1)];
        }
        if(ix == cst::Nx - 1 and iy < cst::Ny){
            p[util::index(iy, cst::Nx - 1)] = p[util::index(iy, cst::Nx - 2)];
        }

        if(iy == 0 and ix < cst::Nx){
            p[util::index(0, ix)] = p[util::index(1, ix)];
        }
        if(iy == cst::Ny - 1 and ix < cst::Nx){
            p[util::index(cst::Ny - 1, ix)] = 0.0;
        }

        __syncthreads();
    }
}

__global__ void step(double *u_new, double *v_new, double *p, const double *u, const double *v, double *pold, double *c)
{
    pressure(p, u, v, pold, c);
    
    auto ix = threadIdx.x + blockDim.x * blockIdx.x;
    auto iy = threadIdx.y + blockDim.y * blockIdx.y;

    auto idx = util::index(iy, ix);
    auto idx_right = util::index(iy, ix + 1);
    auto idx_left = util::index(iy, ix - 1 );
    auto idx_up = util::index(iy + 1, ix);
    auto idx_down = util::index(iy - 1, ix);

    if(iy > 0 and iy < cst::Ny - 1 and ix > 0 and ix < cst::Nx - 1){
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

    if(ix == 0 and iy < cst::Ny){
        auto x_left = util::index(iy, 0);
        u_new[x_left] = 0.0;
        v_new[x_left] = 0.0;
    }
    if(ix == cst::Nx - 1 and cst::Ny){
        auto x_right = util::index(iy, cst::Nx - 1);
        u_new[x_right] = 0.0;
        v_new[x_right] = 0.0;
    }
    if(iy == 0 and ix < cst::Nx){
        auto y_top = util::index(0, ix);
        u_new[y_top] = 0.0;
        v_new[y_top] = 0.0;
    }
    if(iy == cst::Ny - 1 and ix < cst::Nx){
        auto y_bottom = util::index(cst::Ny - 1, ix);
        u_new[y_bottom] = 1.0;
        v_new[y_bottom] = 0.0;
    }
}


void run(){
    Grid u, v, u_new, v_new, p;
    double *u_dev, *v_dev, *p_dev, *u_new_dev, *v_new_dev, *pold_dev, *c_dev;

    cudaDeviceSynchronize();
    auto ti = std::chrono::high_resolution_clock::now();

    auto n_bytes =  cst::N * sizeof(double);

    util::cuda_check_status(cudaMalloc(&u_dev, n_bytes));
    util::cuda_check_status(cudaMalloc(&v_dev, n_bytes));
    util::cuda_check_status(cudaMalloc(&p_dev, n_bytes));
    util::cuda_check_status(cudaMalloc(&u_new_dev, n_bytes));
    util::cuda_check_status(cudaMalloc(&v_new_dev, n_bytes));
    util::cuda_check_status(cudaMalloc(&pold_dev, n_bytes));
    util::cuda_check_status(cudaMalloc(&c_dev, n_bytes));
    
    #pragma omp parallel for simd
    for(size_t idx = 0; idx < cst::N; idx++){
        u[idx] = 0.0;
        v[idx] = 0.0;
        p[idx] = 0.0;
        u_new[idx] = 0.0;
        v_new[idx] = 0.0;
    }

    util::cuda_check_status(cudaMemcpy(u_dev, u.data(), n_bytes, cudaMemcpyHostToDevice));
    util::cuda_check_status(cudaMemcpy(v_dev, v.data(), n_bytes, cudaMemcpyHostToDevice));
    util::cuda_check_status(cudaMemcpy(u_new_dev, u_new.data(), n_bytes, cudaMemcpyHostToDevice));
    util::cuda_check_status(cudaMemcpy(v_new_dev, v_new.data(), n_bytes, cudaMemcpyHostToDevice));

    constexpr size_t d = 32;
    dim3 dimGrid(cst::Nx / d + 1, cst::Ny / d + 1);
    dim3 dimBlock(d, d);

    for(size_t it = 0; it < cst::iters; it++){
        step<<<dimGrid,dimBlock>>>(u_new_dev, v_new_dev, p_dev, u_dev, v_dev, pold_dev, c_dev);
        util::cuda_check_last_error();

        util::cuda_check_status(cudaMemcpy(u_dev, u_new_dev, n_bytes, cudaMemcpyDeviceToDevice));
        util::cuda_check_status(cudaMemcpy(v_dev, v_new_dev, n_bytes, cudaMemcpyDeviceToDevice));
    }

    util::cuda_check_status(cudaMemcpy(u.data(), u_dev, n_bytes, cudaMemcpyDeviceToHost));
    util::cuda_check_status(cudaMemcpy(v.data(), v_dev, n_bytes, cudaMemcpyDeviceToHost));
    util::cuda_check_status(cudaMemcpy(p.data(), p_dev, n_bytes, cudaMemcpyDeviceToHost));

    cudaFree(u_dev);
    cudaFree(v_dev);
    cudaFree(p_dev);
    cudaFree(u_new_dev);
    cudaFree(v_new_dev);
    cudaFree(pold_dev);
    cudaFree(c_dev);

    cudaDeviceSynchronize();
    auto tf = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(tf - ti);
    std::cout << "cu::run[("
        << dimGrid.x  << ',' << dimGrid.y << "),("
        << dimBlock.x << ',' << dimBlock.y
        << ")]\t" << duration.count() << " ms\n";
    
    util::write("p_cu.dat", p);
    util::write("u_cu.dat", u);
    util::write("v_cu.dat", v);
}

} // cu

int main(){
    std::cout << "+++ Cavity Flow +++" << '\n';
    std::cout << "Lx = " << cst::Lx << '\t' << "Ly = " << cst::Ly << '\n';
    std::cout << "Nx = " << cst::Nx << '\t' << "Ny = " << cst::Ny << '\n';
    std::cout << "dx = " << cst::dx << '\t' << "dy = " << cst::dy << '\n';
    std::cout << "dt = " << cst::dt << '\n';
    std::cout << "+++ ----------- +++\n\n";

    serial::run();
    omp::run();
    cu::run();

    return 0;
}