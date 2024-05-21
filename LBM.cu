#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <string_view>

#define NUMDIR 9
#define NUMDIM 2
#define PI 3.141592654f

__device__ int nx, ny, ly, cx, cy, r;
__device__ float uLB, nulb, omega, Re;

__device__ const int col1[3] = {0, 1, 2};
__device__ const int col2[3] = {3, 4, 5};
__device__ const int col3[3] = {6, 7, 8};

__device__ const int d_v[NUMDIR * NUMDIM] = {1, 1, 1,  0,  1, -1, 0, 1,  0,
                                             0, 0, -1, -1, 1, -1, 0, -1, -1};
__device__ const float d_t[NUMDIR] = {1.0 / 36, 1.0 / 9, 1.0 / 36,
                                      1.0 / 9,  4.0 / 9, 1.0 / 9,
                                      1.0 / 36, 1.0 / 9, 1.0 / 36};

extern __global__ void macroscopic(float *fin, float *rho, float *u);
extern __global__ void equilibrium(float *rho, float *u, float *feq);
extern __global__ void equilibrium_to_fin(float *feq, float *fin);
extern __global__ void obstacle_mask(bool *obstacle);
extern __global__ void right_wall_outflow(float *fin);
extern __global__ void collision(float *fin, float *feq, float *fout);
extern __global__ void left_wall_inflow(float *fin, float *rho, float *u);
extern __global__ void inivel(float *u);
extern __global__ void bounce_back(bool *obstacle, float *fin, float *fout);
extern __global__ void set_nxy(int h_nx, int h_ny, float h_Re);
extern __global__ void streaming(float *fout, float *fin);
extern __global__ void setRhoTo1f(float *rho);
extern __host__ void writeToFile(std::string_view baseName, int number,
                                 const float *d_u, float *h_u, int h_nx,
                                 int h_ny);

int main() {
  const int maxIter = 2000, printInterval = 1000;
  int h_nx = 420, h_ny = 180;
  float h_Re = 100;
  dim3 threadSizingAll(3, 32, 1);
  dim3 threadSizingAllDIM(3, 32, NUMDIM);
  dim3 threadSizingAllDIR(3, 32, NUMDIR);
  dim3 threadSizingJustY(1, 32, 1);
  dim3 threadSizingJustY3(1, 32, 3);
  dim3 blockSizingAll((h_nx - 1) / 3 + 1, (h_ny - 1) / 32 + 1, 1);
  dim3 blockSizingJustY(1, (h_ny - 1) / 32 + 1, 1);
  set_nxy<<<1, 1>>>(h_nx, h_ny, h_Re);
  cudaDeviceSynchronize();
  const int f_size = NUMDIR * h_nx * h_ny * sizeof(float);
  const int u_size = NUMDIM * h_nx * h_ny * sizeof(float);
  const int rho_size = 1 * h_nx * h_ny * sizeof(float);
  const int obs_size = 1 * h_nx * h_ny * sizeof(bool);
  float *d_fin, *d_fout, *d_feq, *d_rho, *d_u, *h_u;
  bool *d_obstacle, *h_obstacle;
  cudaMalloc((bool **)&d_obstacle, obs_size);
  cudaMallocHost((bool **)&h_obstacle, obs_size);
  cudaMalloc((float **)&d_fin, f_size);
  cudaMalloc((float **)&d_fout, f_size);
  cudaMalloc((float **)&d_feq, f_size);
  cudaMalloc((float **)&d_rho, rho_size);
  cudaMalloc((float **)&d_u, u_size);
  cudaMallocHost((float **)&h_u, u_size);
  cudaDeviceSynchronize();

  obstacle_mask<<<blockSizingAll, threadSizingAll>>>(d_obstacle);
  cudaDeviceSynchronize();

  inivel<<<blockSizingAll, threadSizingAllDIM>>>(d_u);
  cudaDeviceSynchronize();

  setRhoTo1f<<<blockSizingAll, threadSizingAll>>>(d_rho);
  cudaDeviceSynchronize();

  equilibrium<<<blockSizingAll, threadSizingAll>>>(d_rho, d_u, d_fin);
  cudaDeviceSynchronize();
  for (int i = 1; i <= maxIter; ++i) {
    // std::cout << "Step: " << i << std::endl;
    right_wall_outflow<<<blockSizingJustY, threadSizingJustY3>>>(d_fin);
    cudaDeviceSynchronize();

    macroscopic<<<blockSizingAll, threadSizingAll>>>(d_fin, d_rho, d_u);
    cudaDeviceSynchronize();

    left_wall_inflow<<<blockSizingJustY, threadSizingJustY>>>(d_fin, d_rho,
                                                              d_u);
    cudaDeviceSynchronize();

    equilibrium<<<blockSizingAll, threadSizingAll>>>(d_rho, d_u, d_feq);
    cudaDeviceSynchronize();

    equilibrium_to_fin<<<blockSizingJustY, threadSizingJustY3>>>(d_feq, d_fin);
    cudaDeviceSynchronize();

    collision<<<blockSizingAll, threadSizingAllDIR>>>(d_fin, d_feq, d_fout);
    cudaDeviceSynchronize();

    bounce_back<<<blockSizingAll, threadSizingAllDIR>>>(d_obstacle, d_fin,
                                                        d_fout);
    cudaDeviceSynchronize();

    streaming<<<blockSizingAll, threadSizingAllDIR>>>(d_fout, d_fin);
    cudaDeviceSynchronize();

    if (i % printInterval == 0) {
      writeToFile("output", i, d_u, h_u, h_nx, h_ny);
    }
  }
  writeToFile("output", 0, d_u, h_u, h_nx, h_ny);

  return 0;
}

__global__ void setRhoTo1f(float *rho) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < nx && y < ny) {
    rho[nx * y + x] = 1.0;
  }
}

__global__ void obstacle_mask(bool *obstacle) {
  // define the shape of the obstacle by creating a boolean mask
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < nx && y < ny) {
    obstacle[nx * y + x] =
        ((x - cx) * (x - cx) + (y - cy) * (y - cy)) < (r * r);
  }
}
__global__ void macroscopic(float *fin, float *rho, float *u) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < nx && y < ny) {
    rho[nx * y + x] = 0;
    u[(nx * y + x) * NUMDIM + 0] = 0;
    u[(nx * y + x) * NUMDIM + 1] = 0;

    for (int i = 0; i < NUMDIR; ++i) {
      rho[nx * y + x] += fin[(nx * y + x) * NUMDIR + i];
      u[(nx * y + x) * NUMDIM + 0] +=
          d_v[i * NUMDIM + 0] * fin[(nx * y + x) * NUMDIR + i];
      u[(nx * y + x) * NUMDIM + 1] +=
          d_v[i * NUMDIM + 1] * fin[(nx * y + x) * NUMDIR + i];
    }

    u[(nx * y + x) * NUMDIM + 0] /= rho[nx * y + x];
    u[(nx * y + x) * NUMDIM + 1] /= rho[nx * y + x];
  }
}

__global__ void equilibrium(float *rho, float *u, float *feq) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < nx && y < ny) {
    const float _rho = rho[nx * y + x];
    const float _u0 = u[(nx * y + x) * NUMDIM + 0];
    const float _u1 = u[(nx * y + x) * NUMDIM + 1];
    const float usqr = 1.5 * (_u0 * _u0 + _u1 * _u1);

    for (int i = 0; i < NUMDIR; ++i) {
      const float cu =
          3 * (d_v[i * NUMDIM + 0] * _u0 + d_v[i * NUMDIM + 1] * _u1);
      feq[(nx * y + x) * NUMDIR + i] =
          _rho * d_t[i] * (1 + cu + 0.5 * cu * cu - usqr);
    }
  }
}

__global__ void right_wall_outflow(float *fin) {
  const int x = nx - 1;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int i = col3[threadIdx.z];

  if (x < nx && y < ny) {
    fin[(nx * y + x) * NUMDIR + i] = fin[(nx * y + x - 1) * NUMDIR + i];
  }
}

__global__ void set_nxy(int h_nx, int h_ny, float h_Re) {
  Re = h_Re;
  nx = h_nx;
  ny = h_ny;
  ly = ny - 1;
  cx = nx / 4;
  cy = ny / 2;
  r = ny / 9;
  uLB = 0.04;
  nulb = uLB * r / Re;
  omega = 1.0 / (3.0 * nulb + 0.5);
}

__global__ void inivel(float *u) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int i = threadIdx.z;

  if (x < nx && y < ny) {
    u[(nx * y + x) * NUMDIM + i] =
        (1 - i) * uLB * (1 + 1e-4 * sin(1. * y / ly * 2.0 * PI));
  }
}

__global__ void left_wall_inflow(float *fin, float *rho, float *u) {
  const int x = 0;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < nx && y < ny) {
    const float _u0 = uLB * (1 + 1e-4 * sin(1.0 * y / ly * 2 * PI));
    const float fincol2 = fin[(nx * y + x) * NUMDIR + col2[0]] +
                          fin[(nx * y + x) * NUMDIR + col2[1]] +
                          fin[(nx * y + x) * NUMDIR + col2[2]];
    const float fincol3 = fin[(nx * y + x) * NUMDIR + col3[0]] +
                          fin[(nx * y + x) * NUMDIR + col3[1]] +
                          fin[(nx * y + x) * NUMDIR + col3[2]];

    u[(nx * y + x) * NUMDIM + 0] = _u0;
    u[(nx * y + x) * NUMDIM + 1] = 0;
    rho[nx * y + x] = 1.0 / (1 - _u0) * (fincol2 + 2 * fincol3);
  }
}

__global__ void equilibrium_to_fin(float *feq, float *fin) {
  const int x = 0;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int i = threadIdx.z;

  if (x < nx && y < ny) {
    fin[(nx * y + x) * NUMDIR + col1[i]] =
        feq[(nx * y + x) * NUMDIR + col1[i]] +
        fin[(nx * y + x) * NUMDIR + col3[2 - i]] -
        feq[(nx * y + x) * NUMDIR + col3[2 - i]];
  }
}

__global__ void collision(float *fin, float *feq, float *fout) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int i = threadIdx.z;
  if (x < nx && y < ny) {
    const int idx = (nx * y + x) * NUMDIR + i;

    fout[idx] = fin[idx] - omega * (fin[idx] - feq[idx]);
  }
}

__global__ void bounce_back(bool *obstacle, float *fin, float *fout) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int i = threadIdx.z;
  if (x < nx && y < ny) {
    const int idx = nx * y + x;

    if (obstacle[idx]) {
      fout[idx * NUMDIR + i] = fin[idx * NUMDIR + 8 - i];
    }
  }
}

__global__ void streaming(float *fout, float *fin) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int i = threadIdx.z;

  if (x < nx && y < ny) {
    int next_x = x + d_v[NUMDIM * i + 0];
    next_x = (next_x < 0) ? (nx - 1) : next_x;
    next_x = (next_x >= nx) ? 0 : next_x;

    int next_y = y + d_v[NUMDIM * i + 1];
    next_y = (next_y < 0) ? (ny - 1) : next_y;
    next_y = (next_y >= ny) ? 0 : next_y;

    const int idx = (nx * y + x) * NUMDIR + i;
    const int next_idx = (nx * next_y + next_x) * NUMDIR + i;

    fin[next_idx] = fout[idx];
  }
}

__host__ void writeToFile(std::string_view baseName, int number,
                          const float *d_u, float *h_u, int h_nx, int h_ny) {

  cudaMemcpy(h_u, d_u, NUMDIM * h_nx * h_ny * sizeof(float),
             cudaMemcpyDeviceToHost);

  std::string fileName =
      std::string(baseName) + "_" + std::to_string(number) + ".txt";

  std::ofstream outputFile(fileName);
  if (!outputFile.is_open()) {
    std::cerr << "Error opening the file " << fileName << std::endl;
    return;
  }

  for (int x = 0; x < h_nx; ++x) {
    for (int y = 0; y < h_ny; ++y) {
      const int idx = (h_nx * y + x) * NUMDIM;
      const float u0 = h_u[idx];
      const float v0 = h_u[idx + 1];
      const float V = std::sqrt(u0 * u0 + v0 * v0);
      outputFile << V << std::endl;
    }
  }

  outputFile.close();
}
