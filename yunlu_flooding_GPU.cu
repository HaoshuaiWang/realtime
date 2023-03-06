#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <string.h>
#include "swmm5.h"
#include <cuda_runtime.h>

#define MAXC(a,b)    ({ __typeof__(a) _a = (a); __typeof__(b) _b = (b); _a > _b ? _a : _b; })
#define MINC(a,b)    ({ __typeof__(a) _a = (a); __typeof__(b) _b = (b); _a < _b ? _a : _b; })

#define       GET_CELL_INDEX(i, j)    (i * (JM + 2) + j)
#define       GET_EDGE_INDEX(i, j, k) (i * (JM + 2) * 4 + j * 4 + k)
#define        GET_CELL_BASE(i, j)    (i * (JM + 2) * 3 + j * 3)
#define   GET_INTERFACE_BASE(i, j, k) ((i + A[k][0]) * (JM + 2) * 2 * 3 + (j + A[k][1]) * 2 * 3 + A[k][2] * X_AXIS * 3 + A[k][3] * Y_AXIS * 3)
#define GET_INTERFACE_BASE_d(i, j, k) ((i + d_A[k][0]) * (JM + 2) * 2 * 3 + (j + d_A[k][1]) * 2 * 3 + d_A[k][2] * X_AXIS * 3 + d_A[k][3] * Y_AXIS * 3)

#define YL_LOGGING_FLAG 0

#define NoData_value -9999

#define X_AXIS 0
#define Y_AXIS 1

#define EAST  0
#define NORTH 1
#define WEST  2
#define SOUTH 3

#define INTERNAL_EDGE         0
#define DRY_BOUNDARY          1
#define SLIP_BOUNDARY         2
#define TRANSMISSIVE_BOUNDARY 3

#define IM 1763
#define JM 2103

#define DX 2.0
#define DY 2.0
#define DA (DX * DY)

#define DW (JM * DX)
#define DH (IM * DY)

#define NM 0.04
#define I0 0.0 //(0.03 / 3600)

#define GRAV  9.81
#define HDRY  0.001
#define DT    0.1
#define TMAX  3600 
#define TSMAX 10//TSMAX根据一维时间步长进行调整，直接给值，如一维时间步长1s，二维时间步长0.1s，TSMAX = delta1d / delta2d = 10
#define LOGGINE_INTERVAL 300

#define CELL_DEM_SIZE   sizeof(double) * (IM + 2) * (JM + 2)
#define CELL_U_SIZE     sizeof(double) * (IM + 2) * (JM + 2) * 3
#define CELL_NM_SIZE    sizeof(double) * (IM + 2) * (JM + 2)
#define CELL_I0_SIZE    sizeof(double) * (IM + 2) * (JM + 2)
#define EDGE_H_SIZE     sizeof(double) * (IM + 2) * (JM + 2) * 4
#define EDGE_TYPE_SIZE  sizeof(   int) * (IM + 2) * (JM + 2) * 4
#define INTERFACES_SIZE sizeof(double) * (IM + 2) * (JM + 2) * 2 * 3
#define UMAX_SIZE       sizeof(double) * 3
#define MRN 2000
int    its;
double t;
double t_start, t_end;

int A[4][4] = {{0, 0, 1, 0}, {0, 0, 0, 1}, {0, -1, 1, 0}, {1, 0, 0, 1}};

int block_size = 512;
int grid_size = (int) ceil((float) (IM * JM) / block_size);

double *CELL_DEM;
double *CELL_U;
double *CELL_NM;
double *CELL_I0;
double *EDGE_H;
int    *EDGE_TYPE;
double *INTERFACES;
double *UMAX;
typedef struct{
    char nodeName[50];
    int  row;
    int  column;
} rel_t;

__device__ int    d_A[4][4] = {{0, 0, 1, 0}, {0, 0, 0, 1}, {0, -1, 1, 0}, {1, 0, 0, 1}};

void log_main_title() {
    printf("==============================================================================\n");
    printf("                                                                              \n");
    printf(" YUNLU_FLOODING: BE THE FASTEST SWE SOLVER EVER                               \n");
    printf("                                                                              \n");
    printf("------------------------------------------------------------------------------\n");
}

void log_end_title(clock_t clock) {
    printf("------------------------------------------------------------------------------\n");
    printf(" TOTAL WALL CLOCKS: %.2lf Seconds\n", clock * 1.0 / CLOCKS_PER_SEC               );
    printf("------------------------------------------------------------------------------\n");
    printf(" BY CHUNSHUI YU & Jennifer Duan, 2010 - 2023, Univ. of Arizona, YUNLU TECH    \n");
    printf("==============================================================================\n");
}

void log_CELL_U(int minute) {
    int i, j;
    FILE *file;
    char filename[100];
    sprintf(filename, "%s_%d%s","Output/CELL_U", minute, ".BIN");
    file = fopen(filename, "wb");
    fwrite(CELL_U, sizeof(double), (IM + 2) * (JM + 2) * 3, file);
    fclose(file);

    //i = int(IM / 2.0);
    //j = int(JM / 2.0);
    printf("CELL_U[%i, %i]: %lf, %lf, %lf\n", 1568, 853, CELL_U[GET_CELL_BASE(1568, 853) + 0], CELL_U[GET_CELL_BASE(1568, 853) + 1], CELL_U[GET_CELL_BASE(1568, 853) + 2]);
    printf("CELL_U[%i, %i]: %lf, %lf, %lf\n", 1241, 1059, CELL_U[GET_CELL_BASE(1241, 1059) + 0], CELL_U[GET_CELL_BASE(1241, 1059) + 1], CELL_U[GET_CELL_BASE(1241, 1059) + 2]);

    printf("CELL_U[%i, %i]: %lf, %lf, %lf\n", 1061, 1452, CELL_U[GET_CELL_BASE(1061, 1452) + 0], CELL_U[GET_CELL_BASE(1061, 1452) + 1], CELL_U[GET_CELL_BASE(1061, 1452) + 2]);
    printf("CELL_U[%i, %i]: %lf, %lf, %lf\n", 1065, 1847, CELL_U[GET_CELL_BASE(1065, 1847) + 0], CELL_U[GET_CELL_BASE(1065, 1847) + 1], CELL_U[GET_CELL_BASE(1065, 1847) + 2]);
}

void log_CUDA_error() {
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGetLastError(): %s\n", cudaGetErrorString(cudaStatus));
    }
}

__device__ double calc_drag_coefficient(double *d_CELL_U, double *d_CELL_NM, int i, int j) {
    int cell_index;
    int base_index;
    double cz;
    double cd;

    cell_index = GET_CELL_INDEX(i, j);
    base_index = GET_CELL_BASE(i, j);

    cz = 0.0;
    cd = 0.0;
    
    if (d_CELL_U[base_index + 0] < HDRY) {
        // dry cell
        cd = 0.0;
    } else if (d_CELL_NM[cell_index] == 0.0) {
        cd = 0.0;
    } else {
        // wet cell
        cz = MAXC(1.0, pow(d_CELL_U[base_index + 0], 0.1667) / d_CELL_NM[cell_index]);
        cd = GRAV * sqrt(pow(d_CELL_U[base_index + 1], 2.0) + pow(d_CELL_U[base_index + 2], 2.0)) / (cz * cz);
    }

    return cd;
}

__device__ void calc_gravity_source(double *d_CELL_U, double *d_EDGE_H, int i, int j, double *sg) {
    int sgi;
    double hp, hm;

    // dry cell
    if (d_CELL_U[GET_CELL_BASE(i, j) + 0] < HDRY) {
        for (sgi = 0; sgi < 2; sgi++) {
            sg[sgi] = 0.0;
        }

        return;
    }

    // wet cell
    // x axis
    hp = d_EDGE_H[GET_EDGE_INDEX(i, j, 0)];
    hm = d_EDGE_H[GET_EDGE_INDEX(i, j, 2)];
    
    sg[0] = 0.5 * GRAV * (hp * hp - hm * hm) / DX;
    
    // y axis
    hp = d_EDGE_H[GET_EDGE_INDEX(i, j, 1)];
    hm = d_EDGE_H[GET_EDGE_INDEX(i, j, 3)];
    
    sg[1] = 0.5 * GRAV * (hp * hp - hm * hm) / DX;  // for square cell only (dx = dy)
}

__device__ void sum_advective_fluxes(double *d_INTERFACES, int i, int j, double *flux) {
    int k, fi, c;
    int interface_base;

    for (fi = 0; fi < 3; fi++) {
        flux[fi] = 0.0;
    }

    for (k = 0; k < 4; k++) {
        interface_base = GET_INTERFACE_BASE_d(i, j, k);
        c = k < 2? 1: -1;

        for (fi = 0; fi < 3; fi++) {
            flux[fi] = flux[fi] + c * d_INTERFACES[interface_base + fi];
        }
    }
}

__global__ void solve_swe(double *d_CELL_U, double *d_INTERFACES, double *d_CELL_DEM, double *d_EDGE_H, double *d_CELL_NM, double *d_CELL_I0) {
    int i, j;
    int fi;
    int cell_index;
    int base_index;
    double dtoa;
    double h;
    double fa[3];
    double u[2];
    double sg[2];
    double cd;

    cell_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell_index >= IM * JM) return;

    i = cell_index / (JM + 2);
    j = cell_index - (JM + 2) * i;
    base_index = GET_CELL_BASE(i, j);

    if (abs(d_CELL_DEM[cell_index] - NoData_value) < DBL_MIN) return;

    dtoa = DT / DA;

    // sum up fluxes
    sum_advective_fluxes(d_INTERFACES, i, j, fa);
    
    // calculate flow depth
    h = d_CELL_U[base_index + 0] - dtoa * fa[0] + DT * d_CELL_I0[cell_index];  // to do CELL_U index

    if (h < 0.0) {
        printf("Negative flow depth occured at: %i, %i\n", i, j);
        return;
    }

    // calculate flow velocities
    if (h < HDRY) {
        // dry cell
        for (fi = 0; fi < 2; fi++) {
            u[fi] = 0.0;
        }
    } else {
        // wet cell
        calc_gravity_source(d_CELL_U, d_EDGE_H, i, j, sg);
        cd = calc_drag_coefficient(d_CELL_U, d_CELL_NM, i, j);

        // semi-implicit scheme
        for (fi = 0; fi < 2; fi++) {
            u[fi] = (d_CELL_U[base_index + 0] * d_CELL_U[base_index + fi + 1] - dtoa * fa[fi + 1] + DT * sg[fi]) / (h + DT * cd);
        }
    }

if (YL_LOGGING_FLAG && i == int(IM / 2.0) && j == int(JM / 2.0)) {
    printf("%lf, %lf, %lf\n", h, u[0], u[1]);
}

    // update variables
    d_CELL_U[base_index + 0] = h;
    d_CELL_U[base_index + 1] = u[0];
    d_CELL_U[base_index + 2] = u[1];
}

__device__ void hll_flux(double *ul, double *ur, int *n, double *flux) {
    int i;
    double vl, vr;
    double ql[3], qr[3];
    double cl, cr;
    double sl, sr;
    double vs, cs;
    double fl[3], fr[3];
    double ptl, ptr;

    // dry interface
    if ((ul[0] < HDRY) && (ur[0] < HDRY)) {
        for (i = 0; i < 3; i++) {
            flux[i] = 0.0;
        }

        return;
    }

    // normal velocity
    vl = ul[1] * n[0] + ul[2] * n[1];
    vr = ur[1] * n[0] + ur[2] * n[1];

    // restore conservative variables
    ql[0] = ul[0];
    ql[1] = ul[0] * ul[1];
    ql[2] = ul[0] * ul[2];
    
    qr[0] = ur[0];
    qr[1] = ur[0] * ur[1];
    qr[2] = ur[0] * ur[2];

    // celerities
    cl = sqrt(GRAV * ql[0]);
    cr = sqrt(GRAV * qr[0]);

    // wavespeeds
    if (ul[0] <= HDRY) {
        // left side is dry bed
        sl = vr - 2.0 * cr;
        sr = vr + cr;
    } else if (ur[0] <= HDRY) {
        // right side is dry bed
        sl = vl - cl;
        sr = vl + 2.0 * cl;
    } else {
        // both sides are wet beds
        // estimate wavespeeds (two rarefaction approximate solver)
        vs = (vl + vr) / 2.0 + (cl - cr);
        cs = (cl + cr) / 2.0 + (vl - vr) / 4.0;
        
        sl = MINC(vl - cl, vs - cs);
        sr = MAXC(vr + cr, vs + cs);
    }

    // the hll flux...
    if (sl >= 0.0) {
        // supercritical flow from left
        ptl = 0.5 * GRAV * ql[0] * ql[0];
        
        fl[0] = ql[0] * vl;
        fl[1] = ql[1] * vl + ptl * n[0];
        fl[2] = ql[2] * vl + ptl * n[1];
        
        for (int i = 0; i < 3; i++) {
            flux[i] = fl[i];
        }
    } else if (sr <= 0.0) {
        // supercritical flow from right
        ptr = 0.5 * GRAV * qr[0] * qr[0];
        
        fr[0] = qr[0] * vr;
        fr[1] = qr[1] * vr + ptr * n[0];
        fr[2] = qr[2] * vr + ptr * n[1];
        
        for (int i = 0; i < 3; i++) {
            flux[i] = fr[i];
        }
    } else {
        // subcritical flow
        // left flux
        ptl = 0.5 * GRAV * ql[0] * ql[0];
        
        fl[0] = ql[0] * vl;
        fl[1] = ql[1] * vl + ptl * n[0];
        fl[2] = ql[2] * vl + ptl * n[1];
        
        // right flux
        ptr = 0.5 * GRAV * qr[0] * qr[0];
        
        fr[0] = qr[0] * vr;
        fr[1] = qr[1] * vr + ptr * n[0];
        fr[2] = qr[2] * vr + ptr * n[1];

        // star flux
        for (i = 0; i < 3; i++) {
            flux[i] = (sr * fl[i] - sl * fr[i] + sl * sr * (qr[i] - ql[i])) / (sr - sl);
        }
    }

    return;
}

__device__ void reconstruct(double *d_CELL_U, int base_index, double eta, double zbi, double *ui) {
    ui[0] = MAXC(0.0, eta - zbi);
    ui[1] = d_CELL_U[base_index + 1];
    ui[2] = d_CELL_U[base_index + 2];
}

__device__ void calc_internal_flux(double *d_CELL_U, double *d_INTERFACES, double *d_CELL_DEM, double *d_EDGE_H, int i, int j, int k) {
    int ri, rj;
    int fi;
    int nnv[2];
    int cell_index, neighbor_index;
    int base_index, neighbor_base;
    int interface_base;
    double zbi;
    double ul[3], ur[3];
    double flux[3];
    
    cell_index = GET_CELL_INDEX(i, j);
    base_index = GET_CELL_BASE(i, j);

    ri = i - k % 2;
    rj = j + 1 - k % 2;
    neighbor_index = GET_CELL_INDEX(ri, rj);
    neighbor_base = GET_CELL_BASE(ri, rj);

    zbi = MAXC(d_CELL_DEM[cell_index], d_CELL_DEM[neighbor_index]);

    reconstruct(d_CELL_U, base_index, d_CELL_DEM[cell_index] + d_CELL_U[base_index + 0], zbi, ul);
    reconstruct(d_CELL_U, neighbor_base, d_CELL_DEM[neighbor_index] + d_CELL_U[neighbor_base + 0], zbi, ur);

    d_EDGE_H[GET_EDGE_INDEX(i, j, k)] = ul[0];
    d_EDGE_H[GET_EDGE_INDEX(ri, rj, k + 2)] = ur[0];

    nnv[0] = 1.0 - (k % 2);
    nnv[1] = k % 2;

    hll_flux(ul, ur, nnv, flux);

    interface_base = GET_INTERFACE_BASE_d(i, j, k);
    for (fi = 0; fi < 3; fi++) {
        d_INTERFACES[interface_base + fi] = flux[fi] * DX;  // for square cell only (dx = dy)
    }
}

__device__ void calc_boundary_flux(double *d_CELL_U, double *d_INTERFACES, double *d_CELL_DEM, int *d_EDGE_TYPE, double *d_EDGE_H, int i, int j, int k) {
    int fi;
    int cell_index, base_index, edge_index, interface_base;
    int nnv[2];
    double b, eta;
    double ul[3], ur[3];
    double flux[3];

    cell_index = GET_CELL_INDEX(i, j);
    base_index = GET_CELL_BASE(i, j);
    edge_index = GET_EDGE_INDEX(i, j, k);
    interface_base = GET_INTERFACE_BASE_d(i, j, k);

    b = d_CELL_DEM[cell_index];
    eta = d_CELL_DEM[cell_index] + d_CELL_U[base_index + 0];

    nnv[0] = 1.0 - (k % 2);
    nnv[1] = k % 2;

    switch (d_EDGE_TYPE[edge_index]) {
        case SLIP_BOUNDARY:
            switch (k) {
                case EAST:
                    reconstruct(d_CELL_U, base_index, eta, b, ul);
                    for (fi = 0; fi < 3; fi++) {
                        ur[fi] = ul[fi];
                    }
                    ur[1] = -ur[1];

                    d_EDGE_H[edge_index] = ul[0];

                    break;
                case NORTH:
                    reconstruct(d_CELL_U, base_index, eta, b, ul);
                    for (fi = 0; fi < 3; fi++) {
                        ur[fi] = ul[fi];
                    }
                    ur[2] = -ur[2];

                    d_EDGE_H[edge_index] = ul[0];

                    break;
                case WEST:
                    reconstruct(d_CELL_U, base_index, eta, b, ur);
                    for (fi = 0; fi < 3; fi++) {
                        ul[fi] = ur[fi];
                    }
                    ul[1] = -ul[1];

                    d_EDGE_H[edge_index] = ur[0];

                    break;
                case SOUTH:
                    reconstruct(d_CELL_U, base_index, eta, b, ur);
                    for (fi = 0; fi < 3; fi++) {
                        ul[fi] = ur[fi];
                    }
                    ul[2] = -ul[2];

                    d_EDGE_H[edge_index] = ur[0];

                    break;
            }

            hll_flux(ul, ur, nnv, flux);

            for (fi = 0; fi < 3; fi++) {
                d_INTERFACES[interface_base + fi] = flux[fi] * DX;  // for square cell only (dx = dy)
            }

            break;
    }
}

__global__ void calculate_riemann_fluxes(double *d_CELL_U, double *d_INTERFACES, double *d_CELL_DEM, int *d_EDGE_TYPE, double *d_EDGE_H) {
    int i, j, k;
    int cell_index;

    cell_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell_index >= IM * JM) return;

    if (abs(d_CELL_DEM[cell_index] - NoData_value) < DBL_MIN) {
        return;
    }

    i = cell_index / (JM + 2);
    j = cell_index - (JM + 2) * i;

    for (k = 0; k < 4; k++) {
        if (d_EDGE_TYPE[GET_EDGE_INDEX(i, j, k)] != INTERNAL_EDGE) {
            calc_boundary_flux(d_CELL_U, d_INTERFACES, d_CELL_DEM, d_EDGE_TYPE, d_EDGE_H, i, j, k);
        } else if (k < WEST) {
            calc_internal_flux(d_CELL_U, d_INTERFACES, d_CELL_DEM, d_EDGE_H, i, j, k);
        }
    }
}

void device_to_host(double *d_CELL_U) {
    cudaMemcpy(CELL_U, d_CELL_U, CELL_U_SIZE, cudaMemcpyDeviceToHost);
}

void host_to_device(double *d_CELL_U, double *d_INTERFACES, double *d_CELL_DEM, double *d_CELL_NM, double *d_CELL_I0, double *d_EDGE_H, int *d_EDGE_TYPE, double *d_UMAX) {
    cudaMemcpy(d_CELL_U,     CELL_U,     CELL_U_SIZE,     cudaMemcpyHostToDevice);
    cudaMemcpy(d_CELL_NM,    CELL_NM,    CELL_NM_SIZE,    cudaMemcpyHostToDevice);
    cudaMemcpy(d_CELL_I0,    CELL_I0,    CELL_I0_SIZE,    cudaMemcpyHostToDevice);
    cudaMemcpy(d_CELL_DEM,   CELL_DEM,   CELL_DEM_SIZE,   cudaMemcpyHostToDevice);
    cudaMemcpy(d_EDGE_H,     EDGE_H,     EDGE_H_SIZE,     cudaMemcpyHostToDevice);
    cudaMemcpy(d_EDGE_TYPE,  EDGE_TYPE,  EDGE_TYPE_SIZE,  cudaMemcpyHostToDevice);
    cudaMemcpy(d_INTERFACES, INTERFACES, INTERFACES_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_UMAX,       UMAX,       UMAX_SIZE,       cudaMemcpyHostToDevice);
}

//NEW ADDING
void height_host_to_device(double *d_CELL_U, double *d_CELL_I0) {
    cudaMemcpy(d_CELL_U, CELL_U, CELL_U_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_CELL_I0, CELL_I0, CELL_I0_SIZE, cudaMemcpyHostToDevice);
}

void set_boundary_conditions() {
    int i, j, k;
    int ni, nj;
    int id, nid;
    double x, y;

    for (i = 1; i < IM + 1; i++) {
        for (j = 1; j < JM + 1; j++) {
            id = GET_CELL_INDEX(i, j);
            if (abs(CELL_DEM[id] - NoData_value) < DBL_MIN) {
                continue;
            }

            for (k = 0; k < 4; k++) {
                ni = i + (k - 2) % 2;
                nj = j + (~(k - 2) % 2);
                nid = GET_CELL_INDEX(ni, nj);

                if (abs(CELL_DEM[nid] - NoData_value) < DBL_MIN) {

                    EDGE_TYPE[GET_EDGE_INDEX(i, j, k)] = SLIP_BOUNDARY;
                }
            }

            x = (j - 0.5) * DX;
            y = (IM - i + 0.5) * DX;

            if ((fabs(x - (DW / 2.0 - DX / 2.0)) < DBL_EPSILON) && (y < (95.0 * DH / 200.0) || y > (170 * DH / 200.0))) {
                EDGE_TYPE[GET_EDGE_INDEX(i, j, EAST)] = SLIP_BOUNDARY;
            }

            if ((fabs(x - (DW / 2.0 + DX / 2.0)) < DBL_EPSILON) && (y < (95 * DH / 200.0) || y > (170 * DH / 200.0))) {
                EDGE_TYPE[GET_EDGE_INDEX(i, j, WEST)] = SLIP_BOUNDARY;
            }
        }
    }
}

void set_initial_conditions() {
    int i, j;
    int base_index, cell_index;
    double x;//, y, z;
    double h;

    for (i = 1; i < IM + 1; i++) {
        for (j = 1; j < JM + 1; j++) {
            cell_index = GET_CELL_INDEX(i, j);
            base_index = GET_CELL_BASE(i, j);

            if (abs(CELL_DEM[cell_index] - NoData_value) < DBL_MIN) {
                continue;
            }

            CELL_U[base_index + 0] = 0.0;
            CELL_U[base_index + 1] = 0.0;
            CELL_U[base_index + 2] = 0.0;
        }
    }
}

void init_interfaces(double **d_INTERFACES) {
	int i, j, k, l;
	int interface_base;
	
	INTERFACES = (double*) malloc(INTERFACES_SIZE);
    if (INTERFACES == NULL) {
		printf("Could not allocate memory for INTERFACES.");
		return;
	}
    cudaMalloc(d_INTERFACES, INTERFACES_SIZE);

	for (i = 0; i < IM + 2; i++) {
		for (j = 0; j < JM + 2; j++) {
            for (k = 0; k < 2; k++) {
                interface_base = GET_INTERFACE_BASE(i, j, k);
                for (l = 0; l < 3; l++) {
                    INTERFACES[interface_base + l] = 0.0;
                }
            }
		}
	}
}

void init_edges(int **d_EDGE_TYPE, double **d_EDGE_H) {
    int i, j, k;

    // init EDGE_TYPE
    EDGE_TYPE = (int*) malloc(EDGE_TYPE_SIZE);
    if (EDGE_TYPE == NULL) {
		printf("Could not allocate memory for EDGE_TYPE.");
		return;
	}
    cudaMalloc(d_EDGE_TYPE, EDGE_TYPE_SIZE);

    for (i = 0; i < IM + 2; i++) {
        for (j = 0; j < JM + 2; j++) {
            for (k = 0; k < 4; k++) {
                EDGE_TYPE[GET_EDGE_INDEX(i, j, k)] = INTERNAL_EDGE;
            }
        }
    }

    // init EDGE_H
    EDGE_H = (double*) malloc(EDGE_H_SIZE);
    if (EDGE_H == NULL) {
		printf("Could not allocate memory for EDGE_H.");
		return;
	}
    cudaMalloc(d_EDGE_H, EDGE_H_SIZE);

    for (i = 0; i < IM + 2; i++) {
        for (j = 0; j < JM + 2; j++) {
            for (k = 0; k < 4; k++) {
                EDGE_H[GET_EDGE_INDEX(i, j, k)] = 0.0;
            }
        }
    }
}

void init_cells(double **d_CELL_U, double **d_CELL_NM, double **d_CELL_I0) {
    int i, j, k;
    int base_index;

    // init CELL_U
    CELL_U = (double*) malloc(CELL_U_SIZE);
    if (CELL_U == NULL) {
		printf("Could not allocate memory for CELL_U.");
		return;
	}
    cudaMalloc(d_CELL_U, CELL_U_SIZE);

    for (i = 0; i < IM + 2; i++) {
        for (j = 0; j < JM + 2; j++) {
            base_index = GET_CELL_BASE(i, j);
            for (k = 0; k < 3; k++) {
                CELL_U[base_index + k] = 0.0;
            }
        }
    }

    // init CELL_NM
    CELL_NM = (double*) malloc(CELL_NM_SIZE);
    if (CELL_NM == NULL) {
		printf("Could not allocate memory for CELL_NM.");
		return;
	}
    cudaMalloc(d_CELL_NM, CELL_NM_SIZE);

    for (i = 0; i < IM + 2; i++) {
        for (j = 0; j < JM + 2; j++) {
            CELL_NM[GET_CELL_INDEX(i, j)] = NM;
        }
    }

    // init CELL_I0
    CELL_I0 = (double*) malloc(CELL_I0_SIZE);
    if (CELL_I0 == NULL) {
		printf("Could not allocate memory for CELL_I0.");
		return;
	}
    cudaMalloc(d_CELL_I0, CELL_I0_SIZE);
	
    for (i = 0; i < IM + 2; i++) {
        for (j = 0; j < JM + 2; j++) {
            CELL_I0[GET_CELL_INDEX(i, j)] = I0;
        }
    }
}

void init_cell_dem(double **d_CELL_DEM) {
	//int i, j;
    //int cell_index;
    
    FILE *file = NULL;
	
	CELL_DEM = (double*) malloc(CELL_DEM_SIZE);
    if (CELL_DEM == NULL) {
		printf("Could not allocate memory for CELL_DEM.");
		return;
	}
    cudaMalloc(d_CELL_DEM, CELL_DEM_SIZE);

    file = fopen("dem/TTY_DEM_2M.BIN", "rb");
    fread(CELL_DEM, sizeof(double), (IM + 2) * (JM + 2), file);
    fclose(file);
    
    /*for (i = 0; i < IM + 2; i++) {
		for (j = 0; j < JM + 2; j++) {
            cell_index = GET_CELL_INDEX(i, j);
            if ((i == 0) || (j == 0) || (i == IM + 1) || (j == JM + 1)) {
                CELL_DEM[cell_index] = NoData_value;
            } else {
                CELL_DEM[cell_index] = 0.0;
            }
		}
	}*/
}

void define_problem(double **d_UMAX) {
	t = 0.0;
    
    UMAX = (double *) malloc(UMAX_SIZE);
    if (UMAX == NULL) {
		printf("Could not allocate memory for UMAX.");
		return;
	}
    
    cudaMalloc(d_UMAX, UMAX_SIZE);
}

void clean_up(double *d_UMAX, double *d_CELL_DEM, double *d_CELL_U, double *d_CELL_NM, double *d_CELL_I0, int *d_EDGE_TYPE, double *d_EDGE_H, double *d_INTERFACES) {
    cudaFree(d_UMAX);
    cudaFree(d_CELL_DEM);
    cudaFree(d_CELL_U);
    cudaFree(d_CELL_NM);
    cudaFree(d_CELL_I0);
    cudaFree(d_EDGE_TYPE);
    cudaFree(d_EDGE_H);
    cudaFree(d_INTERFACES);

    free(UMAX);
    free(CELL_DEM);
    free(CELL_U);
    free(CELL_NM);
    free(CELL_I0);
    free(EDGE_TYPE);
    free(EDGE_H);
    free(INTERFACES);
}

//---------------------------------------------------
//NEW ADDING FOR COUPLING
int read_relation(char* rf, rel_t* relation){
    FILE* fp;
    int   relationCount = 0;
    char  nodeName[50];
    char  relationLine[100];
    int   row, column;
    fp = fopen(rf, "r");
    if ( fp == NULL){
        printf("Can't find relation file: %s\n", rf);
	exit(-1);
    }

    while( fgets(relationLine, 100, fp) != NULL){
        sscanf(relationLine, "%s %d %d", nodeName, &row, &column);
	sscanf(nodeName, "%s", relation[relationCount].nodeName);
	relation[relationCount].row = row;
	relation[relationCount].column = column;
	relationCount++;
    }
    fclose(fp);
    return relationCount;
}

//---------------------------------------------------
//Reset the ground elevation of manhole.
void reset_elev(rel_t* relation, int relationCount){
    int i;
    int row;
    int column;
    int nodeId;
    double dem;
    int cellIndex;
    FILE* fp;
    //将修改后的检查井的最大水深值写入到文件。
    fp = fopen("elevation.dat", "w");

    for( i = 0; i < relationCount; i++){
        nodeId = swmm_getIndex(swmm_NODE, relation[i].nodeName);
	row    = relation[i].row;
	column = relation[i].column;
	cellIndex = GET_CELL_INDEX(row, column);
	dem    = CELL_DEM[cellIndex];
	//printf("relation[i].nodeName: %s row: %d column: %d cellIndex: %d dem: %lf\n", relation[i].nodeName, row, column, cellIndex, dem);
        if ( swmm_getValue(swmm_NODE_TYPE, nodeId) == swmm_JUNCTION){ //可以添加跳过无效高程网格点的程序，使其不参与计算。
	    swmm_setValue(swmm_NODE_MAXDEPTH, nodeId, dem - swmm_getValue(swmm_NODE_ELEV, nodeId));
            //fprintf(fp, "%s %lf\n", relation[i].nodeName, swmm_getValue(swmm_NODE_MAXDEPTH, nodeId));	    
	} 
        	
    }
    printf("Reset manhole ground elevation done!!\n");
    fclose(fp);
}

//---------------------------------------
//Flow transfer between 1D and 2D.
void flow_transfer(int relationCount, rel_t* relation){
    int i,j;
    double head1D;
    double cw = 0.5;
    double omigaw = (0.4 + 0.7) * 2;
    double gravity = 9.8;
    double Qexchange = 0.0;
    double apiflow;
    double head2D, z2D, h2D;
    int    row, column, nodeId;
    for( i = 0; i < relationCount; i++){ //如果有无效网格高程对应的节点，在该节点处也不进行流量交互。计算时需要进行跳过。
        nodeId = swmm_getIndex(swmm_NODE, relation[i].nodeName);
	row    = relation[i].row;
	column = relation[i].column;
	h2D    = CELL_U[GET_CELL_BASE(row, column) + 0];
	z2D    = CELL_DEM[GET_CELL_INDEX(row, column)];
	head1D = swmm_getValue(swmm_NODE_HEAD, nodeId);
	head2D = h2D + z2D;
	/*
        if (abs(z2D - NoData_value) < DBL_MIN) {
                continue;
        }
        */

	if(swmm_getValue(swmm_NODE_TYPE, nodeId) == swmm_JUNCTION){
	    if(head1D >= head2D){
	        Qexchange = cw * omigaw * (head1D - head2D) * sqrt(2 * gravity * (head1D - head2D));
		Qexchange = MINC(Qexchange, swmm_getValue(swmm_NODE_VOLUME, nodeId) / swmm_getValue(swmm_ROUTESTEP, 0)); //swmm_REPORTSTEP
		apiflow = -1.0 * Qexchange;
		CELL_U[GET_CELL_BASE(row, column) + 0] = h2D + Qexchange * swmm_getValue(swmm_ROUTESTEP, 0) / 4;
	    }
	    else if (head1D < head2D && head1D > z2D){
	        Qexchange = cw * omigaw * (head2D - head1D) * sqrt(2 * gravity * (head2D - head1D));
		Qexchange = MINC(Qexchange, h2D * 4 / swmm_getValue(swmm_ROUTESTEP, 0));
		apiflow = Qexchange;
		CELL_U[GET_CELL_BASE(row, column) + 0] = h2D - Qexchange * swmm_getValue(swmm_ROUTESTEP, 0) / 4;
	    } else if (head1D <= z2D){
	        if( h2D > 0.001){
		    Qexchange = cw * omigaw * h2D * sqrt(2 * gravity * h2D);
		    Qexchange = MINC(Qexchange, h2D * 4 / swmm_getValue(swmm_ROUTESTEP, 0));
		    apiflow = Qexchange;
		    CELL_U[GET_CELL_BASE(row, column) + 0] = h2D - Qexchange * swmm_getValue(swmm_ROUTESTEP, 0) / 4;
	        } 
		else{
		    Qexchange = apiflow = 0.0;
		    CELL_U[GET_CELL_BASE(row, column) + 0] = h2D;
		}

	    } 	
	
	}
	else if( swmm_getValue(swmm_NODE_TYPE, nodeId) == swmm_OUTFALL){
	    for( j = 0; j < swmm_getCount(swmm_LINK); j++){
	        if( swmm_getValue(swmm_LINK_NODE1, j) == nodeId || swmm_getValue(swmm_LINK_NODE2, j) == nodeId){
		    Qexchange = swmm_getValue(swmm_LINK_FLOW, j);
		    CELL_I0[GET_CELL_INDEX(row, column)] = Qexchange / 4; //可能会出现回流量过大，导致二维网格水深出现负值的情况。
		    CELL_U[GET_CELL_BASE(row, column) + 0] = h2D + Qexchange * DT / 4;
		    //如果通过计算得到的网格深度小于零则把网格深度置为零。
		    if( CELL_U[GET_CELL_BASE(row, column) + 0] < 0.0)
			    CELL_U[GET_CELL_BASE(row, column) + 0] = 0.0;
		}
	    }
	    swmm_setValue(swmm_NODE_HEAD, nodeId, CELL_U[GET_CELL_BASE(row, column) + 0] + z2D);
	    //将二维网格水位不是水深赋值给出口节点，流量通过水位在下一时间步进行交互。不需额外对apiflow进行修改。
	    apiflow = 0.0;
	    CELL_U[GET_CELL_BASE(row, column) + 0] = h2D; //将二维网格水深重置为交换前的水深。在下次迭代时开始增加交互水深。
	}
	swmm_setValue(swmm_NODE_LATFLOW, nodeId, apiflow);
	printf("nodeName:%s, row: %d , column: %d, head1D: %lf, head2D: %lf , h2D: %lf, apiflow: %lf\n", relation[i].nodeName, row, column, head1D, head2D, h2D, apiflow);
    }
    printf("Flow transfer process done!\n");//TEST CODE
}

//---------------------------------------
//流量交互之后，一维管网的外部入流量应重置为0.0.使用接口函数进行的流量交互变量apiflow，在不进行流量交互时应置为0.0.
void apiflow_reset(int relationCount, rel_t* relation){
    int i;
    for ( i = 0; i < relationCount; i++){
        swmm_setValue(swmm_NODE_LATFLOW, swmm_getIndex(swmm_NODE, relation[i].nodeName), 0.0);
    
    }
}


//---------------------------------------
//读取实时降雨数据
void set_rain(char* rainFile){
    FILE*  fp;
    char   gageName[100];
    double rainRate;
    char   rainLine[100];
    
    fp = fopen(rainFile, "r");
    if(fp == NULL){
        printf("Can't open rainfile: %s\n", rainFile);
	exit(-1);
    }
    while(fgets(rainLine, 100, fp) != NULL){
        sscanf(rainLine, "%s %lf", gageName, &rainRate);
	swmm_setValue(swmm_GAGE_RAINFALL, swmm_getIndex(swmm_GAGE, gageName), rainRate);
    }
    fclose(fp);
    printf("Reading rainfile %s done!\n",rainFile);
}


//---------------------------------------
int main(int argc, char *argv[]) {
    double *d_CELL_DEM;
    double *d_CELL_U;
    double *d_CELL_NM;
    double *d_CELL_I0;
    double *d_EDGE_H;
    int    *d_EDGE_TYPE;
    double *d_INTERFACES;
    double *d_UMAX;
    double elapsedTime = 0;
    rel_t  relation[MRN];
    int    relationCount;
    char   rainFile[100];
    swmm_open(argv[1], argv[2], argv[3]);
    swmm_start(1);
    relationCount = read_relation(argv[4], relation);//读取关系文件
    log_main_title();

    define_problem(&d_UMAX);
    init_cell_dem(&d_CELL_DEM);
    init_cells(&d_CELL_U, &d_CELL_NM, &d_CELL_I0);
    init_edges(&d_EDGE_TYPE, &d_EDGE_H);
    init_interfaces(&d_INTERFACES);
    
    set_initial_conditions();
    set_boundary_conditions();
    reset_elev(relation,relationCount);//重置检查井地表高程

    cudaSetDevice(0);
    host_to_device(d_CELL_U, d_INTERFACES, d_CELL_DEM, d_CELL_NM, d_CELL_I0, d_EDGE_H, d_EDGE_TYPE, d_UMAX);

    t_start = clock();
    sprintf(rainFile, "%s", "RainData/rain_0.dat");
    set_rain(rainFile);
    
    do{
	    swmm_step(&elapsedTime);
	    //printf("Node tz151 apiExtInflow is: %lf\n", swmm_getValue(swmm_NODE_LATFLOW, swmm_getIndex(swmm_NODE, "tz151")));
	    for (its = 1; its <= TSMAX; its++) { //TSMAX->10 TSMAX在当前算例中设置为10，一维时间步长为1s，即一维运行一个时间步，二维需要运行10个时间步。
		calculate_riemann_fluxes<<<grid_size, block_size>>>(d_CELL_U, d_INTERFACES, d_CELL_DEM, d_EDGE_TYPE, d_EDGE_H);
		solve_swe<<<grid_size, block_size>>>(d_CELL_U, d_INTERFACES, d_CELL_DEM, d_EDGE_H, d_CELL_NM, d_CELL_I0);

		t = t + DT;
		//if (t >= TMAX) break;
	    }
	    //if(进行同步) 先将水深结果从Gpu拉回cpu交互完成后，再将cpu中数组推送到Gpu
	    if ( fabs(swmm_getValue(swmm_ELAPSEDTIME, 0) * 24 * 3600 - swmm_getValue(swmm_REPORTSTEP, 0) * swmm_getValue(swmm_TOTALSTEPS, 0)) < 0.001){
	        device_to_host(d_CELL_U);
                //每隔6min从文件中读取一次降雨数据。
	        if ( (int)swmm_getValue(swmm_TOTALSTEPS, 0) % 6 == 0){
	            sprintf(rainFile, "%s_%d%s", \
		                "RainData/rain", \
			         (int)swmm_getValue(swmm_TOTALSTEPS, 0), \
			        ".dat");
		    set_rain(rainFile);
		    log_CELL_U((int)swmm_getValue(swmm_TOTALSTEPS, 0));
	        }
	        flow_transfer(relationCount, relation);
		height_host_to_device(d_CELL_U, d_CELL_I0);
		printf("Time is: %lfs\n", swmm_getValue(swmm_ELAPSEDTIME, 0) * 24 * 3600);
	    }
	    else
	    {
	        apiflow_reset(relationCount, relation);
	    }
    } while ( elapsedTime > 0 );
    swmm_end();
    swmm_report();
    swmm_close();

    t_end = clock();
    
    device_to_host(d_CELL_U);
    cudaDeviceReset();

    log_CUDA_error();
    //log_CELL_U();

    log_end_title(t_end - t_start);
    clean_up(d_UMAX, d_CELL_DEM, d_CELL_U, d_CELL_NM, d_CELL_I0, d_EDGE_TYPE, d_EDGE_H, d_INTERFACES);

    return EXIT_SUCCESS;
}


