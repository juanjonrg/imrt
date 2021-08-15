#include <stdio.h> 
#include <stdlib.h> 
#include <dirent.h> 
#include <signal.h>

//#include "mkl.h"

#include <cuda_runtime.h>
#include <cusparse.h>

static volatile int running = 1;

void interrupt_handler(int signal) {
    running = 0;
}

#define cudaCheck(result) __cudaCheck(result, __FILE__, __LINE__)
inline cudaError_t __cudaCheck(cudaError_t result, const char *file, const int line, bool abort = true) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s[%i]: %s\n", file, line, cudaGetErrorString(result));
        if (abort) {
            exit(result);
        }
    }
    return result;
}

const char* cusparseGetErrorString(cusparseStatus_t result) {
    switch (result) {
        case CUSPARSE_STATUS_SUCCESS:
            return "CUSPARSE_STATUS_SUCCESS";
        case CUSPARSE_STATUS_NOT_INITIALIZED:
            return "CUSPARSE_STATUS_NOT_INITIALIZED";
        case CUSPARSE_STATUS_ALLOC_FAILED:
            return "CUSPARSE_STATUS_ALLOC_FAILED";
        case CUSPARSE_STATUS_INVALID_VALUE:
            return "CUSPARSE_STATUS_INVALID_VALUE";
        case CUSPARSE_STATUS_ARCH_MISMATCH:
            return "CUSPARSE_STATUS_ARCH_MISMATCH";
        case CUSPARSE_STATUS_MAPPING_ERROR:
            return "CUSPARSE_STATUS_MAPPING_ERROR";
        case CUSPARSE_STATUS_EXECUTION_FAILED:
            return "CUSPARSE_STATUS_EXECUTION_FAILED";
        case CUSPARSE_STATUS_INTERNAL_ERROR:
            return "CUSPARSE_STATUS_INTERNAL_ERROR";
        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
        case CUSPARSE_STATUS_ZERO_PIVOT:
            return "CUSPARSE_STATUS_ZERO_PIVOT";
    }
    return "CUSPARSE_UNKNOWN_ERROR";
}

#define cusparseCheck(result) __cusparseCheck(result, __FILE__, __LINE__)
inline cusparseStatus_t __cusparseCheck(cusparseStatus_t result, const char *file, const int line, bool abort = true) {
    if (result != CUSPARSE_STATUS_SUCCESS) {
        fprintf(stderr, "CUSPARSE error at %s[%i]: %s\n", file, line, cusparseGetErrorString(result));
        if (abort) {
            exit(result);
        }
    }
    return result;
}

double get_time_ms() {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0) {
        return ts.tv_sec * 1000 + ts.tv_nsec / 1000000.0;
    } else {
        return 0;
    }
}

double get_time_s() {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0) {
        return ts.tv_sec + ts.tv_nsec / 1e9;
    } else {
        return 0;
    }
}

int compare_strings(const void *va, const void *vb) {

    char **a = (char **) va;
    char **b = (char **) vb;
    return strcmp(*a, *b);
}

int read_files(const char *path, const char *pattern, char **files) {

    int n_files = 0;
    DIR *d = opendir(path);
    struct dirent *dir;
    if (d) {
        while ((dir = readdir(d)) != NULL) {
            if (strstr(dir->d_name, pattern) != NULL) {
                files[n_files] = (char *) malloc(1000 * sizeof(char));
                strcpy(files[n_files], path);
                if (path[strlen(path) - 1] != '/') {
                    strcat(files[n_files], (char *) "/");
                }
                strcat(files[n_files], dir->d_name);
                n_files++;
            }
        }
        closedir(d);
    }

    qsort(files, n_files, sizeof(char *), compare_strings);
    return n_files;
}

struct SparseMatrix {
    int n_nz;
    int n_rows;
    int n_cols;
    int *rows;
    int *cols;
    double *vals;
    int *d_rows;
    int *d_cols;
    double *d_vals;
    int *d_rows_t;
    int *d_cols_t;
    double *d_vals_t;
    cusparseHandle_t handle;
    cusparseMatDescr_t descr;

    cusparseSpMatDescr_t sp_descr;
    cusparseSpMatDescr_t sp_descr_t;

    void malloc_cpu() {
        rows = (int *) malloc(n_nz*sizeof(int));
        cols = (int *) malloc(n_nz*sizeof(int));
        vals = (double *) malloc(n_nz*sizeof(double));
        n_rows = 0;
        n_cols = 0;
    }

    void free_cpu() {
        free(rows);
        free(cols);
        free(vals);
    }
    
    void free_gpu() {
        cudaCheck(cudaFree(d_rows));
        cudaCheck(cudaFree(d_cols));
        cudaCheck(cudaFree(d_vals));
        cudaCheck(cudaFree(d_rows_t));
        cudaCheck(cudaFree(d_cols_t));
        cudaCheck(cudaFree(d_vals_t));
    }

    void copy_to_gpu() {
        cudaCheck(cudaMalloc(&d_rows, n_nz*sizeof(int)));
        cudaCheck(cudaMalloc(&d_cols, n_nz*sizeof(int)));
        cudaCheck(cudaMalloc(&d_vals, n_nz*sizeof(double)));
        cudaCheck(cudaMalloc(&d_rows_t, (n_cols+1)*sizeof(int)));
        cudaCheck(cudaMalloc(&d_cols_t, n_nz*sizeof(int)));
        cudaCheck(cudaMalloc(&d_vals_t, n_nz*sizeof(double)));
        cudaCheck(cudaMemcpy(d_rows, rows, n_nz*sizeof(int), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(d_cols, cols, n_nz*sizeof(int), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(d_vals, vals, n_nz*sizeof(double), cudaMemcpyHostToDevice));
    }

    void sort_by_row() {
        int *p;
        void *p_buffer;
        size_t p_buffer_size = 0;
        double *sorted_vals;

        cudaCheck(cudaMalloc(&sorted_vals, n_nz*sizeof(double)));
        cusparseCheck(cusparseXcsrsort_bufferSizeExt(handle, n_rows, n_cols, n_nz, d_rows, d_cols, &p_buffer_size));
        cudaCheck(cudaMalloc(&p, n_nz*sizeof(int)));
        cudaCheck(cudaMalloc(&p_buffer, p_buffer_size*sizeof(char)));
        cusparseCheck(cusparseCreateIdentityPermutation(handle, n_nz, p));
        cusparseCheck(cusparseXcoosortByRow(handle, n_rows, n_cols, n_nz, d_rows, d_cols, p, p_buffer));
        cusparseCheck(cusparseDgthr(handle, n_nz, d_vals, sorted_vals, p, CUSPARSE_INDEX_BASE_ZERO));
    
        cudaCheck(cudaFree(d_vals));
        cudaCheck(cudaFree(p));
        cudaCheck(cudaFree(p_buffer));
        d_vals = sorted_vals;
    }
    
    void coo_to_csr() {
        int *csr;
        cudaCheck(cudaMalloc(&csr, (n_rows+1)*sizeof(int)));
        cusparseCheck(cusparseXcoo2csr(handle, d_rows, n_nz, n_rows, csr, CUSPARSE_INDEX_BASE_ZERO));
        cudaCheck(cudaFree(d_rows));
        d_rows = csr;
	
	cusparseCheck(cusparseCreateCsr(&sp_descr, n_rows, n_cols, n_nz, d_rows, d_cols, d_vals,
		CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    }

    void transpose_csr() {
        void *p_buffer;
        size_t p_buffer_size = 0;

        cusparseCheck(cusparseCsr2cscEx2_bufferSize(handle, n_rows, n_cols, n_nz, d_vals, d_rows, d_cols, 
            d_vals_t, d_rows_t, d_cols_t, CUDA_R_64F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, 
	    CUSPARSE_CSR2CSC_ALG2, &p_buffer_size));
        cudaCheck(cudaMalloc(&p_buffer, p_buffer_size*sizeof(char)));
        cusparseCheck(cusparseCsr2cscEx2(handle, n_rows, n_cols, n_nz, d_vals, d_rows, d_cols, 
            d_vals_t, d_rows_t, d_cols_t, CUDA_R_64F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, 
	    CUSPARSE_CSR2CSC_ALG2, p_buffer));
        cudaCheck(cudaFree(p_buffer));

	cusparseCheck(cusparseCreateCsr(&sp_descr_t, n_cols, n_rows, n_nz, d_rows_t, d_cols_t, d_vals_t,
		CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    }

    void setup_gpu() {
        cusparseCheck(cusparseCreate(&handle));
        cusparseCheck(cusparseCreateMatDescr(&descr));
        cusparseCheck(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
        cusparseCheck(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));

        copy_to_gpu();
        sort_by_row();
        coo_to_csr();
        transpose_csr();
    }
};

struct Region {
    char *name;
    int n_voxels;
    double min;
    double max;
    double avg;

    double f; // Objective function evaluation
    double eud;
    double dF_dEUD;
    double sum_alpha;
    // Virtual EUD to control PTV overdosage
    // Hardcoded to eud + 1 for now
    double v_f;
    double v_eud;
    double v_dF_dEUD;
    double v_sum_alpha;

    bool is_optimized;
    bool is_ptv;
    double pr_min;
    double pr_max;
    double pr_avg_min;
    double pr_avg_max;
    double *grad_avg;
    
    double pr_eud;
    int penalty;
    int alpha;

    void set_targets(bool t_ptv, double t_min, double t_avg_min, double t_avg_max, double t_max, 
                     double t_eud, int t_alpha, int t_penalty) {
        if (t_eud < 0 && t_min < 0 && t_max < 0 && 
            t_avg_min < 0 && t_avg_max < 0) {
            is_optimized = false;
        } else {
            is_optimized = true;
            is_ptv = t_ptv;
            pr_min = t_min;
            pr_max = t_max;
            pr_avg_min = t_avg_min;
            pr_avg_max = t_avg_max;
            pr_eud = t_eud;
            alpha = t_alpha;
            penalty = t_penalty;
            f = 0;
            v_f = 0;
            eud = 0;
            v_eud = 0;
            dF_dEUD = 0;
            v_dF_dEUD = 0;
            sum_alpha = 0;
            v_sum_alpha = 0;
        }
    }
};

__inline__ __device__ float warpReduceMin(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        float v = __shfl_down_sync(0xffffffff, val, offset);
        if (v < val) {
            val = v;
        }
    }
    return val;
}

__inline__ __device__ float warpReduceMax(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        float v = __shfl_down_sync(0xffffffff, val, offset);
        if (v > val) {
            val = v;
        }
    }
    return val;
}

template <class T>
__inline__ __device__ T warpReduceSum(T val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <class T>
__inline__ __device__ T blockReduce(T val, T (*warp_reduction)(T), T defval) {
    __shared__ T shared[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    val = warp_reduction(val);

    // Write reduced value to shared memory
    if (lane == 0) { 
        shared[wid] = val;
    }
    __syncthreads();

    // Ensure we only grab a value from shared memory if that warp existed
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : defval;
    if (wid == 0) {
        val = warp_reduction(val);
    }

    return val;
}

__global__ void stats_gpu(double *doses, char *voxel_regions, int n_regions, int n_voxels, 
        Region *regions) {
    int rid = blockIdx.x;

    float min = 1e10;
    float max = 0;
    float sum = 0;
    double eud = 0;
    double v_eud = 0;
    double sum_alpha = 0;
    double v_sum_alpha = 0;
    for (int i = threadIdx.x; i < n_voxels; i += blockDim.x) {
        if (voxel_regions[rid*n_voxels + i]) {
            float dose = doses[i];
            if (dose < min) {
                min = dose;
            } 
            if (dose > max) {
                max = dose;
            }
            sum += dose;
            if (dose > 0) {
                sum_alpha += pow((double) dose, (double) regions[rid].alpha);
                if (regions[rid].is_ptv) {
                    v_sum_alpha += pow((double) dose, (double) -regions[rid].alpha);
                }
            }
        }
    }

    min = blockReduce<float>(min, warpReduceMin, 1e10);
    max = blockReduce<float>(max, warpReduceMax, 0);
    sum = blockReduce<float>(sum, warpReduceSum, 0);
    sum_alpha = blockReduce<double>(sum_alpha, warpReduceSum, 0);
    if (regions[rid].is_ptv) {
        v_sum_alpha = blockReduce<double>(v_sum_alpha, warpReduceSum, 0);
    }

    if (threadIdx.x == 0) {
        regions[rid].min = min;
        regions[rid].max = max;
        regions[rid].avg = sum / regions[rid].n_voxels;

        eud = pow(sum_alpha/regions[rid].n_voxels, 1.0/regions[rid].alpha);
        regions[rid].sum_alpha = sum_alpha;
        regions[rid].eud = eud;
        if (regions[rid].is_ptv) {
            v_eud = pow(v_sum_alpha/regions[rid].n_voxels, 1.0/-regions[rid].alpha);
            regions[rid].v_sum_alpha = v_sum_alpha;
            regions[rid].v_eud = v_eud;
        }
        
        if (regions[rid].is_optimized) {
            int n = regions[rid].penalty;
            int pd = regions[rid].pr_eud;
            int v_pd = pd + 1; // Hardcoded virtual PTV prescribed dose
            if (regions[rid].is_ptv) {
                regions[rid].f = 1/(1 + pow(pd/eud, n));
                regions[rid].dF_dEUD =  (n*regions[rid].f/eud) * pow(pd/eud, n);
                // Virtual EUD to control PTV over-dosage
                regions[rid].v_f = 1/(1 + pow(v_eud/v_pd, n));
                regions[rid].v_dF_dEUD = -(n*regions[rid].v_f/v_eud) * pow(v_eud/v_pd, n);
            } else {
                regions[rid].f = 1/(1 + pow(eud/pd, n));
                regions[rid].dF_dEUD = -(n*regions[rid].f/eud) * pow(eud/pd, n);
            }
        }
    }
}

__global__ void init_fluence(double *fluence, int n_beamlets, double value) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (idx < n_beamlets) {
        fluence[idx] = value;
    }
}

__global__ void scale_doses(double *doses, int n_voxels, double dose_grid_scaling) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (idx < n_voxels) {
        doses[idx] *= dose_grid_scaling;
    }
}

__global__ void apply_gradients_OLD(double *gradients, double *momentum, int n_beamlets, int n_gradients, float step, double *fluence) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int beta = 0.9;
    
    if (idx < n_beamlets) {
        double gradient = 0;
        for (int i = 0; i < n_gradients; i++) {
            gradient += gradients[i*n_beamlets + idx];
        }
        
        momentum[idx] = beta*momentum[idx] + (1-beta)*gradient;
        fluence[idx] += step*momentum[idx];

        if (fluence[idx] < 0) {
            fluence[idx] = 0;
        }
        if (fluence[idx] > 1) {
            fluence[idx] = 1;
        }
    }
}

//__global__ void apply_gradients(double *gradients, double *momentum, int n_beamlets, int n_gradients, float step, double *fluence) {
//    int idx = blockIdx.x*blockDim.x + threadIdx.x;
//    int beta = 0.9;
//    
//    if (idx < n_beamlets) {
//        double gradient = 0;
//        for (int i = 0; i < n_gradients; i++) {
//            momentum[i*n_beamlets + idx] = beta*momentum[i*n_beamlets + idx] + (1-beta)*gradients[i*n_beamlets + idx];
//            gradient += momentum[i*n_beamlets + idx];
//        }
//        
//        fluence[idx] += step*gradient;
//
//        if (fluence[idx] < 0) {
//            fluence[idx] = 0;
//        }
//        if (fluence[idx] > 0.2) {
//            fluence[idx] = 0.2;
//        }
//    }
//}

__global__ void apply_gradients(double *gradients, double *momentum, int n_beamlets, int n_gradients, float step, double *fluence) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int beta = 0.9;
    
    if (idx < n_beamlets) {
        momentum[idx] = beta*momentum[idx] + (1-beta)*gradients[idx];
        fluence[idx] += step*momentum[idx];

        // Implementación muy shitty
        //double threshold = 0.02;
        //if (idx > 0) {
        //    double diff = fluence[idx] - fluence[idx-1];
        //    if (diff > threshold) {
        //        fluence[idx] -= 0.01;
        //    } 
        //}
        //if (idx < n_beamlets - 1) {
        //    double diff = fluence[idx] - fluence[idx+1];
        //    if (diff > threshold) {
        //        fluence[idx] -= 0.01;
        //    } 
        //}

        // Smoothing cutre :)
        //double w = 1e-1;
        //if (idx > 0 && idx < n_beamlets - 1) {
        //    double neigh_avg = (fluence[idx - 1] + fluence[idx] + fluence[idx + 1])/3;
        //    if (fluence[idx] > neigh_avg) {
        //        fluence[idx] -= w*abs(fluence[idx] - neigh_avg);
        //    }
        //    if (fluence[idx] < neigh_avg) {
        //        fluence[idx] += w*abs(fluence[idx] - neigh_avg);
        //    }
        //}

        if (fluence[idx] < 0) {
            fluence[idx] = 0;
        }
        if (fluence[idx] > 0.3) {
            fluence[idx] = 0.3;
        }
    }
}

#define BEAM_MAP_X 120
#define BEAM_MAP_Y 120

struct Plan {
    char *name;
    int n_beams;
    int n_beamlets;
    int *n_beamlets_beam;
    int n_voxels;
    int n_regions;
    double dose_grid_scaling;
    Region* regions;
    Region* d_regions;
    char *voxel_regions;
    char *d_voxel_regions;
    SparseMatrix spm;
    double *fluence;
    double *smoothed_fluence;
    double *doses;
    double *d_fluence;
    double *d_doses;
    char *files[100];
    int *beam_maps;
    int *d_beam_maps;

    cusparseDnVecDescr_t fluence_descr;
    cusparseDnVecDescr_t doses_descr;

    void check_line(int result) {
        if (result < 0) {
            fprintf(stderr, "ERROR in %s (%s:%d): Unable to read line.\n", 
                    __func__, __FILE__, __LINE__);
            exit(EXIT_FAILURE);
        }
    }

    char* get_str(char *line, size_t len) {
        char *temp = (char *) malloc(len*sizeof(char));
        snprintf(temp, len, "%s", line);
        temp[strcspn(temp, "\r\n")] = 0; // Remove newline
        return temp;
    }

    int get_int(char *line, char **end) {
        return strtoll(line, end, 10);
    }

    float get_float(char *line, char **end) {
        return strtof(line, end);
    }

    void parse_config(const char *path) {
        int n_files = read_files(path, "m_", files);

        FILE *f = fopen(files[0], "r");
        if (f == NULL) {
            fprintf(stderr, "ERROR in %s (%s:%d): Unable to open file.\n", 
                    __func__, __FILE__, __LINE__);
            exit(EXIT_FAILURE);
        }
        printf("Loading %s...\n", files[0]);

        char *line = NULL;
        char *end = NULL;
        size_t len = 0;

        check_line(getline(&line, &len, f));
        name = get_str(line, len);

        check_line(getline(&line, &len, f));
        n_beams = get_int(line, &end);

        n_beamlets = 0;
        n_beamlets_beam = (int *)malloc(n_beams * sizeof(int));
        for (int i = 0; i < n_beams; i++) {
            check_line(getline(&line, &len, f));
            int index = get_int(line, &end);
            int beamlets = get_int(end, &line);
            n_beamlets_beam[index - 1] = beamlets;
            n_beamlets += beamlets;
        }

        check_line(getline(&line, &len, f));
        n_voxels = get_int(line, &end);

        check_line(getline(&line, &len, f));
        dose_grid_scaling = get_float(line, &end);

        check_line(getline(&line, &len, f));
        n_regions = get_int(line, &end);

        regions = (Region *) malloc(n_regions*sizeof(Region));
        for (int i = 0; i < n_regions; i++) {
            check_line(getline(&line, &len, f));
            get_int(line, &end);
            char *name = get_str(end + 1, len);
            regions[i].name = name;
            regions[i].n_voxels = 0;
        }

        line = NULL;
        len = 0;
        while (getline(&line, &len, f) != -1) {
            fprintf(stderr, "[WARNING] Line not processed: %s", line);
        }

        fclose(f);
        free(files[0]);

        cudaCheck(cudaMalloc(&d_regions, n_regions*sizeof(Region)));
    }

    void parse_voxel_regions(const char *path) {
        int n_files = read_files(path, "v_", files);

        FILE *f = fopen(files[0], "r");
        if (f == NULL) {
            fprintf(stderr, "ERROR in %s (%s:%d): Unable to open file.\n",
                    __func__, __FILE__, __LINE__);
            exit(EXIT_FAILURE);
        }
        printf("Loading %s...\n", files[0]);

        voxel_regions = (char *) malloc(n_voxels*n_regions*sizeof(char));
        char line[1024];
        int num = 0;
        int offset = 0;
        while (fgets(line, sizeof line, f)) {
            if (sscanf(line, "%d", &num)) {
                for (int i = 0; i < n_regions; i++) {
                    voxel_regions[offset + i*n_voxels] = num & 1;
                    num >>= 1;
                }
                offset++;
            } else {
                fprintf(stderr, "ERROR in %s (%s:%d): Unable to read voxel regions.\n",
                        __func__, __FILE__, __LINE__);
                exit(EXIT_FAILURE);
            }
        }

        for (int i = 0; i < n_regions; i++) {
            for (int j = 0; j < n_voxels; j++) {
                if (voxel_regions[i*n_voxels + j]) {
                    regions[i].n_voxels += 1;
                }
            }
        }

        fclose(f);
        free(files[0]);

        cudaCheck(cudaMalloc(&d_voxel_regions, n_voxels*n_regions*sizeof(char)));
        cudaCheck(cudaMemcpy(d_voxel_regions, voxel_regions, n_voxels*n_regions*sizeof(char), cudaMemcpyHostToDevice));
    }

    void load_spm(const char *path) {
        int n_files = read_files(path, "d_", files);

        FILE **fp = (FILE **) malloc(n_files*sizeof(FILE *));
        for (int i = 0; i < n_files; i++) {
            fp[i] = fopen(files[i], "r");
            if (fp[i] == NULL) {
                fprintf(stderr, "ERROR in %s (%s:%d): Unable to open file.\n",
                        __func__, __FILE__, __LINE__);
                exit(EXIT_FAILURE);
            }
            int n_nz = 0;
            int count = fscanf(fp[i], "%d", &n_nz);
            spm.n_nz += n_nz;
        }

        spm.malloc_cpu();

        int idx = 0;
        int offset = 0;
        for (int i = 0; i < n_files; i++) {
            printf("Loading %s... ", files[i]);

            int n_read = 0;
            while (true) {
                int row, col;
                double val;
                int count = fscanf(fp[i], "%d %d %lf", &row, &col, &val);
                if(count == EOF || !count) {
                    break;
                }

                int new_col = offset + col;
                spm.rows[idx] = row;
                spm.cols[idx] = new_col;
                spm.vals[idx] = val;
                idx++;
                n_read++;

                if (row > spm.n_rows) {
                    spm.n_rows = row;
                }
                if (new_col > spm.n_cols) {
                    spm.n_cols = new_col;
                }
            }

            printf("%d values read.\n", n_read);
            offset = spm.n_cols + 1;
            fclose(fp[i]);
            free(files[i]);
        }

        spm.n_rows++;
        // Sometimes there's missing voxels,
        // but we want the dimensions to match for SpMM
        if (spm.n_rows < n_voxels) {
            spm.n_rows = n_voxels;
        }
        spm.n_cols++;

        free(fp);
    
    }

    void load_fluence(const char *path, const char *prefix) {
        int n_files = read_files(path, prefix, files);

        FILE **fp = (FILE **) malloc(n_files*sizeof(FILE *));
        int idx = 0;
        for (int i = 0; i < n_files; i++) {
            printf("Loading %s... ", files[i]);
            fp[i] = fopen(files[i], "r");
            if (fp[i] == NULL) {
                fprintf(stderr, "ERROR in %s (%s:%d): Unable to open file.\n",
                        __func__, __FILE__, __LINE__);
                exit(EXIT_FAILURE);
            }

            int n_read = 0;
            while (true) {
                int count = fscanf(fp[i], "%lf", &(fluence[idx]));
                if(count == EOF || !count) {
                    break;
                }

                idx++;
                n_read++;
            }

            printf("%d values read.\n", n_read);
            fclose(fp[i]);
            free(files[i]);
        }

        cudaCheck(cudaMemcpy(d_fluence, fluence, n_beamlets*sizeof(double), cudaMemcpyHostToDevice));

        free(fp);
    }

    void load_coords(const char *path) {
        int n_files = read_files(path, "xcoords_", files);

        beam_maps = (int *) malloc(n_beams*BEAM_MAP_Y*BEAM_MAP_X*sizeof(int));

        for (int i = 0; i < n_beams*BEAM_MAP_Y*BEAM_MAP_X; i++) {
            beam_maps[i] = -1;
        }

        int idx = 0;
        FILE **fp = (FILE **) malloc(n_files*sizeof(FILE *));
        for (int i = 0; i < n_files; i++) {
            fp[i] = fopen(files[i], "r");
            if (fp[i] == NULL) {
                fprintf(stderr, "ERROR in %s (%s:%d): Unable to open file.\n",
                        __func__, __FILE__, __LINE__);
                exit(EXIT_FAILURE);
            }
            printf("Loading %s... ", files[i]);

            char ignored[1024];
            int n_read = 0;
            while (true) {
                int col, row;
                int count = fscanf(fp[i], "%d %d", &col, &row);
                if(count == EOF) {
                    break;
                } else if (!count) {
                    fgets(ignored, sizeof(ignored), fp[i]);
                } else if (count == 1) {
                    // Header values, ignored
                    continue;
                } else if (count == 2) {
                    beam_maps[i*BEAM_MAP_Y*BEAM_MAP_X + col*BEAM_MAP_Y + row] = idx;
                    n_read++;
                    idx++;
                } else {
                    fprintf(stderr, "ERROR in %s (%s:%d): While reading coordinate file.\n",
                            __func__, __FILE__, __LINE__);
                    exit(EXIT_FAILURE);
                }
            }
            printf("%d coordinates read.\n", n_read);
            fclose(fp[i]);
            free(files[i]);
        }

        //cudaCheck(cudaMemcpy(d_beam_maps, beam_maps, 
        //          n_beams*BEAM_MAP_Y*BEAM_MAP_X*sizeof(int), cudaMemcpyHostToDevice));

        free(fp);
    }

    void init_fluence(float value) {
        for (int i = 0; i < n_beamlets; i++) {
            fluence[i] = value;
        }
        cudaCheck(cudaMemcpy(d_fluence, fluence, n_beamlets*sizeof(double), cudaMemcpyHostToDevice));
    }


    void print() {
        printf("Name: %s\n", name);
        printf("Number of beams: %d\n", n_beams);
        for (int i = 0; i < n_beams; i++) {
            printf("  Beam %d: %d beamlets\n", i + 1, n_beamlets_beam[i]);
        }
        printf("Total: %d beamlets\n", n_beamlets);
        printf("Number of voxels: %d\n", n_voxels);
        printf("Dose Grid Scaling: %e\n", dose_grid_scaling);
        printf("Number of regions: %d\n", n_regions);
        for (int i = 0; i < n_regions; i++) {
            printf("  Region %2d (%4d): %-16s %8d voxels\n", i, (int) pow(2, i), regions[i].name, regions[i].n_voxels);
        }
        printf("Dose matrix: %d x %d with %d nonzeros.\n", spm.n_rows, spm.n_cols, spm.n_nz);
    }

    void compute_dose() {
        //cudaCheck(cudaMemcpy(d_fluence, fluence, n_beamlets*sizeof(double), cudaMemcpyHostToDevice));
        //memset(doses, 0, n_voxels*sizeof(*doses));
        cudaCheck(cudaMemset(d_doses, 0, n_voxels*sizeof(*d_doses)));

        //cusparseCheck(cusparseDcsrmv(spm.handle, CUSPARSE_OPERATION_NON_TRANSPOSE, spm.n_rows, spm.n_cols, spm.n_nz, &alpha, spm.descr, spm.d_vals, spm.d_rows, spm.d_cols, d_fluence, &beta, d_doses));
        double alpha = 1.0, beta = 0.0;
        void *p_buffer;
        size_t p_buffer_size = 0;
	cusparseCheck(cusparseSpMV_bufferSize(spm.handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, spm.sp_descr, fluence_descr, &beta,
		doses_descr, CUDA_R_64F, CUSPARSE_CSRMV_ALG1, &p_buffer_size));
        cudaCheck(cudaMalloc(&p_buffer, p_buffer_size*sizeof(char)));
	cusparseCheck(cusparseSpMV(spm.handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, spm.sp_descr, fluence_descr, &beta,
		doses_descr, CUDA_R_64F, CUSPARSE_CSRMV_ALG1, p_buffer));

        int block = 512;
        int grid = (n_voxels + block - 1)/block;
        scale_doses<<<grid, block>>>(d_doses, n_voxels, dose_grid_scaling);

        //cudaCheck(cudaMemcpy(doses, d_doses, n_voxels*sizeof(double), cudaMemcpyDeviceToHost));
    }

    void stats() {
        //cudaCheck(cudaMemcpy(d_regions, regions, n_regions*sizeof(Region), cudaMemcpyHostToDevice));
        stats_gpu<<<n_regions, 512>>>(d_doses, d_voxel_regions, n_regions, n_voxels, d_regions);
        //cudaCheck(cudaMemcpy(regions, d_regions, n_regions*sizeof(Region), cudaMemcpyDeviceToHost));
    }

    void print_table() {
        printf("      Region          Min         Avg         Max         EUD     dF_dEUD         v_EUD     v_dF_dEUD      f       v_f\n"); 
        for (int i = 0; i < n_regions; i++) {
            if (regions[i].is_optimized) {
                printf("%-15s %11.6lf %11.6lf %11.6lf %11.6lf %11.6lf %11.6lf %11.6lf %11.6lf %11.6lf\n", regions[i].name, regions[i].min, regions[i].avg, regions[i].max, regions[i].eud, regions[i].dF_dEUD, regions[i].v_eud, regions[i].v_dF_dEUD, regions[i].f, regions[i].v_f);
            }
        }
    }

    void load(const char *plan_path, const char *fluence_path, const char *fluence_prefix) {
        parse_config(plan_path);
        parse_voxel_regions(plan_path);
        load_spm(plan_path);
        spm.setup_gpu();

        fluence = (double *) malloc(n_beamlets*sizeof(double));
        smoothed_fluence = (double *) malloc(n_beamlets*sizeof(double));
        cudaCheck(cudaMalloc(&d_fluence, n_beamlets*sizeof(double)));
        doses = (double *) malloc(n_voxels*sizeof(double));
        cudaCheck(cudaMalloc(&d_doses, n_voxels*sizeof(double)));

        load_coords(plan_path);

        load_fluence(fluence_path, fluence_prefix);
        //init_fluence(1e-2);
        print();

        cusparseCreateDnVec(&fluence_descr, n_beamlets, d_fluence, CUDA_R_64F);
        cusparseCreateDnVec(&doses_descr, n_voxels, d_doses, CUDA_R_64F);
    }

    void smooth_cpu() {
        int n_neighbors = 8;
        int sum_weights = 1000;
        int *neighbors = (int *) malloc(n_neighbors*sizeof(int));

        for (int i = 0; i < n_beams; i++) {
            for (int y = 0; y < BEAM_MAP_Y; y++) {
                for (int x = 0; x < BEAM_MAP_X; x++) {
                    int offset = i*BEAM_MAP_Y*BEAM_MAP_X;
                    int idx = beam_maps[offset + BEAM_MAP_Y*y + x];
                    float center_weight = sum_weights - n_neighbors;
                    if (idx >= 0) {
                        smoothed_fluence[idx] = 0;
                        neighbors[0] = beam_maps[offset + BEAM_MAP_Y*(y-1) + (x-1)];
                        neighbors[1] = beam_maps[offset + BEAM_MAP_Y*(y-1) + (x  )];
                        neighbors[2] = beam_maps[offset + BEAM_MAP_Y*(y-1) + (x+1)];
                        neighbors[3] = beam_maps[offset + BEAM_MAP_Y*(y  ) + (x-1)];
                        neighbors[4] = beam_maps[offset + BEAM_MAP_Y*(y  ) + (x+1)];
                        neighbors[5] = beam_maps[offset + BEAM_MAP_Y*(y+1) + (x-1)];
                        neighbors[6] = beam_maps[offset + BEAM_MAP_Y*(y+1) + (x  )];
                        neighbors[7] = beam_maps[offset + BEAM_MAP_Y*(y+1) + (x+1)];

                        //if (neighbors[3] < 0 || neighbors[4] < 0) {
                        //    // This is a border beamlet, ignore other rows
                        //    neighbors[0] = -1;
                        //    neighbors[1] = -1;
                        //    neighbors[2] = -1;
                        //    neighbors[5] = -1;
                        //    neighbors[6] = -1;
                        //    neighbors[7] = -1;

                        //}
                        for (int j = 0; j < n_neighbors; j++) {
                            if (neighbors[j] >= 0) {
                                smoothed_fluence[idx] += fluence[neighbors[j]];
                            } else {
                                center_weight += 0.8;
                            }
                        }
                        smoothed_fluence[idx] += center_weight*fluence[idx];
                        smoothed_fluence[idx] /= sum_weights;
                    }
                }
            }
        }

        for (int i = 0; i < n_beamlets; i++) {
            fluence[i] = smoothed_fluence[i];
        }

        free(neighbors);
    }
};

__global__ void voxels_min(Region *regions, char *voxel_regions,double *doses, int n_voxels, int rid, double *voxels) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (idx < n_voxels) {
        if (voxel_regions[rid*n_voxels + idx]) {
            voxels[idx] = (doses[idx] < regions[rid].pr_min);
        } else {
            voxels[idx] = 0;
        }
    }
}

__global__ void voxels_min_old(Region *regions, char *voxel_regions,double *doses, int n_voxels, int rid, double *voxels) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (idx < n_voxels) {
        if (voxel_regions[rid*n_voxels + idx] && doses[idx] < regions[rid].pr_min) {
            voxels[idx] = (regions[rid].pr_min - doses[idx]);
        } else {
            voxels[idx] = 0;
        }
    }
}

__global__ void voxels_max(Region *regions, char *voxel_regions,double *doses, int n_voxels, int rid, double *voxels) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (idx < n_voxels) {
        if (voxel_regions[rid*n_voxels + idx]) {
            voxels[idx] = -1*(doses[idx] > regions[rid].pr_max);
        } else {
            voxels[idx] = 0;
        }
    }
}

__global__ void voxels_max_old(Region *regions, char *voxel_regions,double *doses, int n_voxels, int rid, double *voxels) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (idx < n_voxels) {
        if (voxel_regions[rid*n_voxels + idx] && doses[idx] > regions[rid].pr_max) {
            voxels[idx] = (regions[rid].pr_max - doses[idx]);
        } else {
            voxels[idx] = 0;
        }
    }
}

__global__ void voxels_average(Region *regions, char *voxel_regions,double *doses, int n_voxels, int rid, double *voxels) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (idx < n_voxels) {
        if (regions[rid].avg < regions[rid].pr_avg_min) {
            voxels[idx] = voxel_regions[rid*n_voxels + idx];
        } else if (regions[rid].avg > regions[rid].pr_avg_max) {
            voxels[idx] = -voxel_regions[rid*n_voxels + idx];
        } else {
            voxels[idx] = 0;
        }
    }
}

__global__ void voxels_eud(Region *regions, char *voxel_regions, double *doses, int n_voxels, int rid, double *voxels) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    Region r = regions[rid];
    
    if (idx < n_voxels) {
        if (voxel_regions[rid*n_voxels + idx]) {
            double dEUD_dd = r.eud*pow(doses[idx], r.alpha - 1)/r.sum_alpha;
            voxels[idx] = r.dF_dEUD * dEUD_dd;
            if (r.is_ptv) {
                dEUD_dd = r.v_eud*pow(doses[idx], -r.alpha - 1)/r.v_sum_alpha;
                voxels[n_voxels + idx] = r.v_dF_dEUD * dEUD_dd;
            }
        } else {
            voxels[idx] = 0;
            if (r.is_ptv) {
                voxels[n_voxels + idx] = 0;
            }
        }
    }
}

__global__ void voxels_average_old(Region *regions, char *voxel_regions,double *doses, int n_voxels, int rid, double *voxels) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (idx < n_voxels) {
        if (voxel_regions[rid*n_voxels + idx] && regions[rid].avg < regions[rid].pr_avg_min) {
            voxels[idx] = (regions[rid].pr_avg_min - regions[rid].avg);
        } else if (voxel_regions[rid*n_voxels + idx] && regions[rid].avg > regions[rid].pr_avg_max) {
            voxels[idx] = (regions[rid].pr_avg_max - regions[rid].avg);
        } else {
            voxels[idx] = 0;
        }
    }
}

__global__ void voxels_average_objective(Region *regions, char *voxel_regions, int n_voxels, int rid, float penalty, double *voxels) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (idx < n_voxels) {
        voxels[idx] = penalty*voxel_regions[rid*n_voxels + idx];
    }
}

__global__ void reduce_gradient(double *voxels, int n_voxels, int n_gradients) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (idx < n_voxels) {
        double influence = 0;
        for (int i = 0; i < n_gradients; i++) {
            influence += voxels[i*n_voxels + idx];
        }
        voxels[idx] = influence;
    }
}

double penalty(Plan plan) {
    double penalty = 0;

    for (int i = 0; i < plan.n_regions; i++) {
        Region region = plan.regions[i];
        if (region.is_optimized) {
            if (region.pr_min > 0 &&
                region.min < region.pr_min) {
                penalty += region.pr_min - region.min;
            }
            if (region.pr_max > 0 && 
                region.max > region.pr_max) {
                penalty += region.max - region.pr_max;
            }
            if (region.pr_avg_min > 0 && 
                region.avg < region.pr_avg_min) {
                penalty += region.pr_avg_min - region.avg;
            }
            if (region.pr_avg_max > 0 && 
                region.avg > region.pr_avg_max) {
                penalty += region.avg - region.pr_avg_max;
            }
        }
    }

    return penalty;
}

double objective(Plan plan) {
    double objective = 1;

    for (int i = 0; i < plan.n_regions; i++) {
        Region region = plan.regions[i];
        if (region.is_optimized) {
            objective *= region.f;
            if (region.is_ptv) {
                objective *= region.v_f;
            }
        }
    }
    return objective;
}

//double objective(Plan plan) {
//    double objective = 0;
//
//    for (int i = 0; i < plan.n_regions; i++) {
//        Region region = plan.regions[i];
//        if (region.is_optimized) {
//            objective += region.avg*region.n_voxels;
//        }
//    }
//    return objective/plan.n_voxels;
//}

void add_grad(Plan plan, double *a, double *b, double step) {
    for (int i = 0; i < plan.n_beamlets; i++) {
        a[i] += step*b[i];
    }
    free(b);
}

void vector_stats(const char *name, double *vector, int n_values) {
    double min = 1e10, max = 0, avg = 0;
    for (int i = 0; i < n_values; i++) {
        if (vector[i] < min) {
            min = vector[i];
        }
        if (vector[i] > max) {
            max = vector[i];
        }
        avg += vector[i];
    }
    avg /= n_values;

    printf("%s: %f %f %f\n", name, min, max, avg);
}

int descend(Plan plan, double *d_momentum, float step, int rid_sll, int rid_slr) {
    int block = 512;
    int grid = (plan.n_voxels + block - 1)/block;

    double *d_voxels;
    int gradients_per_region = 3; // Warning, hardcoded!
    cudaCheck(cudaMalloc(&d_voxels, gradients_per_region*plan.n_regions*plan.n_voxels*sizeof(double)));

    // Hardcoded objective function gradients
    int n_gradients = 2;
    float penalty = -0.0000;
    voxels_average_objective<<<grid, block>>>(plan.d_regions, plan.d_voxel_regions, plan.n_voxels, rid_sll, penalty, &(d_voxels[0]));
    voxels_average_objective<<<grid, block>>>(plan.d_regions, plan.d_voxel_regions, plan.n_voxels, rid_slr, penalty, &(d_voxels[plan.n_voxels]));

    int offset = n_gradients*plan.n_voxels;
    for (int i = 0; i < plan.n_regions; i++) {
        Region region = plan.regions[i];
        if (region.is_optimized) {
            voxels_eud<<<grid, block>>>(plan.d_regions, plan.d_voxel_regions, plan.d_doses, plan.n_voxels, i, &(d_voxels[offset]));
            offset += plan.n_voxels;
            n_gradients++;
            if (region.is_ptv) {
                offset += plan.n_voxels;
                n_gradients++;
            }
        }
    }
    
    //printf("n_gradients: %d\n", n_gradients);
    reduce_gradient<<<grid, block>>>(d_voxels, plan.n_voxels, n_gradients);

    double *d_gradients;
    cudaCheck(cudaMalloc(&d_gradients, n_gradients*plan.n_beamlets*sizeof(double)));

    SparseMatrix spm = plan.spm;
    //cusparseCheck(cusparseDcsrmm(spm.handle, CUSPARSE_OPERATION_TRANSPOSE, spm.n_rows, n_gradients, spm.n_cols, spm.n_nz, &alpha, spm.descr, spm.d_vals, spm.d_rows, spm.d_cols, d_voxels, spm.n_rows, &beta, d_gradients, spm.n_cols));
    //cusparseCheck(cusparseDcsrmm(spm.handle, CUSPARSE_OPERATION_NON_TRANSPOSE, spm.n_cols, n_gradients, spm.n_rows, spm.n_nz, &alpha, spm.descr, spm.d_vals_t, spm.d_rows_t, spm.d_cols_t, d_voxels, spm.n_rows, &beta, d_gradients, spm.n_cols));
    //cusparseCheck(cusparseDcsrmv(spm.handle, CUSPARSE_OPERATION_NON_TRANSPOSE, spm.n_cols, spm.n_rows, spm.n_nz, &alpha, spm.descr, spm.d_vals_t, spm.d_rows_t, spm.d_cols_t, d_voxels, &beta, d_gradients));

    cusparseDnVecDescr_t voxels_descr;
    cusparseDnVecDescr_t gradient_descr;
    cusparseCreateDnVec(&voxels_descr, plan.n_voxels, d_voxels, CUDA_R_64F);
    cusparseCreateDnVec(&gradient_descr, plan.n_beamlets, d_gradients, CUDA_R_64F);

    double alpha = 1.0, beta = 0.0;
    void *p_buffer;
    size_t p_buffer_size = 0;
    cusparseCheck(cusparseSpMV_bufferSize(spm.handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, spm.sp_descr_t, voxels_descr, &beta,
    	gradient_descr, CUDA_R_64F, CUSPARSE_CSRMV_ALG1, &p_buffer_size));
    cudaCheck(cudaMalloc(&p_buffer, p_buffer_size*sizeof(char)));
    cusparseCheck(cusparseSpMV(spm.handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, spm.sp_descr_t, voxels_descr, &beta,
    	gradient_descr, CUDA_R_64F, CUSPARSE_CSRMV_ALG1, p_buffer));

    //double *gradients = (double *) malloc(n_gradients*plan.n_beamlets*sizeof(double));;
    //cudaCheck(cudaMemcpy(gradients, d_gradients, n_gradients*plan.n_beamlets*sizeof(double), cudaMemcpyDeviceToHost));

    //vector_stats("obj_1", &gradients[0], plan.n_beamlets);
    //vector_stats("obj_2", &gradients[plan.n_beamlets], plan.n_beamlets);
    //offset = 2*plan.n_beamlets;
    //for (int i = 0; i < plan.n_regions; i++) {
    //    Region region = plan.regions[i];
    //    if (region.is_optimized) {
    //        printf("gpu %s\n", region.name);
    //        if (region.pr_avg_min > 0 || region.pr_avg_max > 0) {
    //            vector_stats("grad_avg", &gradients[offset], plan.n_beamlets);
    //            offset += plan.n_beamlets;
    //        }
    //        if (region.pr_min > 0) {
    //            vector_stats("grad_min", &gradients[offset], plan.n_beamlets);
    //            offset += plan.n_beamlets;
    //        }
    //        if (region.pr_max > 0) {
    //            vector_stats("grad_max", &gradients[offset], plan.n_beamlets);
    //            offset += plan.n_beamlets;
    //        }
    //    }
    //}

    grid = (plan.n_beamlets + block - 1)/block;
    apply_gradients<<<grid, block>>>(d_gradients, d_momentum, plan.n_beamlets, n_gradients, step, plan.d_fluence);

    cudaCheck(cudaFree(d_gradients));
    cudaCheck(cudaFree(d_voxels));
    return n_gradients;
}


void optimize_gpu(Plan plan, int rid_sll, int rid_slr, float gurobi_avg_sll, float gurobi_avg_slr, float stop_ratio) {

    cudaCheck(cudaMemcpy(plan.d_regions, plan.regions, plan.n_regions*sizeof(Region), cudaMemcpyHostToDevice));
    //cudaCheck(cudaMemset(plan.d_fluence, 0, plan.n_beamlets*sizeof(double)));

    double *d_momentum;
    int gradients_per_region = 3; // Warning, hardcoded!
    cudaCheck(cudaMalloc(&d_momentum, gradients_per_region*plan.n_regions*plan.n_beamlets*sizeof(double)));
    cudaCheck(cudaMemset(d_momentum, 0, gradients_per_region*plan.n_regions*plan.n_beamlets*sizeof(double)));

    plan.compute_dose();
    plan.stats();
    cudaCheck(cudaMemcpy(plan.regions, plan.d_regions, plan.n_regions*sizeof(Region), cudaMemcpyDeviceToHost));
    printf("Initial solution:\n");
    plan.print_table();
    //exit(0);

    // TODO: Inicializar la fluencia a 0 rompe el algoritmo por los EUD. Hay que revisarlo o inicializar a 0.1 o algo.
    //cudaCheck(cudaMemset(plan.d_fluence, 0, plan.n_beamlets*sizeof(double)));
    //plan.compute_dose();
    //plan.stats();
    //cudaCheck(cudaMemcpy(plan.regions, plan.d_regions, plan.n_regions*sizeof(Region), cudaMemcpyDeviceToHost));
    //plan.print_table();

    //float step = 2e-9;
    //float decay = 1e-7;
    //float min_step = 1e-9;
    float step = 2e-7;
    float decay = 1e-7;
    float min_step = 1e-1;
    double start_time = get_time_s();
    double current_time;
    double last_pen = 0;
    double last_obj = 0;
    double last_obj2 = 0;

    int it = 0;
    while (running && get_time_s() - start_time < 60) {
        descend(plan, d_momentum, step, rid_sll, rid_slr);

        cudaCheck(cudaMemcpy(plan.fluence, plan.d_fluence, plan.n_beamlets*sizeof(double), cudaMemcpyDeviceToHost));
        plan.smooth_cpu();
        cudaCheck(cudaMemcpy(plan.d_fluence, plan.fluence, plan.n_beamlets*sizeof(double), cudaMemcpyHostToDevice));

        plan.compute_dose();
        plan.stats();

        if (it % 100 == 0) {
            cudaCheck(cudaMemcpy(plan.regions, plan.d_regions, plan.n_regions*sizeof(Region), cudaMemcpyDeviceToHost));

            current_time = get_time_s();
            double pen = penalty(plan);
            double obj = plan.regions[rid_sll].avg + plan.regions[rid_slr].avg;
            double obj2 = objective(plan);
            printf("\n[%.3f] Iteration %d %e\n", current_time - start_time, it, step);
            //printf("penalty: %9.6f (%9.6f percent)\n", pen, ((pen-last_pen)*100/last_pen));
            //printf("    obj: %9.6f (%9.6f percent)\n", obj, ((obj-last_obj)*100/last_obj));
            printf("penalty: %9.6f (%9.6f)\n", pen, pen-last_pen);
            printf("    obj: %9.6f (%9.6f)\n", obj, obj-last_obj); 
            printf("   obj2: %9.24f %9.24f %9.24f\n", obj2, obj2-last_obj2, (obj2-last_obj2)/obj2);
            plan.print_table();

            //if (abs(obj2-last_obj2)/obj2 < stop_ratio) {
            //    break;
            //}

            last_pen = pen;
            last_obj = obj;
            last_obj2 = obj2;

            if (it % 10000 == 0) {
                const char* out = "x_temp.txt";
                cudaCheck(cudaMemcpy(plan.fluence, plan.d_fluence, plan.n_beamlets*sizeof(double), cudaMemcpyDeviceToHost));
                FILE *f = fopen(out, "w");
                for (int i = 0; i < plan.n_beamlets; i++) {
                    fprintf(f, "%.10e\n", plan.fluence[i]);
                }
                fclose(f);
                printf("Last fluence written to %s\n", out);
            }
        }
        //if (it % 100000 == 0) {
        //    step /= 10;
        //}
        if (step > min_step) 
            step = step/(1 + decay*it);
        it++;
        //if (it == 10000) 
        //    break;
    }
    cudaCheck(cudaMemcpy(plan.regions, plan.d_regions, plan.n_regions*sizeof(Region), cudaMemcpyDeviceToHost));
    double elapsed = get_time_s() - start_time;
    printf("\nRan %d iterations in %.4f seconds (%.4f sec/it) \n", it, elapsed, elapsed/it);
    printf("penalty: %f\n", penalty(plan));
    printf("    obj: %f\n", plan.regions[rid_sll].avg + plan.regions[rid_slr].avg);
    plan.print_table();

    cudaCheck(cudaFree(d_momentum));
}

int main(int argc, char **argv) {

    signal(SIGINT, interrupt_handler);

    int plan_n = atoi(argv[1]);
    const char* plan_path = argv[2];
    const char* out_path = argv[3];
    const char* fluence_path;
    const char* fluence_prefix;
    float stop_ratio = 1e-5;

    if (argc > 4) {
        fluence_path = argv[4];
        fluence_prefix = argv[5];
    } else {
        // We use the starting plan from Eclipse
        fluence_path = plan_path;
        fluence_prefix = "x_PARETO";
    }

    Plan plan = {};
    plan.load(plan_path, fluence_path, fluence_prefix);

    int rid_sll, rid_slr;
    float gurobi_avg_sll, gurobi_avg_slr;

    if (plan_n == 3) {
        rid_sll = 5;
        rid_slr = 6;
        plan.regions[ 0].set_targets(false,    -1,    -1,    -1,    -1,    -1,  10,   5);
        plan.regions[ 1].set_targets(false,    -1,    -1,    -1,    -1,    -1,  10,   5);
        plan.regions[ 2].set_targets(false,    -1,    -1,    -1,    60,    60,  10,   5);
        plan.regions[ 3].set_targets(false,    -1,    -1,    -1,    -1,    -1,  10,   5);
        plan.regions[ 4].set_targets(false,    -1,    -1,    -1,    50,    50,  10,   5);
        plan.regions[ 5].set_targets(false,    -1,    -1,    26,    -1,    1,   1,   1);
        plan.regions[ 6].set_targets(false,    -1,    -1,    26,    -1,    1,   1,   1);
        plan.regions[ 7].set_targets(false,    -1,    -1,    -1,    70,    70,  10,   5);
        plan.regions[ 8].set_targets(false,    -1,    -1,    -1, 74.25, 74.25,  40,   5);
        plan.regions[ 9].set_targets( true, 60.75, 66.15, 68.85, 74.25, 67.50, -40,  50);
        plan.regions[10].set_targets( true, 54.00, 58.80, 61.20, 66.00, 60.00, -50, 100);
        plan.regions[11].set_targets( true, 48.60, 52.92, 55.08, 59.40, 54.00, -40, 100);
        gurobi_avg_sll = -1;
        gurobi_avg_slr = -1;
    } else if (plan_n == 4) {
        rid_sll = 2;
        rid_slr = 1;
        plan.regions[ 0].set_targets(false,    -1,    -1,    -1,    70,    70,  10,   5);
        plan.regions[ 1].set_targets(false,    -1,    -1,    26,    -1,    1,   1,   5);
        plan.regions[ 2].set_targets(false,    -1,    -1,    26,    -1,    1,   1,   5);
        plan.regions[ 3].set_targets(false,    -1,    -1,    -1,    50,    50,  10,   5);
        plan.regions[ 4].set_targets(false,    -1,    -1,    -1,    -1,    -1,  10,   5);
        plan.regions[ 5].set_targets( true, 59.40, 64.67, 67.32, 72.60, 66.00, -40,  100);
        plan.regions[ 6].set_targets( true, 53.46, 58.21, 60.59, 65.34, 59.40, -40,  100);
        plan.regions[ 7].set_targets(false,    -1,    -1,    -1,    60,    60,  10,   5);
        plan.regions[ 8].set_targets(false,    -1,    -1,    -1,    -1,    -1,  10,   5);
        plan.regions[ 9].set_targets(false,    -1,    -1,    -1, 74.25, 74.25,  40,   5);
        plan.regions[10].set_targets(false,    -1,    -1,    -1,    -1,    -1,  10,   5);
        gurobi_avg_sll = -1;
        gurobi_avg_slr = -1;

    } else if (plan_n == 5) {
        rid_sll = 3;
        rid_slr = 4;

        plan.regions[ 0].set_targets(false,    -1,    -1,    -1,    -1,    -1,  10,   5);
        plan.regions[ 1].set_targets(false,    -1,    -1,    -1, 74.25, 74.25,  40,  5);
        plan.regions[ 2].set_targets(false,    -1,    -1,    -1,    70,    70,  10,   5);
        plan.regions[ 3].set_targets(false,    -1,    -1,    26,    -1,    26,   1,   5);
        plan.regions[ 4].set_targets(false,    -1,    -1,    26,    -1,    26,   1,   5);
        plan.regions[ 5].set_targets(false,    -1,    -1,    -1,    50,    50,  10,   5);
        plan.regions[ 6].set_targets(false,    -1,    -1,    -1,    -1,    -1,  10,   5);
        plan.regions[ 7].set_targets(false,    -1,    -1,    -1,    60,    60,  10,   5);
        plan.regions[ 8].set_targets( true, 48.60, 52.92, 55.08, 59.40, 54.00, -100,  150);
        plan.regions[ 9].set_targets( true, 54.00, 58.80, 61.20, 66.00, 60.00, -100,  150);
        plan.regions[10].set_targets( true, 59.40, 64.67, 67.32, 72.60, 66.00, -60,  50);
        plan.regions[11].set_targets(false,    -1,    -1,    -1,    -1,    -1,  10,   5);
        gurobi_avg_sll = -1;
        gurobi_avg_slr = -1;
    }
        

    optimize_gpu(plan, rid_sll, rid_slr, gurobi_avg_sll, gurobi_avg_slr, stop_ratio);

    cudaCheck(cudaMemcpy(plan.fluence, plan.d_fluence, plan.n_beamlets*sizeof(double), cudaMemcpyDeviceToHost));
    FILE *f = fopen(out_path, "w");
    for (int i = 0; i < plan.n_beamlets; i++) {
        fprintf(f, "%.10e\n", plan.fluence[i]);
    }
    fclose(f);
    printf("Last fluence written to %s\n", out_path);
}
