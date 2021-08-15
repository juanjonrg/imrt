#include <stdio.h> 
#include <stdlib.h> 
#include <dirent.h> 
#include <signal.h>
#include <string.h>
#include <time.h>
#include "mkl.h"

static volatile int running = 1;

void interrupt_handler(int signal) {
    running = 0;
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
};

struct Region {
    char *name;
    int n_voxels;
    double min;
    double max;
    double avg;
    double eud;
    double dF_dEUD;
    double sum_alpha;
    // Virtual EUD to control PTV overdosage
    // Hardcoded to eud + 1 for now
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
            eud = 0;
            v_eud = 0;
            dF_dEUD = 0;
            v_dF_dEUD = 0;
            sum_alpha = 0;
            v_sum_alpha = 0;
        }
    }
};

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
    double *doses;
    double *d_fluence;
    double *d_doses;
    char *files[100];
    sparse_matrix_t m;
    sparse_matrix_t m_t;
    struct matrix_descr descr;

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

        descr.type = SPARSE_MATRIX_TYPE_GENERAL;
        mkl_sparse_d_create_coo(&m, SPARSE_INDEX_BASE_ZERO, spm.n_rows, spm.n_cols, spm.n_nz, spm.rows, spm.cols, spm.vals);
        mkl_sparse_convert_csr(m, SPARSE_OPERATION_TRANSPOSE, &m_t);
        mkl_sparse_convert_csr(m, SPARSE_OPERATION_NON_TRANSPOSE, &m);
        mkl_sparse_set_mv_hint(m, SPARSE_OPERATION_NON_TRANSPOSE, descr, 1e6);
        mkl_sparse_set_mv_hint(m_t, SPARSE_OPERATION_NON_TRANSPOSE, descr, 1e6);
        mkl_sparse_optimize(m);
        mkl_sparse_optimize(m_t);
    
    }

    void load_fluence(const char *path) {
        int n_files = read_files(path, "x_PARETO", files);

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

        free(fp);
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
            printf("  Region %2d: %-16s %8d voxels\n", i, regions[i].name, regions[i].n_voxels);
        }
        printf("Dose matrix: %d x %d with %d nonzeros.\n", spm.n_rows, spm.n_cols, spm.n_nz);
    }

    void compute_dose() {
        memset(doses, 0, n_voxels*sizeof(*doses));

        double alpha = 1.0, beta = 0.0;

        double start_time = get_time_s();
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, alpha, m, descr, fluence, beta, doses);
        double elapsed = get_time_s() - start_time;
        //printf("Compute dose mv: %.4f seconds.\n", elapsed);

        for (int i = 0; i < n_voxels; i++) {
            doses[i] *= dose_grid_scaling;
        }
    }

    void stats() {
        for (int i = 0; i < n_regions; i++) {
            regions[i].min = 1e10;
            regions[i].max = 0;
            regions[i].avg = 0;
            for (int j = 0; j < n_voxels; j++) {
                if (voxel_regions[i*n_voxels + j]) {
                    double dose = doses[j];
                    if (regions[i].min > dose) {
                        regions[i].min = dose;
                    }
                    if (regions[i].max < dose) {
                        regions[i].max = dose;
                    }
                    regions[i].avg += dose;
                }
            }
            regions[i].avg /= regions[i].n_voxels;
        }
    }

    void print_table() {
        printf("    Region          Min         Avg         Max\n"); 
        for (int i = 0; i < n_regions; i++) {
            if (regions[i].is_optimized) {
                printf("%-15s %11.6lf %11.6lf %11.6lf\n", regions[i].name, regions[i].min, regions[i].avg, regions[i].max);
            }
        }
    }

    void load(const char *path) {
        parse_config(path);
        parse_voxel_regions(path);
        load_spm(path);

        fluence = (double *) malloc(n_beamlets*sizeof(double));
        doses = (double *) malloc(n_voxels*sizeof(double));

        load_fluence(path);
        print();
    }
};

void voxels_max(Plan plan, int rid, double *voxels) {
    for (int i = 0; i < plan.n_voxels; i++) {
        if (plan.voxel_regions[rid*plan.n_voxels + i] &&
            plan.doses[i] > plan.regions[rid].pr_max) {
            voxels[i] -= 1;
        }
    }
}

void voxels_min(Plan plan, int rid, double *voxels) {
    for (int i = 0; i < plan.n_voxels; i++) {
        if (plan.voxel_regions[rid*plan.n_voxels + i] &&
            plan.doses[i] < plan.regions[rid].pr_min) {
            voxels[i] += 1;
        }
    }
}

void voxels_avg(Plan plan, int rid, double *voxels) {
    int sign = 0;
    if (plan.regions[rid].avg < plan.regions[rid].pr_avg_min) {
        sign = 1;
    } else if (plan.regions[rid].avg > plan.regions[rid].pr_avg_max) {
        sign = -1;
    }

    if (sign != 0) {
        for (int i = 0; i < plan.n_voxels; i++) {
            if (plan.voxel_regions[rid*plan.n_voxels + i]) {
                voxels[i] += sign;
            }
        }
    }
}

void voxels_avg_obj(Plan plan, int rid, double *voxels, double penalty) {
    for (int i = 0; i < plan.n_voxels; i++) {
        if (plan.voxel_regions[rid*plan.n_voxels + i]) {
            voxels[i] += penalty;
        }
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

void descend(Plan plan, double *voxels, double *gradient, float step, int rid_sll, int rid_slr) {
    memset(voxels, 0, plan.n_voxels*sizeof(*voxels));
    memset(gradient, 0, plan.n_beamlets*sizeof(*gradient));

    // Hardcoded objective function gradients
    double penalty = -0.07;
    voxels_avg_obj(plan, rid_sll, voxels, penalty);
    voxels_avg_obj(plan, rid_slr, voxels, penalty);

    for (int i = 0; i < plan.n_regions; i++) {
        Region region = plan.regions[i];
        if (region.is_optimized) {
            if (region.pr_avg_min > 0 || region.pr_avg_max > 0) {
                voxels_avg(plan, i, voxels);
            }
            if (region.pr_min > 0) {
                voxels_min(plan, i, voxels);
            }
            if (region.pr_max > 0) {
                voxels_max(plan, i, voxels);
            }
        }
    }
    
    double alpha = 1.0, beta = 0.0;

    double start_time = get_time_s();
    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, alpha, plan.m_t, plan.descr, voxels, beta, gradient);
    double elapsed = get_time_s() - start_time;
    //printf("Descend mv: %.4f seconds.\n", elapsed);

    for (int i = 0; i < plan.n_beamlets; i++) {
        plan.fluence[i] += step*gradient[i];
        if (plan.fluence[i] > 0.2) {
            plan.fluence[i] = 0.2;
        } else if (plan.fluence[i] < 0) {
            plan.fluence[i] = 0;
        }
    }
}

void optimize(Plan plan, int rid_sll, int rid_slr, float gurobi_avg_sll, float gurobi_avg_slr) {

    //double step = 5e-9;
    //double decay = 1e-7;
    //double min_step = 1e-9;
    double step = 1e-9;
    double decay = 1e-7;
    double min_step = 1e-1;
    double start_time = get_time_s();
    double current_time;

    double *voxels = (double *) malloc(plan.n_voxels*sizeof(*voxels));
    double *gradient = (double *) malloc(plan.n_beamlets*sizeof(*gradient));

    memset(plan.fluence, 0, plan.n_beamlets*sizeof(*plan.fluence));

    plan.compute_dose();
    plan.stats();

    int it = 0;
    while (running && get_time_s() - start_time < 1e8) {
        descend(plan, voxels, gradient, step, rid_sll, rid_slr);
        plan.compute_dose();
        plan.stats();

        if (it % 100 == 0) {
            current_time = get_time_s();
            double pen = penalty(plan);
            printf("\n[%.2f] Iteration %d %e\n", current_time - start_time, it, step);
            printf("penalty: %f\n", pen);
            printf("    obj: %f\n", plan.regions[rid_sll].avg + plan.regions[rid_slr].avg);
            plan.print_table();

        }
        if (step > min_step) 
            step = step/(1 + decay*it);
        it++;
        if (it == 1000) 
            break;
    }
    double elapsed = get_time_s() - start_time;
    printf("\nRan %d iterations in %.4f seconds (%.4f sec/it) \n", it, elapsed, elapsed/it);
    printf("penalty: %f\n", penalty(plan));
    printf("    obj: %f\n", plan.regions[rid_sll].avg + plan.regions[rid_slr].avg);
    plan.print_table();
}

int main(int argc, char **argv) {

    signal(SIGINT, interrupt_handler);

    int plan_n = atoi(argv[1]);
    const char* path;
    const char* out;
    if (plan_n == 3) {
        path = "../plans/3";
        out = "x_gradient_3.txt";
    } else if (plan_n == 4) {
        path = "../plans/4";
        out = "x_gradient_4.txt";
    } else if (plan_n == 5) {
        path = "../plans/5";
        out = "x_gradient_5.txt";
    } else if (plan_n == 2) {
        path = "../plans/old/3";
        out = "x_gradient_old_3.txt";
    }

    Plan plan = {};
    plan.load(path);

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
        plan.regions[ 3].set_targets(false,    -1,    -1,    26,    -1,    1,   1,   5);
        plan.regions[ 4].set_targets(false,    -1,    -1,    26,    -1,    1,   1,   5);
        plan.regions[ 5].set_targets(false,    -1,    -1,    -1,    50,    50,  10,   5);
        plan.regions[ 6].set_targets(false,    -1,    -1,    -1,    -1,    -1,  10,   5);
        plan.regions[ 7].set_targets(false,    -1,    -1,    -1,    60,    60,  10,   5);
        plan.regions[ 8].set_targets( true, 48.60, 52.92, 55.08, 59.40, 54.00, -40,  50);
        plan.regions[ 9].set_targets( true, 54.00, 58.80, 61.20, 66.00, 60.00, -40,  50);
        plan.regions[10].set_targets( true, 59.40, 64.67, 67.32, 72.60, 66.00, -100,  100);
        plan.regions[11].set_targets(false,    -1,    -1,    -1,    -1,    -1,  10,   5);
        gurobi_avg_sll = -1;
        gurobi_avg_slr = -1;
    }
    optimize(plan, rid_sll, rid_slr, gurobi_avg_sll, gurobi_avg_slr);

    FILE *f = fopen(out, "w");
    for (int i = 0; i < plan.n_beamlets; i++) {
        fprintf(f, "%.10e\n", plan.fluence[i]);
    }
    fclose(f);
    printf("Last fluence written to %s\n", out);
}
