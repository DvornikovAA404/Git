#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "cblas.h"

using namespace std;

#define RESET "\033[0m"
#define GREEN "\033[32m"
#define RED "\033[31m"
#define YELLOW "\033[33m"
#define CYAN "\033[36m"

struct TestSuite {
    int passed = 0;
    int total = 0;

    void report(bool (*test_func)(), const string& name) {
        total++;
        try {
            if (test_func()) {
                passed++;
                cout << GREEN << "  [OK]   " << RESET << name << '\n';
            } else {
                cout << RED << "  [FAIL] " << RESET << name
                     << " (Неверный результат вычислений)\n";
            }
        } catch (const exception& e) {
            cout << RED << "  [FAIL] " << RESET << name
                 << " (Критическая ошибка: " << e.what() << ")\n";
        } catch (...) {
            cout << RED << "  [FAIL] " << RESET << name
                 << " (Неизвестный сбой в коде)\n";
        }
    }

    void summary() const {
        cout << '\n' << CYAN << "=== ИТОГИ ТЕСТИРОВАНИЯ ===" << RESET << '\n';
        cout << "Всего тестов: " << total << '\n';
        cout << "Успешно:      " << GREEN << passed << RESET << '\n';
        if (passed < total) {
            cout << "Ошибок:       " << RED << total - passed << RESET << '\n';
        }
    }
};

struct BenchmarkResult {
    string name;
    double manual_ms;
    double openblas_ms;
    double performance_pct; 
};

template <typename T>
bool almost_equal(T lhs, T rhs, T eps) {
    return fabs(lhs - rhs) <= eps;
}

bool test_axpy() {
    float x[2] = {1.0f, 2.0f};
    float y[2] = {3.0f, 4.0f};
    cblas_saxpy(2, 2.0f, x, 1, y, 1);
    return y[0] == 5.0f && y[1] == 8.0f;
}

bool test_scal() {
    double x[2] = {10.0, 20.0};
    cblas_dscal(2, 0.5, x, 1);
    return x[0] == 5.0 && x[1] == 10.0;
}

bool test_copy_swap() {
    float x[2] = {1.0f, 2.0f};
    float y[2] = {0.0f, 0.0f};
    cblas_scopy(2, x, 1, y, 1);
    float a = 5.0f;
    float b = 10.0f;
    cblas_sswap(1, &a, 1, &b, 1);
    return y[0] == 1.0f && a == 10.0f && b == 5.0f;
}

bool test_dot() {
    double x[2] = {1.0, 2.0};
    double y[2] = {3.0, 4.0};
    double res = cblas_ddot(2, x, 1, y, 1);
    return res == 11.0;
}

bool test_norms() {
    float x[3] = {3.0f, -4.0f, 0.0f};
    float n2 = cblas_snrm2(3, x, 1);
    float a1 = cblas_sasum(3, x, 1);
    return almost_equal(n2, 5.0f, 1e-5f) &&
           almost_equal(a1, 7.0f, 1e-5f);
}

bool test_iamax() {
    double x[4] = {1.0, -10.0, 5.0, 2.0};
    size_t idx = cblas_idamax(4, x, 1);
    return idx == 1;
}

bool test_rotations() {
    float a = 1.0f;
    float b = 1.0f;
    float c = 0.0f;
    float s = 0.0f;
    cblas_srotg(&a, &b, &c, &s);

    float x[2] = {1.0f, 0.0f};
    float y[2] = {0.0f, 1.0f};
    cblas_srot(2, x, 1, y, 1, c, s);

    return isfinite(c) && isfinite(s) &&
           isfinite(x[0]) && isfinite(y[0]);
}

bool test_rotm() {
    double d1 = 1.0;
    double d2 = 1.0;
    double b1 = 1.0;
    double b2 = 1.0;
    double param[5] = {};
    cblas_drotmg(&d1, &d2, &b1, b2, param);

    double vx[2] = {1.0, 2.0};
    double vy[2] = {3.0, 4.0};
    cblas_drotm(2, vx, 1, vy, 1, param);

    return isfinite(vx[0]) && isfinite(vx[1]) &&
           isfinite(vy[0]) && isfinite(vy[1]);
}

void manual_saxpy(int n, float alpha, const float* x, float* y) {
    for (int i = 0; i < n; ++i) {
        y[i] += alpha * x[i];
    }
}

void manual_dscal(int n, double alpha, double* x) {
    for (int i = 0; i < n; ++i) {
        x[i] *= alpha;
    }
}

double manual_ddot(int n, const double* x, const double* y) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += x[i] * y[i];
    }
    return sum;
}

template <typename F>
double measure_ms(F&& func, int repeats) {
    using clock = chrono::steady_clock;
    const auto start = clock::now();
    for (int i = 0; i < repeats; ++i) {
        func();
    }
    const auto finish = clock::now();
    chrono::duration<double, milli> elapsed = finish - start;
    return elapsed.count();
}

inline double calc_perf(double manual, double openblas) {
    return (openblas / manual) * 100.0;
}

BenchmarkResult benchmark_saxpy(int n, int repeats) {
    vector<float> x(n);
    vector<float> y_base(n);

    for (int i = 0; i < n; ++i) {
        x[i] = 0.001f * (i % 100 + 1);
        y_base[i] = 0.002f * (i % 70 + 1);
    }

    const float alpha = 1.75f;
    volatile float guard = 0.0f;

    const double manual_ms = measure_ms([&]() {
        vector<float> y = y_base;
        manual_saxpy(n, alpha, x.data(), y.data());
        guard += y[n / 2];
    }, repeats);

    const double openblas_ms = measure_ms([&]() {
        vector<float> y = y_base;
        cblas_saxpy(n, alpha, x.data(), 1, y.data(), 1);
        guard += y[n / 2];
    }, repeats);

    return {"saxpy(float)", manual_ms, openblas_ms,
            calc_perf(manual_ms, openblas_ms)};
}

BenchmarkResult benchmark_dscal(int n, int repeats) {
    vector<double> x_base(n);

    for (int i = 0; i < n; ++i) {
        x_base[i] = 0.003 * (i % 130 + 1);
    }

    const double alpha = 0.75;
    volatile double guard = 0.0;

    const double manual_ms = measure_ms([&]() {
        vector<double> x = x_base;
        manual_dscal(n, alpha, x.data());
        guard += x[n / 3];
    }, repeats);

    const double openblas_ms = measure_ms([&]() {
        vector<double> x = x_base;
        cblas_dscal(n, alpha, x.data(), 1);
        guard += x[n / 3];
    }, repeats);

    return {"dscal(double)", manual_ms, openblas_ms,
            calc_perf(manual_ms, openblas_ms)};
}

BenchmarkResult benchmark_ddot(int n, int repeats) {
    vector<double> x(n), y(n);

    for (int i = 0; i < n; ++i) {
        x[i] = 0.001 * (i % 150 + 1);
        y[i] = 0.002 * (i % 90 + 1);
    }

    volatile double guard = 0.0;

    const double manual_ms = measure_ms([&]() {
        guard += manual_ddot(n, x.data(), y.data());
    }, repeats);

    const double openblas_ms = measure_ms([&]() {
        guard += cblas_ddot(n, x.data(), 1, y.data(), 1);
    }, repeats);

    return {"ddot(double)", manual_ms, openblas_ms,
            calc_perf(manual_ms, openblas_ms)};
}

void run_benchmarks() {
    const int n = 2'000'000;
    const int repeats = 12;

    cout << '\n' << CYAN << "=== СРАВНЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ ===" << RESET << '\n';
    cout << "OpenBLAS принят за 100%\n";

    vector<BenchmarkResult> results;
    results.push_back(benchmark_saxpy(n, repeats));
    results.push_back(benchmark_dscal(n, repeats));
    results.push_back(benchmark_ddot(n, repeats));

    cout << '\n'
         << left << setw(28) << "Операция"
         << right << setw(16) << "Manual, ms"
         << "  " << setw(16) << "OpenBLAS, ms"
         << "  " << setw(12) << "% от OpenBLAS" << '\n';

    cout << string(70, '-') << '\n';

    cout << fixed << setprecision(2);

    for (const auto& r : results) {
        cout << left << setw(18) << r.name
             << right << setw(16) << r.manual_ms
             << "  " << setw(16) << r.openblas_ms
             << "  " << setw(10) << r.performance_pct << "%\n";
    }
}

int main(int argc, char** argv) {
    run_benchmarks();
    return 0;
}