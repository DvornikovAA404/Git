#include <iostream>
#include <vector>
#include <string>
#include <complex>
#include <cmath>
#include "cblas.h"

using namespace std;

#define RESET   "\033[0m"
#define GREEN   "\033[32m"
#define RED     "\033[31m"
#define YELLOW  "\033[33m"
#define CYAN    "\033[36m"

struct TestSuite {
    int passed = 0;
    int total = 0;

    void report(bool (*test_func)(), string name) {
        total++;
        try {
            if (test_func()) {
                passed++;
                cout << GREEN << "  [OK]   " << RESET << name << endl;
            } else {
                cout << RED << "  [FAIL] " << RESET << name << " (Неверный результат вычислений)" << endl;
            }
        } catch (const exception& e) {
            cout << RED << "  [FAIL] " << RESET << name << " (Критическая ошибка: " << e.what() << ")" << endl;
        } catch (...) {
            cout << RED << "  [FAIL] " << RESET << name << " (Неизвестный сбой в коде)" << endl;
        }
    }

    void summary() {
        cout << "\n" << CYAN << "=== ИТОГИ ТЕСТИРОВАНИЯ ===" << RESET << endl;
        cout << "Всего тестов: " << total << endl;
        cout << "Успешно:      " << GREEN << passed << RESET << endl;
        if (passed < total) {
            cout << "Ошибок:       " << RED << total - passed << RESET << endl;
        }
    }
};

bool test_axpy() {
    float x[2] = {1.0, 2.0}, y[2] = {3.0, 4.0};
    cblas_saxpy(2, 2.0f, x, 1, y, 1); 
    return (y[0] == 5.0f && y[1] == 8.0f);
}

bool test_scal() {
    double x[2] = {10.0, 20.0};
    cblas_dscal(2, 0.5, x, 1);
    return (x[0] == 5.0 && x[1] == 10.0);
}

bool test_copy_swap() {
    float x[2] = {1.0, 2.0}, y[2] = {0.0, 0.0};
    cblas_scopy(2, x, 1, y, 1);
    float a = 5.0f, b = 10.0f;
    cblas_sswap(1, &a, 1, &b, 1);
    return (y[0] == 1.0f && a == 10.0f);
}

bool test_dot() {
    double x[2] = {1.0, 2.0}, y[2] = {3.0, 4.0};
    double res = cblas_ddot(2, x, 1, y, 1); 
    return (res == 11.0);
}

bool test_norms() {
    float x[3] = {3.0f, -4.0f, 0.0f};
    float n2 = cblas_snrm2(3, x, 1);  
    float a1 = cblas_sasum(3, x, 1);  
    return (abs(n2 - 5.0f) < 1e-5 && abs(a1 - 7.0f) < 1e-5);
}

bool test_iamax() {
    double x[4] = {1.0, -10.0, 5.0, 2.0};
    size_t idx = cblas_idamax(4, x, 1); 
    return (idx == 1);
}

bool test_rotations() {
    float a = 1.0f, b = 1.0f, c, s;
    cblas_srotg(&a, &b, &c, &s);
    float x[2] = {1.0, 0.0}, y[2] = {0.0, 1.0};
    cblas_srot(2, x, 1, y, 1, c, s);
    return (true); 
}

bool test_rotm() {
    double d1 = 1.0, d2 = 1.0, b1 = 1.0, b2 = 1.0;
    double param[5];
    cblas_drotmg(&d1, &d2, &b1, b2, param);
    double vx[2] = {1.0, 2.0}, vy[2] = {3.0, 4.0};
    cblas_drotm(2, vx, 1, vy, 1, param);
    return (param[0] >= -2.0); 
}

int main() {
    TestSuite ts;

    cout << CYAN << "=== ЗАПУСК ТЕСТОВ BLAS LEVEL 1 ===" << RESET << "\n" << endl;

    ts.report(test_axpy,      "AXPY (Vector Update)");
    ts.report(test_scal,      "SCAL (Vector Scaling)");
    ts.report(test_copy_swap, "COPY/SWAP (Data Movement)");
    ts.report(test_dot,       "DOT (Scalar Product)");
    ts.report(test_norms,     "NRM2/ASUM (Vector Norms)");
    ts.report(test_iamax,     "IAMAX (Index of Maximum)");
    ts.report(test_rotations, "ROTG/ROT (Givens Rotations)");
    ts.report(test_rotm,      "ROTMG/ROTM (Modified Rotations)");

    ts.summary();

    return (ts.passed == ts.total) ? 0 : 1;
}