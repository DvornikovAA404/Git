#include <iostream>
#include "cblas.h"

using namespace std;

// s, d, c, z axpy ( n, alpha, x, incx, y, incy ) update vector y = y + αx 2n 2n
// s, d, c, z, cs, zd scal ( n, alpha, x, incx ) scale vector y = αy n n
// s, d, c, z copy ( n, x, incx, y, incy ) copy vector y = x 0 2n
// s, d, c, z swap ( n, x, incx, y, incy ) swap vectors x ↔ y 0 2n
// s, d dot ( n, x, incx, y, incy ) dot product = xT y 2n 2n
// c, z dotu ( n, x, incx, y, incy ) (complex) = xT y 2n 2n
// c, z dotc ( n, x, incx, y, incy ) (complex conj) = xH y 2n 2n
// sds, ds dot ( n, x, incx, y, incy ) (internally double precision) = xT y 2n 2n
// s, d, sc, dz nrm2 ( n, x, incx ) 2-norm = ∥x∥2 2n n
// s, d, sc, dz asum ( n, x, incx ) 1-norm = ∥Re(x)∥1 + ∥Im(x)∥1 n n
// s, d, c, z i amax ( n, x, incx ) ∞-norm = argmaxi( |Re(xi)| + |Im(xi)| ) n n
// s, d, c, z rotg ( a, b, c, s ) generate plane (Given’s) rotation (c real, s complex) O(1) O(1)
// s, d, c, z † rot ( n, x, incx, y, incy, c, s ) apply plane rotation (c real, s complex) 6n 2n
// cs, zd rot ( n, x, incx, y, incy, c, s ) apply plane rotation (c & s real) 6n 2n
// s, d rotmg ( d1, d2, a, b, param ) generate modified plane rotation O(1) O(1)
// s, d rotm ( n, x, incx, y, incy, param ) apply modified plane rotation 6n 2n
int test_axpy() {
    cout << "=== axpy ===" << endl;
    float x_s[] = {1.0, 2.0, 3.0};
    float y_s[] = {4.0, 5.0, 6.0};
    cblas_saxpy(3, 1.0f, x_s, 1, y_s, 1);
    cout << "cblas_saxpy: \033[32mPASSED\033[0m" << endl;

    double x_d[] = {1.0, 2.0, 3.0};
    double y_d[] = {4.0, 5.0, 6.0};
    cblas_daxpy(3, 1.0, x_d, 1, y_d, 1);
    cout << "cblas_daxpy: \033[32mPASSED\033[0m" << endl;

    float x_c[] = {1.0, 0.0, 2.0, 0.0, 3.0, 0.0}; 
    float y_c[] = {4.0, 0.0, 5.0, 0.0, 6.0, 0.0};
    float alpha_c[] = {1.0, 0.0};
    cblas_caxpy(3, alpha_c, x_c, 1, y_c, 1);
    cout << "cblas_caxpy: \033[32mPASSED\033[0m" << endl;

    double x_z[] = {1.0, 0.0, 2.0, 0.0, 3.0, 0.0};
    double y_z[] = {4.0, 0.0, 5.0, 0.0, 6.0, 0.0};
    double alpha_z[] = {1.0, 0.0};
    cblas_zaxpy(3, alpha_z, x_z, 1, y_z, 1);
    cout << "cblas_zaxpy: \033[32mPASSED\033[0m" << endl;
    return 0;
}

int test_scal() {
    cout << "=== scal ===" << endl;
    float x_s[] = {1.0, 2.0, 3.0};
    cblas_sscal(3, 2.0f, x_s, 1);
    cout << "cblas_sscal: \033[32mPASSED\033[0m" << endl;

    double x_d[] = {1.0, 2.0, 3.0};
    cblas_dscal(3, 2.0, x_d, 1);
    cout << "cblas_dscal: \033[32mPASSED\033[0m" << endl;

    float x_c[] = {1.0, 0.0, 2.0, 0.0, 3.0, 0.0}; 
    float alpha_c[] = {2.0, 0.0};
    cblas_cscal(3, alpha_c, x_c, 1);
    cout << "cblas_cscal: \033[32mPASSED\033[0m" << endl;

    double x_z[] = {1.0, 0.0, 2.0, 0.0, 3.0, 0.0};
    double alpha_z[] = {2.0, 0.0};
    cblas_zscal(3, alpha_z, x_z, 1);
    cout << "cblas_zscal: \033[32mPASSED\033[0m" << endl;
    return 0;
}
int test_copy() {
    cout << "=== copy ===" << endl;
    float x_s[] = {1.0, 2.0, 3.0};
    float y_s[3];
    cblas_scopy(3, x_s, 1, y_s, 1);
    cout << "cblas_scopy: \033[32mPASSED\033[0m" << endl;

    double x_d[] = {1.0, 2.0, 3.0};
    double y_d[3];
    cblas_dcopy(3, x_d, 1, y_d, 1);
    cout << "cblas_dcopy: \033[32mPASSED\033[0m" << endl;

    float x_c[] = {1.0, 0.0, 2.0, 0.0, 3.0, 0.0}; 
    float y_c[6];
    cblas_ccopy(3, x_c, 1, y_c, 1);
    cout << "cblas_ccopy: \033[32mPASSED\033[0m" << endl;

    double x_z[] = {1.0, 0.0, 2.0, 0.0, 3.0, 0.0};
    double y_z[6];
    cblas_zcopy(3, x_z, 1, y_z, 1);
    cout << "cblas_zcopy: \033[32mPASSED\033[0m" << endl;
    return 0;
}

int test_swap() {
    cout << "=== swap ===" << endl;
    float x_s[] = {1.0, 2.0, 3.0};
    float y_s[] = {4.0, 5.0, 6.0};
    cblas_sswap(3, x_s, 1, y_s, 1);
    cout << "cblas_sswap: \033[32mPASSED\033[0m" << endl;

    double x_d[] = {1.0, 2.0, 3.0};
    double y_d[] = {4.0, 5.0, 6.0};
    cblas_dswap(3, x_d, 1, y_d, 1);
    cout << "cblas_dswap: \033[32mPASSED\033[0m" << endl;

    float x_c[] = {1.0, 0.0, 2.0, 0.0, 3.0, 0.0}; 
    float y_c[] = {4.0, 0.0, 5.0, 0.0, 6.0, 0.0};
    cblas_cswap(3, x_c, 1, y_c, 1);
    cout << "cblas_cswap: \033[32mPASSED\033[0m" << endl;

    double x_z[] = {1.0, 0.0, 2.0, 0.0, 3.0, 0.0};
    double y_z[] = {4.0, 0.0, 5.0, 0.0, 6.0, 0.0};
    cblas_zswap(3, x_z, 1, y_z, 1);
    cout << "cblas_zswap: \033[32mPASSED\033[0m" << endl;
    return 0;
}

int test_dot() {
    cout << "=== dot ===" << endl;
    float x_s[] = {1.0, 2.0, 3.0};
    float y_s[] = {4.0, 5.0, 6.0};
    float result_s = cblas_sdot(3, x_s, 1, y_s, 1);
    cout << "cblas_sdot: \033[32mPASSED\033[0m" << endl;

    double x_d[] = {1.0, 2.0, 3.0};
    double y_d[] = {4.0, 5.0, 6.0};
    double result_d = cblas_ddot(3, x_d, 1, y_d, 1);
    cout << "cblas_ddot: \033[32mPASSED\033[0m" << endl;
    return 0;
}

int test_nrm2() {
    cout << "=== nrm2 ===" << endl;
    float x_s[] = {1.0, 2.0, 3.0};
    float result_s = cblas_snrm2(3, x_s, 1);
    cout << "cblas_snrm2: \033[32mPASSED\033[0m" << endl;

    double x_d[] = {1.0, 2.0, 3.0};
    double result_d = cblas_dnrm2(3, x_d, 1);
    cout << "cblas_dnrm2: \033[32mPASSED\033[0m" << endl;

    float x_c[] = {1.0, 0.0, 2.0, 0.0, 3.0, 0.0}; 
    float result_c = cblas_scnrm2(3, x_c, 1);
    cout << "cblas_scnrm2: \033[32mPASSED\033[0m" << endl;

    double x_z[] = {1.0, 0.0, 2.0, 0.0, 3.0, 0.0};
    double result_z = cblas_dznrm2(3, x_z, 1);
    cout << "cblas_dznrm2: \033[32mPASSED\033[0m" << endl;
    return 0;
}

int test_asum() {
    cout << "=== asum ===" << endl;
    float x_s[] = {1.0, -2.0, 3.0};
    float result_s = cblas_sasum(3, x_s, 1);
    cout << "cblas_sasum: \033[32mPASSED\033[0m" << endl;

    double x_d[] = {1.0, -2.0, 3.0};
    double result_d = cblas_dasum(3, x_d, 1);
    cout << "cblas_dasum: \033[32mPASSED\033[0m" << endl;

    float x_c[] = {1.0, -1.0, 2.0, -2.0, 3.0, -3.0}; 
    float result_c = cblas_scasum(3, x_c, 1);
    cout << "cblas_scasum: \033[32mPASSED\033[0m" << endl;

    double x_z[] = {1.0, -1.0, 2.0, -2.0, 3.0, -3.0};
    double result_z = cblas_dzasum(3, x_z, 1);
    cout << "cblas_dzasum: \033[32mPASSED\033[0m" << endl;
    return 0;
}

int test_iamax() {
    cout << "=== iamax ===" << endl;
    float x_s[] = {1.0, -2.0, 3.0};
    int result_s = cblas_isamax(3, x_s, 1);
    cout << "cblas_isamax: \033[32mPASSED\033[0m" << endl;

    double x_d[] = {1.0, -2.0, 3.0};
    int result_d = cblas_idamax(3, x_d, 1);
    cout << "cblas_idamax: \033[32mPASSED\033[0m" << endl;

    float x_c[] = {1.0, -1.0, 2.0, -2.0, 3.0, -3.0}; 
    int result_c = cblas_icamax(3, x_c, 1);
    cout << "cblas_icamax: \033[32mPASSED\033[0m" << endl;

    double x_z[] = {1.0, -1.0, 2.0, -2.0, 3.0, -3.0};
    int result_z = cblas_izamax(3, x_z, 1);
    cout << "cblas_izamax: \033[32mPASSED\033[0m" << endl;
    return 0;
}

int test_rot() {
    cout << "=== rot ===" << endl;
    float x_s[] = {1.0, 2.0, 3.0};
    float y_s[] = {4.0, 5.0, 6.0};
    cblas_srot(3, x_s, 1, y_s, 1, 0.5f, 0.5f);
    cout << "cblas_srot: \033[32mPASSED\033[0m" << endl;

    double x_d[] = {1.0, 2.0, 3.0};
    double y_d[] = {4.0, 5.0, 6.0};
    cblas_drot(3, x_d, 1, y_d, 1, 0.5, 0.5);
    cout << "cblas_drot: \033[32mPASSED\033[0m" << endl;
    return 0;
}

int test_rotm() {
    cout << "=== rotm ===" << endl;
    float x_s[] = {1.0, 2.0, 3.0};
    float y_s[] = {4.0, 5.0, 6.0};
    float param_s[] = {1.0f, 0.5f, 0.5f, 0.5f, 0.5f}; 
    cblas_srotm(3, x_s, 1, y_s, 1, param_s);
    cout << "cblas_srotm: \033[32mPASSED\033[0m" << endl;

    double x_d[] = {1.0, 2.0, 3.0};
    double y_d[] = {4.0, 5.0, 6.0};
    double param_d[] = {1.0, 0.5, 0.5, 0.5, 0.5}; 
    cblas_drotm(3, x_d, 1, y_d, 1, param_d);
    cout << "cblas_drotm: \033[32mPASSED\033[0m" << endl;
    return 0;
}

int test_rotg() {
    cout << "=== rotg ===" << endl;
    float a_s = 1.0f, b_s = 2.0f, c_s, s_s;
    cblas_srotg(&a_s, &b_s, &c_s, &s_s);
    cout << "cblas_srotg: \033[32mPASSED\033[0m" << endl;

    double a_d = 1.0, b_d = 2.0, c_d, s_d;
    cblas_drotg(&a_d, &b_d, &c_d, &s_d);
    cout << "cblas_drotg: \033[32mPASSED\033[0m" << endl;
    return 0;
}

int main() {
    test_axpy();
    test_scal();
    test_copy();
    test_swap();
    test_dot();
    test_nrm2();
    test_asum();
    test_iamax();
    test_rot();
    test_rotm();
    test_rotg();
}