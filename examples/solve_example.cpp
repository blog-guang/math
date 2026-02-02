#include <iostream>
#include "vector.h"

using namespace math;

int main() {
    // --- Create two vectors ---------------------------------------------------
    Vector a = {1.0, 2.0, 3.0, 4.0, 5.0};
    Vector b = {5.0, 4.0, 3.0, 2.0, 1.0};

    std::cout << "a = " << a << "\n";
    std::cout << "b = " << b << "\n\n";

    // --- Addition & subtraction -----------------------------------------------
    Vector sum = a + b;
    Vector diff = a - b;
    std::cout << "a + b = " << sum  << "\n";
    std::cout << "a - b = " << diff << "\n\n";

    // --- Scalar multiplication ------------------------------------------------
    Vector scaled = a * 2.5;
    Vector scaled2 = 3.0 * b;
    std::cout << "a * 2.5 = " << scaled  << "\n";
    std::cout << "3.0 * b = " << scaled2 << "\n\n";

    // --- Dot product & norm ---------------------------------------------------
    double d = a.dot(b);
    double na = a.norm();
    double nb = b.norm();
    std::cout << "a . b   = " << d  << "\n";
    std::cout << "|a|     = " << na << "\n";
    std::cout << "|b|     = " << nb << "\n\n";

    // --- BLAS axpy: c = 2*a + b -----------------------------------------------
    Vector c = b;               // start with b
    c.axpy(2.0, a);             // c = 2*a + c  (i.e., 2*a + b)
    std::cout << "2*a + b = " << c << "\n\n";

    // --- Factory functions ----------------------------------------------------
    Vector z = Vector::zeros(4);
    Vector o = Vector::ones(4);
    Vector r = Vector::random(4, 42);
    std::cout << "zeros(4)       = " << z << "\n";
    std::cout << "ones(4)        = " << o << "\n";
    std::cout << "random(4, 42)  = " << r << "\n";

    return 0;
}
