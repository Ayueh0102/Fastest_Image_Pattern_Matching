// SPDX-License-Identifier: BSD-2-Clause
// Compile the private quantizer in this test TU to verify antipodal symmetry at
// half-bin boundaries without expanding the production public API.
#include "../experiments/sparse_orientation.cpp"
#include <iostream>
int main() {
    constexpr double p=3.14159265358979323846;
    for (int k=-16;k<16;++k) {
        double angle=(k+.5)*p/8;
        int a=fpm::research::orientationBin(angle);
        int b=fpm::research::orientationBin(angle+p);
        if (a!=b) {std::cerr << k << ": " << a << " != " << b << '\n';return 1;}
    }
    for (int k=-3600;k<=3600;++k) {
        double angle=k*p/1800;
        if(fpm::research::orientationBin(angle)!=fpm::research::orientationBin(angle+p)) return 2;
    }
    std::cout << "Canonical unsigned bins agree for 32 half-bin boundaries and 7201 sampled antipodal angles\n";
}
