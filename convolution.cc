// FFTW3 convolution

#include <iostream>
#include <fftw3.h>
#include <vector>

using namespace std;

void print_vec(const vector<double>& v) {
	for (vector<double>::const_iterator i = v.begin(); i != v.end(); ++i) {
		cout << *i << "\t";
	}
	cout << endl;
}

void print_array(const double* a, int length) {
	for (int i = 0; i < length; ++i) {
		cout << a[i] << "\t";
	}
	cout << endl;
}

void vec_to_array(vector<double>& v, double* a, const int allocated_size) {
	int j=0;
	for (vector<double>::const_iterator i = v.begin(); i != v.end(); ++i) {
		a[j++] = *i;
	}

	// initialize the rest with 0.0
	while (j < allocated_size) {
		a[j++] = 0.0;
	}
}

vector<double> convolution(vector<double>& h, vector<double>& x) {
	const int n = h.size() + x.size() - 1;
	double* vec1 = (double *)fftw_malloc(sizeof(double) * n);
	double* vec2 = (double *)fftw_malloc(sizeof(double) * n);
	double* vec3 = (double *)fftw_malloc(sizeof(double) * n);

	vec_to_array(h, vec1, n);
	vec_to_array(x, vec2, n);

	// Fourier transform of h
	fftw_plan pl1 = fftw_plan_r2r_1d(n,
									vec1,
									vec3,
									FFTW_R2HC,
									FFTW_ESTIMATE);
	fftw_execute(pl1);

	// Fourier transform of x
	fftw_plan pl2 = fftw_plan_r2r_1d(n,
									vec2,
									vec1,
									FFTW_R2HC,
									FFTW_ESTIMATE);
	fftw_execute(pl2);

	// multiplying
	vec2[0] = vec1[0] * vec3[0];
	for (int k = 1; k <= n/2; ++k) {
		double x1 = vec1[k];
		double x2 = vec3[k];
		double y1 = vec1[n-k];
		double y2 = vec3[n-k];

		vec2[k] = (x1*x2 - y1*y2);
		vec2[n-k] = (x1*y2 + y1*x2);
	}

	// Inverse Fourier transform
	fftw_plan invpl = fftw_plan_r2r_1d(n,
									vec2,
									vec1,
									FFTW_HC2R,
									FFTW_ESTIMATE);
	fftw_execute(invpl);

	vector<double> result;
	double factor = 1.0/n;
	for (int i=0; i < n; ++i) {
		result.push_back(vec1[i] * factor);
	}


	// cleaning
	fftw_destroy_plan(pl1);
	fftw_destroy_plan(pl2);
	fftw_destroy_plan(invpl);

	fftw_free(vec3);
	fftw_free(vec2);
	fftw_free(vec1);

	return result;
}

int main() {
	vector<double> h, x;
	h.push_back(3); h.push_back(2); h.push_back(1);
	x.push_back(2); x.push_back(-1); x.push_back(1);

	print_vec(h);
	print_vec(x);
	print_vec(convolution(h,x));
}
