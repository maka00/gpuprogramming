// The code is taken from the OpenCV documentation
// An OpenCV with CUDA support is needed
// The app creates a image with a random noise and stores it as a PNG

#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/copy.h>
#include <thrust/random.h>
#include <opencv4/opencv2/core/utility.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/core/cuda.hpp>

template<typename T>
struct step_functor : public thrust::unary_function<int, int> {
    int columns;
    int step;
    int channels;

    __host__ __device__ step_functor(int columns_, int step_, int channels_ = 1) : columns(columns_), step(step_),
                                                                                   channels(channels_) {};

    __host__ step_functor(cv::cuda::GpuMat &mat) {
        CV_Assert(mat.depth() == cv::DataType<T>::depth);
        columns = mat.cols;
        step = mat.step / sizeof(T);
        channels = mat.channels();
    }

    __host__ __device__
    int operator()(int x) const {
        int row = x / columns;
        int idx = (row * step) + (x % columns) * channels;
        return idx;
    }
};

template<typename T>
thrust::permutation_iterator<thrust::device_ptr<T>, thrust::transform_iterator<step_functor<T>, thrust::counting_iterator<int>>>
GpuMatEndItr(cv::cuda::GpuMat mat, int channel = 0) {
    if (channel == -1) {
        mat = mat.reshape(1);
        channel = 0;
    }
    CV_Assert(mat.depth() == cv::DataType<T>::depth);
    CV_Assert(channel < mat.channels());
    return thrust::make_permutation_iterator(thrust::device_pointer_cast(mat.ptr<T>(0) + channel),
                                             thrust::make_transform_iterator(
                                                     thrust::make_counting_iterator(mat.rows * mat.cols),
                                                     step_functor<T>(mat.cols, mat.step / sizeof(T), mat.channels())));
}

template<typename T>
thrust::permutation_iterator<thrust::device_ptr<T>, thrust::transform_iterator<step_functor<T>, thrust::counting_iterator<int>>>
GpuMatBeginItr(cv::cuda::GpuMat mat, int channel = 0) {
    if (channel == -1) {
        mat = mat.reshape(1);
        channel = 0;
    }
    CV_Assert(mat.depth() == cv::DataType<T>::depth);
    CV_Assert(channel < mat.channels());
    return thrust::make_permutation_iterator(thrust::device_pointer_cast(mat.ptr<T>(0) + channel),
                                             thrust::make_transform_iterator(thrust::make_counting_iterator(0),
                                                                             step_functor<T>(mat.cols,
                                                                                             mat.step / sizeof(T),
                                                                                             mat.channels())));
}

struct prg {
    float a, b;

    __host__ __device__
    prg(float _a = 0.f, float _b = 1.f) : a(_a), b(_b) {};

    __host__ __device__
    float operator()(const unsigned int n) const {
        thrust::default_random_engine rng;
        thrust::uniform_real_distribution<float> dist(a, b);
        rng.discard(n);
        return dist(rng);
    }
};

int main(void) {
    cv::cuda::GpuMat d_value(1, 100, CV_32F);
    auto valueBegin = GpuMatBeginItr<float>(d_value);
    auto valueEnd = GpuMatEndItr<float>(d_value);
    thrust::transform(thrust::make_counting_iterator(0), thrust::make_counting_iterator(d_value.cols), valueBegin,
                      prg(-1, 1));
    cv::Mat values(d_value);
    cv::imwrite("test.png", values);
    return 0;
}