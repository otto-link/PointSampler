/*
Simple benchmarks for dkm.hpp

This is just simple benchmarking harness without any external dependencies.
*/

#include "../../include/dkm.hpp"
#include "../../include/dkm_parallel.hpp"
#include "../../include/dkm_utils.hpp"
#include "opencv2/opencv.hpp"

#include <vector>
#include <array>
#include <tuple>
#include <string>
#include <iostream>
#include <chrono>
#include <numeric>

template <typename T, size_t N>
void print_result_dkm(std::tuple<std::vector<std::array<T, N>>, std::vector<uint32_t>>& result) {
	std::cout << "centers: ";
	for (const auto& c : std::get<0>(result)) {
		std::cout << "(";
		for (auto v : c) {
			std::cout << v << ",";
		}
		std::cout << "), ";
	}
	std::cout << std::endl;
}

cv::Mat load_opencv(const std::string& path) {
	std::ifstream file(path);
	cv::Mat data;
	for (auto it = std::istream_iterator<std::string>(file); it != std::istream_iterator<std::string>(); ++it) {
		auto split = dkm::details::split_commas(*it);
		if (split.size() != 2) { // number of values in file must match expected row size
			return cv::Mat();
		}
		cv::Vec<float, 2> values;
		for (auto i = 0; i < 2; ++i) {
			values[i] = std::stof(split[i]);
		}
		data.push_back(values);
	}
	return data;
}

std::chrono::duration<double> profile_opencv(const cv::Mat& data, int k) {
	auto start = std::chrono::high_resolution_clock::now();
	// run the bench 10 times and take the average
	for (int i = 0; i < 10; ++i) {
		std::cout << "." << std::flush;
		cv::Mat centers, labels;
		cv::kmeans(
			data, k, labels, cv::TermCriteria(cv::TermCriteria::EPS, 0, 0.01), 1, cv::KMEANS_PP_CENTERS, centers);
		(void)labels;
	}
	auto end = std::chrono::high_resolution_clock::now();
	return (end - start) / 10.0;
}

template <typename T, size_t N>
std::chrono::duration<double> profile_dkm(const std::vector<std::array<T, N>>& data, int k) {
	auto start = std::chrono::high_resolution_clock::now();
	// run the bench 10 times and take the average
	for (int i = 0; i < 10; ++i) {
		std::cout << "." << std::flush;
		auto result = dkm::kmeans_lloyd(data, k);
		(void)result;
	}
	auto end = std::chrono::high_resolution_clock::now();
	return (end - start) / 10.0;
}

template <typename T, size_t N>
std::chrono::duration<double> profile_dkm_par(const std::vector<std::array<T, N>>& data, int k) {
	auto start = std::chrono::high_resolution_clock::now();
	// run the bench 10 times and take the average
	for (int i = 0; i < 10; ++i) {
		std::cout << "." << std::flush;
		auto result = dkm::kmeans_lloyd_parallel(data, k);
		(void)result;
	}
	auto end = std::chrono::high_resolution_clock::now();
	return (end - start) / 10.0;
}

template <typename T, size_t N>
void bench_dataset(const std::string& path, uint32_t k) {
	std::cout << "## Dataset " << path << " ##" << std::endl;
	std::chrono::duration<double> time_opencv;
	if (N == 2) {
		auto cv_data = load_opencv(path);
		time_opencv = profile_opencv(cv_data, k);
	}

	auto dkm_data = dkm::load_csv<T, N>(path);
	auto time_dkm = profile_dkm(dkm_data, k);
	auto time_dkm_par = profile_dkm_par(dkm_data, k);
	std::cout << "\n";
	std::cout << "DKM: " << std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(time_dkm).count()
			  << "ms" << std::endl;
	std::cout << "DKM parallel: " << std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(time_dkm_par).count()
			  << "ms" << std::endl;
	std::cout << "OpenCV: ";
	if (N == 2) {
		std::cout << std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(time_opencv).count() << "ms";
	} else {
		std::cout << "---";
	}
	std::cout << "\n" << std::endl;
}

int main() {
	std::cout << "# BEGINNING PROFILING #\n" << std::endl;
	bench_dataset<float, 2>("iris.data.csv", 3);
	bench_dataset<float, 2>("s1.data.csv", 15);
	bench_dataset<float, 2>("birch3.data.csv", 100);
	bench_dataset<float, 128>("dim128.data.csv", 16);

	return 0;
}