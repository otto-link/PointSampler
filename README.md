# PointSampler

`PointSampler` is a lightweight C++ header-only library for generating and manipulating points in arbitrary dimensions (1D, 2D, 3D, ... N-dimensional). It includes utilities for random sampling, jittered grids, importance sampling, density warping, and spatial relaxation.

The library is designed to be generic, extensible, and dimensionality-independent via `std::array` and C++ templates.

## Features

- ✔️ Random sampling in arbitrary dimensions  
- ✔️ Grid-based sampling with jitter and stagger  
- ✔️ Importance sampling via rejection or resampling  
- ✔️ Point cloud filtering and normalization  
- ✔️ Density-based spatial distribution
- ✔️ Relaxation using k-nearest neighbors (with nanoflann)  
- ✔️ Dimension-agnostic (supports any `N`)
- ...

## Example

```cpp
#include <iostream>
#include "point_sampler.hpp"

int main()
{
  // Generate 5 random points in 3D space
  constexpr std::size_t dim = 3;
  std::size_t count = 5;
  unsigned int seed = 42;

  std::array<std::pair<float, float>, dim> ranges = {
      std::make_pair(-1.f, 1.f),
      std::make_pair(-2.f, 2.f),
      std::make_pair(0.f, 1.f)
  };

  std::vector<ps::Point<float, dim>> points = ps::random<float, dim>(count, ranges, seed);

  for (const auto &point : points)
  {
    std::cout << "(" << point[0] << ", " << point[1] << ", " << point[2] << ")\n";
  }

  return 0;
}
```

## Build Instructions (for Developers)

To build the `PointSampler` library locally, follow the steps below.

### Clone the repository

```bash
git clone https://github.com/otto-link/PointSampler.git
```

### Configure and build with CMake

```bash
cd PointSampler
mkdir build && cd build
cmake ..
make
```

### CMake Requirements

| Tool   | Minimum Version                             |
| ------ | ------------------------------------------- |
| CMake  | 3.16                                        |
| C++    | 20                                          |
| spdlog | Installed and available as `spdlog::spdlog` |

You can install dependencies using your package manager, e.g., on Ubuntu:

```bash
sudo apt install libspdlog-dev
```

### Run a sample

The project includes a test executable:

```bash
./bin/test_point_sampler
```

## Integrating PointSampler with CMake

This guide shows how to integrate it into your own CMake project.

### Project Structure (Assumption)

Your library lives in a subdirectory, e.g.:

```
my-app/
├── CMakeLists.txt
├── external/
│   └── PointSampler/    # Contains this library
```

### Step 1: Add PointSampler to Your Project

In your **top-level `CMakeLists.txt`**:

```cmake
# Add the PointSampler directory
add_subdirectory(external/PointSampler)

# Link to your target
target_link_libraries(my_app PRIVATE point_sampler)
```

### Optional: Use as External Project

If you're not vendoring the source, you can also use **FetchContent**:

```cmake
include(FetchContent)

FetchContent_Declare(
    PointSampler
    GIT_REPOSITORY https://github.com/otto-link/PointSampler.git
    GIT_TAG        main
)

FetchContent_MakeAvailable(PointSampler)

target_link_libraries(my_app PRIVATE point_sampler)
```

> Don't forget to add `FetchContent_Declare` before `project()` if you're using CMake 3.14+ and need reproducible builds.

### QSliderX Requirements

* **C++20** compiler
* `spdlog` (must be available as `spdlog::spdlog` target)

Make sure the following are available in your project:

```cmake
find_package(spdlog REQUIRED)
```

## Where to Find Examples

More comprehensive examples and usage for all key functions are available in:

```
tests/test_point_sampler/main.cpp
```

This includes:

* Grid-based sampling with jitter and stagger
* Importance resampling with density functions
* Relaxation using nearest neighbors
* Spatial warping via coordinate transforms
* Filtering and range normalization
* ...

## Dependencies

* [nanoflann](https://github.com/jlblancoc/nanoflann) — used for k-nearest neighbor acceleration in relaxation
* [spdlog](https://github.com/gabime/spdlog) — optional logging (already linked if you use `point_sampler` target)

## License

MIT License. See `LICENSE` file for details.
