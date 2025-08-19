# PointSampler

`PointSampler` is a lightweight C++ header-only library for generating and manipulating points in arbitrary dimensions (1D, 2D, 3D, ... N-dimensional). It includes utilities for random sampling, jittered grids, importance sampling, density warping, spatial relaxation, and point clustering (labeling).

The library is designed to be generic, extensible, and dimensionality-independent via `std::array` and C++ templates.

Documentation is available here: [otto-link.github.io/PointSampler](https://otto-link.github.io/PointSampler/).

## Features

- ✔️ Random sampling in arbitrary dimensions  
- ✔️ Grid-based sampling with jitter and stagger  
- ✔️ Importance sampling via rejection or resampling
- ✔️ Poisson disc sampling (uniform and non-uniform)
- ✔️ Point cloud filtering and normalization  
- ✔️ Density-based spatial distribution
- ✔️ Relaxation using k-nearest neighbors (with `nanoflann`)  
- ✔️ Dimension-agnostic (supports any `N`)
- ✔️ Utility wrappers for common spatial metrics (nearest neighbors, minimum distance, etc.)
- ✔️ Utility wrappers for common clustering technics (k-means, DBSCAN, etc.)
- ...

![visu](https://github.com/user-attachments/assets/90d21e6e-34e6-4354-9eba-b9f7a0dd5b56)

## Example

```cpp
#include <iostream>
#include "point_sampler.hpp"

int main()
{
  // Generate 5 random points in 3D space
  constexpr size_t dim = 3;
  size_t count = 5;
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

### Add PointSampler to Your Project

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

### PointSampler Requirements

* **C++20** compiler
* `nanoflann` (embedded as an external)
* `dkm` (embedded as an external)

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
* [dkm](https://github.com/genbattle/dkm) - used for k-means clustering

## License

This project is licensed under the GPL-3.0 license.
