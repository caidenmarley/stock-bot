cmake_minimum_required(VERSION 3.24)
project(stock_bot)

include(FetchContent)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# enable warnings
if(MSVC)
    add_compile_options(/W4 /permissive-)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Werror)
endif()

include_directories(include)

file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS
    src/*/*.cpp
)

find_package(Eigen3 3.3 QUIET NO_MODULE)

# fetch eigen if it is not installed
if(NOT TARGET Eigen3::Eigen)
	FetchContent_Declare(
		Eigen3
  		GIT_REPOSITORY https://gitlab.com/libeigen/eigen
		GIT_TAG        c3f9707824a97ad04fa45bd446610ee5341bf077
		OVERRIDE_FIND_PACKAGE
	)

	FetchContent_MakeAvailable(Eigen3)

	find_package(Eigen3 3.3 REQUIRED NO_MODULE)
endif()

find_package(fmt QUIET)

if(NOT fmt::fmt)
	FetchContent_Declare(
    	fmt
    	GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    	GIT_TAG 11.0.2 # pick a release
	)
	FetchContent_MakeAvailable(fmt)
endif()


# include_directories(${EIGEN3_INCLUDE_DIR})

link_libraries(PRIVATE fmt::fmt)
link_libraries(Eigen3::Eigen)

# create main exe, cmake --build build --target stock_bot
add_executable(stock_bot src/main.cpp ${SRC_FILES})
# create testbed exe, cmake --build build --target testbed
add_executable(testbed tests/testbed.cpp ${SRC_FILES})
