project('testt', 'cpp',default_options : ['cpp_std=c++latest', 'warning_level=3'])

# only for clang
#add_project_arguments('-ftime-trace', language:'cpp')
add_project_arguments('/arch:AVX512', language : 'cpp')

inc_dir = include_directories('include')

libglfw_project = subproject('glfw-3.3.2')
libglfw_dep = libglfw_project.get_variable('glfw_dep')

project_dir  = meson.current_source_dir() # MESON_SOURCE_ROOT
build_dir    = meson.current_build_dir()  # MESON_BUILD_ROOT

fftw_path = join_paths(project_dir, 'libraries/fftw')

compiler = meson.get_compiler('cpp')

libfftw_dep = compiler.find_library('fftw3-3', dirs: fftw_path)

inc_dir = [inc_dir]

source_files = [
	'src/main.cc',
	'src/glad.cc'
]

executable('run',
  source_files,
  include_directories: inc_dir,
  dependencies : [libglfw_dep, libfftw_dep]
  )