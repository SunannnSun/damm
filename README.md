# Dirichlet Process Mixture Model (Parallel Implementation)

Tested on Ubuntu 20.04; i7-1065G7 CPU (8 cores) 

As opposite to using cmakelists.txt in previous iteration, new implementation on Linux uses GCC terminal commands

Package required: Eigen3.4(stable release) and boost_1_81_0; both under the system directory, or /usr/include

GCC can search for package under system directory, but both packages have unconventional names with version information, we need to specify the include path for GCC to search using the -I flag 

Eigen library is completely header-based; hence no separate compilation is needed and can be directly referenced and used once the include path is specified.

On the other hand, boost library; while most of its functionalities are defined in header files, some packages do require separate compilation and linking; e.g., boost::program_options.

Use the built-in build system from the boost:
./bootstrap.sh --help
Also, consider using the --show-libraries and --with-libraries=library-name-list options to limit the long wait you'll experience if you build everything. 
./b2 install 

The binary library, if not specified, by default will be installed under the directory usr/include/boost_1_81_0/stage/lib. Make sure then use the -L flag to specifiy the library path and use the -l flag to search for the specific library in the path


g++ src/main.cpp -I/usr/include/eigen3.4 -I/usr/include/boost_1_81_0 -o -L/usr/include/boost_1_81_0/stage/lib -lboost_program_options -o main