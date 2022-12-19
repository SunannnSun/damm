# Dirichlet Process Mixture Model
# Parallel Implementation

Tested on MacOS12.6 Apple M1 Pro

Apple has explicitly disabled OpenMP support in compilers that they ship in Xcode.

```
brew install llvm
```

<!-- export CC="/usr/local/Cellar/llvm/15.0.3/bin/clang"
export CXX="/usr/local/Cellar/llvm/15.0.3/bin/clang++" 
export LDFLAGS="-L/usr/local/Cellar/llvm/15.0.3/lib"
export CPPFLAGS="-I/usr/local/Cellar/llvm/15.0.3/include" -->

Create temporary environment variables (Note: locate the correct path to "arm64" Brew)
```
export CC="/opt/homebrew/Cellar/llvm/15.0.6/bin/clang"
export CXX="/opt/homebrew/Cellar/llvm/15.0.6/bin/clang++" 
export LDFLAGS="-L/opt/homebrew/Cellar/llvm/15.0.6/lib"
export CPPFLAGS="-I/opt/homebrew/Cellar/llvm/15.0.6/include"
```

Make files using cmake commands
```
mkdir build
cd build
cmake ../src -DCMAKE_OSX_ARCHITECTURES='arm64'
make
```



Alternative methods to make files
```
export CXX="/usr/local/Cellar/llvm/15.0.3/bin/clang++"; cmake ../src
```
or
```
cmake ../src -DCMAKE_CXX_COMPILER="/usr/local/Cellar/llvm/15.0.3/bin/clang++"
```
or
```
CXX="/usr/local/Cellar/llvm/15.0.3/bin/clang++" cmake ../src
```
