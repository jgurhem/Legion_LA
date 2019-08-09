# Legion_LA
This repository contains a few Linear Algebra applications (and and especially a block-based LU factorization) implemented in Legion

# Prerequisites
[Legion](https://github.com/StanfordLegion/legion)

[Regent](http://regent-lang.org/install/)


# Installation
```
git clone https://github.com/jgurhem/Legion_LA.git
```

# Applications
## Run the LU factorization
```
cd la
regent.py blu.rg -T <nb_blocks> -N <blocksize>
```

## Run the 1D Laplacian
Make sure you set up the `LG_RT_DIR` variable. It locates Legion `runtime` directory.

```
cd lap1D
make
./lap
```


# License

[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)

- **[MIT license](http://opensource.org/licenses/mit-license.php)**
