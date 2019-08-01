
module purge
module load python/anaconda3-5.0.1 git gnu/7.3.0 gnu-env/7.3.0 cmake/3.14.1-gnu54 openmpi

export RUNTIME=${HOME}/install/legion/runtime
export LG_RT_DIR=${HOME}/install/legion/runtime
export PATH=${PATH}:/gpfshome/mds/staff/jgurhem/install/legion/language:/gpfshome/mds/staff/jgurhem/install/legion/language/src
export TERRA_PATH=${PATH}:/gpfshome/mds/staff/jgurhem/install/legion/language:/gpfshome/mds/staff/jgurhem/install/legion/language/src

