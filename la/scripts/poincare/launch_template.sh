#@ class            = clallmds+
#@ job_name         = legion-run
#@ total_tasks      = NCORES
#@ node             = NNODES
#@ wall_clock_limit = TIMEWALL
#@ output           = $(job_name).$(jobid).log
#@ error            = $(job_name).$(jobid).err
#@ job_type         = mpich
#@ environment      = COPY_ALL
#@ node_usage       = not_shared
#@ queue

. scripts/poincare/load_env.sh
sh scripts/run_openmpi.sh Poincare blockLU blu.rg NNODES NCORES NBLOCKS DATASIZE BLOCKSIZE test_legion_DATASIZE.json

