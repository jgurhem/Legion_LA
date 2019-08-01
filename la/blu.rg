import "regent"
local c = regentlib.c
local cstring = terralib.includec("string.h")
local std = terralib.includec("stdlib.h")
local stdio = terralib.includec("stdio.h")
local systime = terralib.includec("sys/time.h")
require("bla_common")

task main()
  var nt : int32 = 4
  var np : int32 = 4
  var save : bool = false
  var printm : bool = false
  var te : systime.timeval
  var ts : systime.timeval
  var tz : systime.timezone
  var args = c.legion_runtime_get_input_args()
  for i = 0, args.argc do
    if cstring.strcmp(args.argv[i], "-N") == 0 then
      np = std.atoi(args.argv[i + 1])
    elseif cstring.strcmp(args.argv[i], "-T") == 0 then
      nt = std.atoi(args.argv[i + 1])
    elseif cstring.strcmp(args.argv[i], "-S") == 0 then
      save = true
    elseif cstring.strcmp(args.argv[i], "-P") == 0 then
      printm = true
    end
  end

  var init : int32 = 0

  var gridA = ispace(int2d, { x = nt * np, y = nt * np })
  var tilesA = ispace(int2d, { x = nt, y = nt })
  var A = region(gridA, double)
  var A_p = make_partition_mat(A, tilesA, np)

  var gridA_inv = ispace(int2d, { x = nt * np, y = np })
  var tilesA_inv = ispace(int2d, { x = nt, y = 1 })
  var A_inv = region(gridA_inv, double)
  var A_inv_p = make_partition_mat(A_inv, tilesA_inv, np)

  init_mat(A)
  if save then
    save_mat("a.bin", A, nt * np)
  end
  if printm then
    print_mat("a", A, nt * np)
  end
  systime.gettimeofday(&ts, &tz)
  for k = 0, nt-1 do
    inversion(A_p[int2d({k, k})], A_inv_p[int2d({k, 0})], np)
    for i = k + 1, nt do
      pmm(A_p[int2d({i, k})], A_inv_p[int2d({k, 0})], np)
      for j = k + 1, nt do
        pmm_d(A_p[int2d({i, j})], A_p[int2d({i, k})], A_p[int2d({k, j})], np)
      end
    end
  end
  systime.gettimeofday(&te, &tz)
  stdio.printf("%f\n", (te.tv_sec - ts.tv_sec) + (te.tv_usec - ts.tv_usec) / 1000000.0);
  if save then
    save_mat("lu.bin", A, nt * np)
  end
  if printm then
    print_mat("lu", A, nt * np)
  end

end

regentlib.start(main)
