import "regent"
local c = regentlib.c
require("bla_common")

task main()
  var nt : int32 = 4
  var np : int32 = 4
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
  print_mat("a.bin", A, nt * np)
  for k = 0, nt-1 do
    inversion(A_p[int2d({k, k})], A_inv_p[int2d({k, 0})], np)
    for i = k + 1, nt do
      pmm(A_p[int2d({i, k})], A_inv_p[int2d({k, 0})], np)
      for j = k + 1, nt do
        pmm_d(A_p[int2d({i, j})], A_p[int2d({i, k})], A_p[int2d({k, j})], np)
      end
    end
  end
  print_mat("lu.bin", A, nt * np)

end

regentlib.start(main)
