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

  var gridB = ispace(int2d, { x = nt * np, y = nt * np })
  var tilesB = ispace(int2d, { x = nt, y = nt })
  var B = region(gridB, double)
  var B_p = make_partition_mat(B, tilesB, np)

  var gridC = ispace(int2d, { x = nt * np, y = nt * np })
  var tilesC = ispace(int2d, { x = nt, y = nt })
  var C = region(gridC, double)
  var C_p = make_partition_mat(C, tilesC, np)

  init_mat(A)
  init_mat(B)
  fill(C, 0)
  print_mat("a.bin", A, nt * np)
  print_mat("b.bin", B, nt * np)
  print_mat("c.bin", C, nt * np)
  for i = 0, nt do
    for j = 0, nt do
      for k = 0, nt do
        pmm_a(C_p[int2d({i, j})], A_p[int2d({i, k})], B_p[int2d({k, j})], np)
      end
    end
  end
  print_mat("r.bin", C, nt * np)

end

regentlib.start(main)
