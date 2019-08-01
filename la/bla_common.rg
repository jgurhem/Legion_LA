import "regent"
local c = regentlib.c
local stdio = terralib.includec("stdio.h")

task make_partition_mat(points : region(ispace(int2d), double),
                        tiles : ispace(int2d), np : int32)
  var coloring = c.legion_domain_point_coloring_create()
  for i in tiles do
    var lo = int2d { x = i.x * np, y = i.y * np }
    var hi = int2d { x = (i.x + 1) * np - 1, y = (i.y + 1) * np - 1 }
    var rect = rect2d { lo = lo, hi = hi }
    c.legion_domain_point_coloring_color_domain(coloring, i, rect)
  end
  var p = partition(disjoint, points, coloring, tiles)
  c.legion_domain_point_coloring_destroy(coloring)
  return p
end

task make_partition_vect(points : region(ispace(int1d), double),
                        tiles : ispace(int1d), np : int32)
  var coloring = c.legion_domain_point_coloring_create()
  for i in tiles do
    var lo = int1d { i * np }
    var hi = int1d { (i + 1) * np - 1 }
    var rect = rect1d { lo = lo, hi = hi }
    c.legion_domain_point_coloring_color_domain(coloring, i, rect)
  end
  var p = partition(disjoint, points, coloring, tiles)
  c.legion_domain_point_coloring_destroy(coloring)
  return p
end

task init_mat(A : region(ispace(int2d), double))
where reads writes(A) do
  for i in A do
    A[i] = c.drand48()
  end
end

task print_mat(idstr : regentlib.string, A : region(ispace(int2d), double), n : int)
where reads(A) do
  for i in A.ispace do
    stdio.printf("%s %d %d %lf\n", idstr, i.x, i.y, A[i]);
  end
end

task save_mat(outfilename : regentlib.string, A : region(ispace(int2d), double), n : int)
where reads(A) do
  var file = c.fopen([rawstring](outfilename), "w")
  regentlib.assert(not isnull(file), "save : failed to open file")
  var wdbl : &double = [&double](c.malloc(8))

  for i = 0, n do
    for j = 0, n do
      @wdbl = A[int2d({i, j}) + A.bounds.lo]
      c.fwrite(wdbl, 8, 1, file)
    end
  end

  c.fclose(file)
  c.free(wdbl)
end

task pmm(A_in : region(ispace(int2d), double), B : region(ispace(int2d), double), n : int)
where reads(B), reads writes(A_in) do
  var A = region(A_in.ispace, double)
  copy(A_in, A)
  fill(A_in, 0)
  for i = 0, n do
    for j = 0, n do
      for k = 0, n do
        A_in[int2d({i, j}) + A_in.bounds.lo] = A_in[int2d({i, j}) + A.bounds.lo] + A[int2d({i, k}) + A.bounds.lo] * B[int2d({k, j}) + B.bounds.lo]
      end
    end
  end
end

task pmm_a(A : region(ispace(int2d), double), B : region(ispace(int2d), double), C : region(ispace(int2d), double), n : int)
where reads(B), reads(C), reads writes(A) do
  for i = 0, n do
    for j = 0, n do
      for k = 0, n do
        A[int2d({i, j}) + A.bounds.lo] = A[int2d({i, j}) + A.bounds.lo] + B[int2d({i, k}) + B.bounds.lo] * C[int2d({k, j}) + C.bounds.lo]
      end
    end
  end
end

task pmm_d(A : region(ispace(int2d), double), B : region(ispace(int2d), double), C : region(ispace(int2d), double), n : int)
where reads(B), reads(C), reads writes(A) do
  for i = 0, n do
    for j = 0, n do
      for k = 0, n do
        A[int2d({i, j}) + A.bounds.lo] = A[int2d({i, j}) + A.bounds.lo] - B[int2d({i, k}) + B.bounds.lo] * C[int2d({k, j}) + C.bounds.lo]
      end
    end
  end
end

task inversion(A_in : region(ispace(int2d), double), B : region(ispace(int2d), double), n : int)
where reads(A_in), reads writes(B) do
  fill(B, 0)
  var A = region(A_in.ispace, double)
  copy(A_in, A)
  var tmp : double
  for i = 0, n do
    B[int2d({i, i}) + B.bounds.lo] = 1
  end
  for k = 0, n do
    tmp = A[int2d({k, k}) + A.bounds.lo]
    for j = 0, n do
      A[int2d({k, j}) + A.bounds.lo] = A[int2d({k, j}) + A.bounds.lo] / tmp
      B[int2d({k, j}) + B.bounds.lo] = B[int2d({k, j}) + B.bounds.lo] / tmp
    end
    for i = 0, k do
      tmp = A[int2d({i, k}) + A.bounds.lo]
      for j = 0, n do
        A[int2d({i, j}) + A.bounds.lo] = A[int2d({i, j}) + A.bounds.lo] - tmp * A[int2d({k, j}) + A.bounds.lo]
        B[int2d({i, j}) + B.bounds.lo] = B[int2d({i, j}) + B.bounds.lo] - tmp * B[int2d({k, j}) + B.bounds.lo]
      end
    end
    for i = k + 1, n do
      tmp = A[int2d({i, k}) + A.bounds.lo]
      for j = 0, n do
        A[int2d({i, j}) + A.bounds.lo] = A[int2d({i, j}) + A.bounds.lo] - tmp * A[int2d({k, j}) + A.bounds.lo]
        B[int2d({i, j}) + B.bounds.lo] = B[int2d({i, j}) + B.bounds.lo] - tmp * B[int2d({k, j}) + B.bounds.lo]
      end
    end
  end
end
