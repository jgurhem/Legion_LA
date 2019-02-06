#include <cstdio>
#include <cassert>
#include <cstdlib>
#include "legion.h"
#include <unistd.h>
#include <math.h>
#include <cmath>

using namespace Legion;


inline Point<1> make_point(coord_t x)
{
	Point<1> p(x);
	return p;
}

inline Point<2> make_point(coord_t x, coord_t y)
{
	long long v[] = {x, y};
	Point<2> p(v);
	return p;
}

inline Point<3> make_point(coord_t x, coord_t y, coord_t z)
{
	long long v[] = {x, y, z};
	Point<3> p(v);
	return p;
}

enum TaskIDs {
	TOP_LEVEL_TASK_ID,
	INIT_FIELD_TASK_ID,
	STENCIL_TASK_ID,
	CHECK_TASK_ID,
	SAVE_TASK_ID,
};

enum FieldIDs {
	FID_1,
	FID_2,
};

struct Args {
	public:
		int Nx;
		int Ny;
		int Bx;
		int By;
		int Tx;
		int Ty;
};

struct Args computeArgs(){
	struct Args args;
	args.Nx = 4; args.Ny = 4;
	args.Bx = 2; args.By = 2;
	const InputArgs &command_args = Runtime::get_input_args();
	for (int i = 1; i < command_args.argc; i++)
	{
		if (!strcmp(command_args.argv[i],"-nx"))
			args.Nx = atoi(command_args.argv[++i]);
		if (!strcmp(command_args.argv[i],"-ny"))
			args.Ny = atoi(command_args.argv[++i]);
		if (!strcmp(command_args.argv[i],"-bx"))
			args.Bx = atoi(command_args.argv[++i]);
		if (!strcmp(command_args.argv[i],"-by"))
			args.By = atoi(command_args.argv[++i]);
	}
	args.Tx = args.Nx * args.Bx;
	args.Ty = args.Ny * args.By;
	return args;
}

void top_level_task(const Task *task,
		const std::vector<PhysicalRegion> &regions,
		Context ctx, Runtime *runtime)
{
	struct Args args = computeArgs();

	long long int zero[2], B[2], N[2];
	zero[0] = 0; zero[1] = 0;
	N[0] = args.Tx - 1; N[1] = args.Ty - 1;
	B[0] = args.Bx - 1; B[1] = args.By - 1;

	printf("bx %d by %d\n", args.Bx, args.By);
	Point<2> elem_rect_lo(zero);
	Point<2> elem_rect_hi(N);
	Rect<2> elem_rect(elem_rect_lo, elem_rect_hi);
	IndexSpaceT<2> is = runtime->create_index_space(ctx, elem_rect);
	FieldSpace fs = runtime->create_field_space(ctx);
	{
		FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
		allocator.allocate_field(sizeof(double), FID_1);
		allocator.allocate_field(sizeof(double), FID_2);
	}
	LogicalRegion stencil_lr = runtime->create_logical_region(ctx, is, fs);

	Point<2> color_lo(zero);
	Point<2> color_hi(B);
	Rect<2> color_bounds(color_lo, color_hi);
	IndexSpaceT<2> color_is = runtime->create_index_space(ctx, color_bounds);
	IndexPartition disjoint_ip = runtime->create_equal_partition(ctx, is, color_is);
	//IndexPartition disjoint_ip = runtime->create_partition_by_blockify(ctx, is, make_point(args.Nx, args.Ny));


	Transform<2,2> transform;
	transform[0][0] = args.Nx;
	transform[0][1] = args.Nx;
	transform[1][0] = args.Ny;
	transform[1][1] = args.Ny;
	long long int extent_hi_v[2] = {args.Nx, args.Ny};
	long long int extent_lo_v[2] = {-1, -1};
	Point<2> extent_hi(extent_hi_v);
	Point<2> extent_lo(extent_lo_v);
	Rect<2> extent(extent_lo, extent_hi);
	IndexPartition ghost_ip = runtime->create_partition_by_restriction(ctx, is, color_is, transform, extent);

	LogicalPartition disjoint_lp = runtime->get_logical_partition(ctx, stencil_lr, disjoint_ip);
	LogicalPartition ghost_lp = runtime->get_logical_partition(ctx, stencil_lr, ghost_ip);

	ArgumentMap arg_map;

	IndexLauncher init_launcher(INIT_FIELD_TASK_ID, color_is, TaskArgument(NULL, 0), arg_map);
	init_launcher.add_region_requirement(RegionRequirement(disjoint_lp, 0/*projection ID*/, WRITE_DISCARD, EXCLUSIVE, stencil_lr));
	init_launcher.add_field(0, FID_1);
	runtime->execute_index_space(ctx, init_launcher);



	TaskLauncher save_launcher(SAVE_TASK_ID, TaskArgument(NULL, 0));
	save_launcher.add_region_requirement(RegionRequirement(stencil_lr, READ_ONLY, EXCLUSIVE, stencil_lr));
	save_launcher.add_field(0, FID_1);
	runtime->execute_task(ctx, save_launcher);



	IndexLauncher stencil_launcher(STENCIL_TASK_ID, color_is, TaskArgument(NULL, 0), arg_map);
	stencil_launcher.add_region_requirement(RegionRequirement(ghost_lp, 0/*projection ID*/, READ_ONLY, EXCLUSIVE, stencil_lr));
	stencil_launcher.add_field(0, FID_1);
	stencil_launcher.add_region_requirement(RegionRequirement(disjoint_lp, 0/*projection ID*/, READ_WRITE, EXCLUSIVE, stencil_lr));
	stencil_launcher.add_field(1, FID_2);
	runtime->execute_index_space(ctx, stencil_launcher);

	runtime->destroy_logical_region(ctx, stencil_lr);
	runtime->destroy_field_space(ctx, fs);
	runtime->destroy_index_space(ctx, is);
}

void init_field_task(const Task *task,
		const std::vector<PhysicalRegion> &regions,
		Context ctx, Runtime *runtime)
{
	assert(regions.size() == 1);
	assert(task->regions.size() == 1);
	assert(task->regions[0].privilege_fields.size() == 1);
	struct Args args = computeArgs();

	FieldID fid = *(task->regions[0].privilege_fields.begin());
	const FieldAccessor<WRITE_DISCARD, double, 2> acc(regions[0], fid);

	Rect<2> rect = runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
	for (PointInRectIterator<2> pir(rect); pir(); pir++){
		acc[*pir] = 0;
		if (pir[0] == 0)
			acc[*pir] = 1;
		if (pir[1] == 0)
			acc[*pir] = 2;
		if (pir[0] == args.Tx - 1)
			acc[*pir] = 3;
		if (pir[1] == args.Ty - 1)
			acc[*pir] = 4;
	}
}

void stencil_task(const Task *task,
		const std::vector<PhysicalRegion> &regions,
		Context ctx, Runtime *runtime)
{
	assert(regions.size() == 2);
	assert(task->regions.size() == 2);
	assert(task->regions[0].privilege_fields.size() == 1);
	assert(task->regions[1].privilege_fields.size() == 1);
	struct Args args = computeArgs();

	FieldID read_fid = *(task->regions[0].privilege_fields.begin());
	FieldID write_fid = *(task->regions[1].privilege_fields.begin());

	const FieldAccessor<READ_ONLY, double, 2> read_acc(regions[0], read_fid);
	const FieldAccessor<WRITE_DISCARD, double, 2> write_acc(regions[1], write_fid);

	Rect<2> rect = runtime->get_index_space_domain(ctx, task->regions[1].region.get_index_space());

	const int px = task->index_point.point_data[0];
	const int py = task->index_point.point_data[1];

	if ((px == 0) || (py == 0) || (px == args.Bx - 1) || (py == args.By - 1))
	{
		printf("Running slow stencil path for point %d,%d...\n", px, py);
		// Note in the slow path that there are checks which
		// perform clamps when necessary before reading values.
		for (PointInRectIterator<2> pir(rect); pir(); pir++)
		{
			if(pir[0] == 0 || pir[1] == 0 || pir[0] == args.Nx - 1|| pir[1] == args.Ny - 1) {
				write_acc[*pir] = read_acc[*pir];
			} else {
				double r = read_acc[make_point(pir[0] + 1, pir[1])];
				double l = read_acc[make_point(pir[0] - 1, pir[1])];
				double t = read_acc[make_point(pir[0], pir[1] + 1)];
				double b = read_acc[make_point(pir[0], pir[1] - 1)];

				double result = (r + l + b + t) / 4;
				write_acc[*pir] = result;
			}
		}
	}
	else
	{
		printf("Running fast stencil path for point %d,%d...\n", px, py);
		// In the fast path, we don't need any checks
		for (PointInRectIterator<2> pir(rect); pir(); pir++)
		{
			double r = read_acc[make_point(pir[0] + 1, pir[1])];
			double l = read_acc[make_point(pir[0] - 1, pir[1])];
			double t = read_acc[make_point(pir[0], pir[1] + 1)];
			double b = read_acc[make_point(pir[0], pir[1] - 1)];

			double result = (r + l + b + t) / 4;
			write_acc[*pir] = result;
		}
	}
}

void check_task(const Task *task,
		const std::vector<PhysicalRegion> &regions,
		Context ctx, Runtime *runtime)
{

}

void save_task(const Task *task,
		const std::vector<PhysicalRegion> &regions,
		Context ctx, Runtime *runtime)
{
	assert(regions.size() == 1);
	assert(task->regions.size() == 1);
	assert(task->regions[0].privilege_fields.size() == 1);
	struct Args args = computeArgs();

	FieldID fid = *(task->regions[0].privilege_fields.begin());
	const FieldAccessor<READ_ONLY, double, 2> acc(regions[0], fid);
	FILE *f = fopen("res.bin", "w");
	for(int i = 0; i < args.Tx; i++){
		for(int j = 0; j < args.Ty; j++){
			double v = acc[make_point(i, j)];
			fwrite(&v, 1, sizeof(double), f);
			printf("%2.5lf ", v);
		}
		printf("\n");
	}
	fclose(f);
}

int main(int argc, char **argv)
{
	Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

	{
		TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
		registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
		Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
	}

	{
		TaskVariantRegistrar registrar(INIT_FIELD_TASK_ID, "init_field");
		registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
		registrar.set_leaf();
		Runtime::preregister_task_variant<init_field_task>(registrar, "init_field");
	}

	{
		TaskVariantRegistrar registrar(STENCIL_TASK_ID, "stencil");
		registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
		registrar.set_leaf();
		Runtime::preregister_task_variant<stencil_task>(registrar, "stencil");
	}

	{
		TaskVariantRegistrar registrar(CHECK_TASK_ID, "check");
		registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
		registrar.set_leaf();
		Runtime::preregister_task_variant<check_task>(registrar, "check");
	}

	{
		TaskVariantRegistrar registrar(SAVE_TASK_ID, "save");
		registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
		registrar.set_leaf();
		Runtime::preregister_task_variant<save_task>(registrar, "save");
	}

	return Runtime::start(argc, argv);
}
