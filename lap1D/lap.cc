/* Copyright 2018 Stanford University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#include <cstdio>
#include <cassert>
#include <cstdlib>
#include "legion.h"

using namespace Legion;

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  INIT_FIELD_TASK_ID,
  STENCIL_TASK_ID,
};

enum FieldIDs {
  FID_VAL,
  FID_DERIV,
};

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  int num_elements = 1024;
  int num_subregions = 4;
  int num_iterations = 20;
  // Check for any command line arguments
  {
      const InputArgs &command_args = Runtime::get_input_args();
    for (int i = 1; i < command_args.argc; i++)
    {
      if (!strcmp(command_args.argv[i],"-n"))
        num_elements = atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-b"))
        num_subregions = atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-it"))
        num_iterations = atoi(command_args.argv[++i]);
    }
  }
  printf("Running stencil computation for %d elements...\n", num_elements);
  printf("Partitioning data into %d sub-regions...\n", num_subregions);

  // For this example we'll create a single logical region with two
  // fields.  We'll initialize the field identified by 'FID_VAL' with
  // our input data and then compute the derivatives stencil values 
  // and write them into the field identified by 'FID_DERIV'.
  Rect<1> elem_rect(0,num_elements-1);
  IndexSpaceT<1> is = runtime->create_index_space(ctx, elem_rect);
  FieldSpace fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, fs);
    allocator.allocate_field(sizeof(double),FID_VAL);
    allocator.allocate_field(sizeof(double),FID_DERIV);
  }
  LogicalRegion stencil_lr = runtime->create_logical_region(ctx, is, fs);
  
  // Make our color_domain based on the number of subregions
  // that we want to create.
  Rect<1> color_bounds(0,num_subregions-1);
  IndexSpaceT<1> color_is = runtime->create_index_space(ctx, color_bounds);

  // In this example we need to create two partitions: one disjoint
  // partition for describing the output values that are going to
  // be computed by each sub-task that we launch and a second
  // aliased partition which will describe the input values needed
  // for performing each task.  Note that for the second partition
  // each subregion will be a superset of its corresponding region
  // in the first partition, but will also require two 'ghost' cells
  // on each side.  The need for these ghost cells means that the
  // subregions in the second partition will be aliased.
  IndexPartition disjoint_ip = 
    runtime->create_equal_partition(ctx, is, color_is);
  const int block_size = (num_elements + num_subregions - 1) / num_subregions;
  Transform<1,1> transform;
  transform[0][0] = block_size;
  Rect<1> extent(-1, block_size);
  IndexPartition ghost_ip = 
    runtime->create_partition_by_restriction(ctx, is, color_is, transform, extent);

  // Once we've created our index partitions, we can get the
  // corresponding logical partitions for the stencil_lr
  // logical region.
  LogicalPartition disjoint_lp = 
    runtime->get_logical_partition(ctx, stencil_lr, disjoint_ip);
  LogicalPartition ghost_lp = 
    runtime->get_logical_partition(ctx, stencil_lr, ghost_ip);

  // Our launch domain will again be isomorphic to our coloring domain.
  ArgumentMap arg_map;

  // First initialize the 'FID_VAL' field with some data
  IndexLauncher init_launcher(INIT_FIELD_TASK_ID, color_is,
                              TaskArgument(NULL, 0), arg_map);
  init_launcher.add_region_requirement(
      RegionRequirement(disjoint_lp, 0/*projection ID*/,
                        WRITE_DISCARD, EXCLUSIVE, stencil_lr));
  init_launcher.add_field(0, FID_VAL);
  runtime->execute_index_space(ctx, init_launcher);

  // Now we're going to launch our stencil computation.  We
  // specify two region requirements for the stencil task.
  // Each region requirement is upper bounded by one of our
  // two partitions.  The first region requirement requests
  // read-only privileges on the ghost partition.  Note that
  // because we are only requesting read-only privileges, all
  // of our sub-tasks in the index space launch will be 
  // non-interfering.  The second region requirement asks for
  // read-write privileges on the disjoint partition for
  // the 'FID_DERIV' field.  Again this meets with the 
  // mandate that all points in our index space task
  // launch be non-interfering.
  FutureMap fm;
  double norm;

  IndexLauncher stencil_launcher(STENCIL_TASK_ID, color_is,
       TaskArgument(&num_elements, sizeof(num_elements)), arg_map);
  stencil_launcher.add_region_requirement(
      RegionRequirement(ghost_lp, 0/*projection ID*/,
                        READ_ONLY, EXCLUSIVE, stencil_lr));
  stencil_launcher.add_field(0, FID_VAL);
  stencil_launcher.add_region_requirement(
      RegionRequirement(disjoint_lp, 0/*projection ID*/,
                        READ_WRITE, EXCLUSIVE, stencil_lr));
  stencil_launcher.add_field(1, FID_DERIV);

  IndexLauncher stencil2_launcher(STENCIL_TASK_ID, color_is,
       TaskArgument(&num_elements, sizeof(num_elements)), arg_map);
  stencil2_launcher.add_region_requirement(
      RegionRequirement(ghost_lp, 0/*projection ID*/,
                        READ_ONLY, EXCLUSIVE, stencil_lr));
  stencil2_launcher.add_field(0, FID_DERIV);
  stencil2_launcher.add_region_requirement(
      RegionRequirement(disjoint_lp, 0/*projection ID*/,
                        READ_WRITE, EXCLUSIVE, stencil_lr));
  stencil2_launcher.add_field(1, FID_VAL);


  for(int it = 0; it < num_iterations; it++) {
    if(it % 2 == 0)
      fm = runtime->execute_index_space(ctx, stencil_launcher);
    else
      fm = runtime->execute_index_space(ctx, stencil2_launcher);
    fm.wait_all_results();
    norm = 0;
    for (int i = 0; i < num_subregions; i++) {
      double received = fm.get_result<double>(i);
	  norm += received;
    }
    printf("it %d norm %lf\n", it, norm/num_elements);
  }

  // Clean up our region, index space, and field space
  runtime->destroy_logical_region(ctx, stencil_lr);
  runtime->destroy_field_space(ctx, fs);
  runtime->destroy_index_space(ctx, is);
}

// The standard initialize field task from earlier examples
void init_field_task(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context ctx, Runtime *runtime)
{
  assert(regions.size() == 1); 
  assert(task->regions.size() == 1);
  assert(task->regions[0].privilege_fields.size() == 1);

  FieldID fid = *(task->regions[0].privilege_fields.begin());
  const int point = task->index_point.point_data[0];
  printf("Initializing field %d for block %d...\n", fid, point);

  const FieldAccessor<WRITE_DISCARD,double,1> acc(regions[0], fid);

  Rect<1> rect = runtime->get_index_space_domain(ctx,
                  task->regions[0].region.get_index_space());
  for (PointInRectIterator<1> pir(rect); pir(); pir++)
    acc[*pir] = drand48();
}

// Our stencil tasks is interesting because it
// has both slow and fast versions depending
// on whether or not its bounds have been clamped.
double stencil_task(const Task *task,
                  const std::vector<PhysicalRegion> &regions,
                  Context ctx, Runtime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  assert(task->regions[0].privilege_fields.size() == 1);
  assert(task->regions[1].privilege_fields.size() == 1);
  assert(task->arglen == sizeof(int));
  const int max_elements = *((const int*)task->args);
  const int point = task->index_point.point_data[0];
  
  FieldID read_fid = *(task->regions[0].privilege_fields.begin());
  FieldID write_fid = *(task->regions[1].privilege_fields.begin());

  const FieldAccessor<READ_ONLY,double,1> read_acc(regions[0], read_fid);
  const FieldAccessor<WRITE_DISCARD,double,1> write_acc(regions[1], write_fid);

  Rect<1> rect = runtime->get_index_space_domain(ctx,
                  task->regions[1].region.get_index_space());
  // If we are on the edges of the entire space we are 
  // operating over, then we're going to do the slow
  // path which checks for clamping when necessary.
  // If not, then we can do the fast path without
  // any checks.
  double diff = 0;
  if ((rect.lo[0] < 1) || (rect.hi[0] > (max_elements-2)))
  {
    printf("Running slow stencil path for point %d...\n", point);
    // Note in the slow path that there are checks which
    // perform clamps when necessary before reading values.
    for (PointInRectIterator<1> pir(rect); pir(); pir++)
    {
      double l, r;
      if (pir[0] < 1)
      {
      	write_acc[*pir] = read_acc[*pir];
		continue;
      }
      else
        l = read_acc[*pir - 1];
      if (pir[0] > (max_elements-2))
      {
      	write_acc[*pir] = read_acc[*pir];
		continue;
      }
      else
        r = read_acc[*pir + 1];
      
      double result = (l + r) / 2.0;
      write_acc[*pir] = result;
      double p = read_acc[*pir];
      diff += (p - result) * (p - result);
    }
  }
  else
  {
    printf("Running fast stencil path for point %d...\n", point);
    // In the fast path, we don't need any checks
    for (PointInRectIterator<1> pir(rect); pir(); pir++)
    {
      double l = read_acc[*pir - 1];
      double r = read_acc[*pir + 1];

      double result = (l + r) / 2.0;
      write_acc[*pir] = result;
      double p = read_acc[*pir];
      diff += (p - result) * (p - result);
    }
  }
  return diff;
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
    Runtime::preregister_task_variant<double, stencil_task>(registrar, "stencil");
  }

  return Runtime::start(argc, argv);
}
