#include <stdio.h>
#include <inttypes.h>
#include "read_tree.h"

// doing something like all_halos.halos[i] will iterate through the roots of each tree.
// to iterate through every single halo, do all_halos.halo_lookup[i], not sure what order is.

int main(void) {
  read_tree("/home/imendoza/alcca/nbody-relaxed/intro/data/trees_bolshoi/tree_0_0_0.dat");
  printf("%"PRId64" halos found in tree_0_0_0.dat!\n", all_halos.num_halos);
//  printf("%"PRId64" lists found in the halo tree!\n", halo_tree.num_lists);


//  int64_t my_id = 3060299107;
//  struct halo* my_halo = lookup_halo_in_list(&all_halos, my_id); // this will never work.

//  int64_t count=0;
//  while(count < all_halos.num_halos){
//    printf("root id, tree root id, progenitor id, mmp: (%f, %f, %f, %f, %f)\n",
//           (double)all_halos.halos[count].id,
//           (double)all_halos.halos[count].tree_root_id,
//           (double)all_halos.halos[count].prog->id,
//           (double)all_halos.halos[count].prog->mmp,
//           (double)count
//           );
//    count++;
//  }

  //find all main progenitor lines.
  //Total number of halos: 24875523
  FILE *fp;
  fp = fopen("test1.txt", "w");
  fprintf(fp, "# (id, tree root id, mmp, mvir, scale)\n\n");
  int64_t count=0;
  int64_t count_root=0;
  while(count < all_halos.num_halos){ // how many root ids are they?
      struct halo *curr_halo = all_halos.halos + count;
      if(count % 10000 ==0){
        printf("Count is %ld\n", (long)count);
      }

      if(curr_halo->id == curr_halo->tree_root_id){
            count_root++;

            fprintf(fp, "# tree root id: %ld \n", (long)curr_halo->tree_root_id);
            while (curr_halo->prog != NULL){

                printf("(%ld, %f, %f, %ld, %d)\n",
                    (long)curr_halo->id,
                    curr_halo->mvir,
                    curr_halo->scale,
                    (long)curr_halo->tree_root_id,
                    curr_halo->mmp
                    );

                if(curr_halo->mmp == 0){
                    printf("Something is wrong !\n");
                    return 1;
                }

                fprintf(fp, "(%ld, %f, %f)\n",
                (long)curr_halo->id,
                curr_halo->mvir,
                curr_halo->scale
                );

                //need to print out a whole tree to make sure that we understand what progenitor and coprogenitor are.
                //look at all these parameters: *desc, *parent, *uparent, *prog, *next_coprog;
                if (curr_halo->prog->mmp==1){
                    curr_halo = curr_halo->prog;
                }

            }
            fprintf(fp, "\n\n");
            fflush(fp);
      }
      count++;
  }
  printf("Number of root nodes is: %ld\n", (long)count_root);
  printf("final count is: %ld\n", (long)count);
  fclose(fp);

  return 0;
}

//      if(curr_root->mmp == 0 || curr_root->id != curr_root->tree_root_id){
//        printf("Something is wrong!\n");
//        return 1;
//      }

//total haloes:
  // Get the first halo from tree_0_0_0 file
//  int64_t id = 3060299107;
//  printf("ID is: %" PRId64 "\n", id);
//  printf("Number of lists is %" PRId64 "\n", halo_tree.num_lists);
//  printf("Second halo id is %" PRId64 "\n", halo_tree.halo_lists[1].halos->id);
//  printf("Second halo tree root id is %" PRId64 "\n", halo_tree.halo_lists[1].halos->tree_root_id);


//  struct halo *halo1 = lookup_halo_in_list(halo_tree, id);
//  printf("Hi!\n");
//  printf("First halo id is %" PRId64 "\n", halo1->id);
//  printf("First progenitor has id is %" PRId64 "\n", halo1->prog->id);

//  printf("First halo has id is %" PRId64 "\n", all_halos.halos->id);
//  printf("First halo has tree root id is %" PRId64 "\n", all_halos.halos->tree_root_id);
////
//  printf("First progenitor has id is %" PRId64 "\n", all_halos.halos->prog->id);
//  printf("First progenitor has tree root id is %" PRId64 "\n", all_halos.halos->prog->tree_root_id);
//  printf("First progenitor is main %" PRId64 "\n", all_halos.halos->prog->mmp);
//
//  printf("Second halo has id is %" PRId64 "\n", all_halos.halos[1].id);
//  printf("Second halo has tree root id is %" PRId64 "\n", all_halos.halos[1].tree_root_id);
//
//  printf("Second halo has progenitor has id is %" PRId64 "\n", all_halos.halos[1].prog->id);
//  printf("Second halo has progenitor has tree root id is %" PRId64 "\n", all_halos.halos[1].prog->tree_root_id);
//  printf("Second halo has progenitor is main %" PRId64 "\n", all_halos.halos[1].prog->mmp);

