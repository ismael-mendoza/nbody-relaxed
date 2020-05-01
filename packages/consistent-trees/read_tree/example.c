#include <stdio.h>
#include <inttypes.h>
#include "read_tree.h"

// doing something like all_halos.halos[i] will iterate through the roots of each tree.
// to iterate through every single halo, do all_halos.halo_lookup[i], not sure what order is.

int main(void) {
  printf("Will read trees now...\n");


  read_tree("/home/imendoza/alcca/nbody-relaxed/data/raw/trees_bolshoi/tree_0_0_0.dat");
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

  // fprintf(fp, "# (id, tree root id, mmp, mvir, scale)\n\n");
  int64_t count=0;
  int64_t count_root=0; // how many root ids are they?

  FILE *fp;
  char *file_name = "test3.txt";
  fp = fopen(file_name, "w");
  printf("Overwriting file %s \n", file_name);

  while(count < all_halos.num_halos){
      struct halo *curr_halo = all_halos.halos + count;

      if(count % 1000 ==0){
        printf("Count is %ld\n", (long)count);
      }

      if(curr_halo->id == curr_halo->tree_root_id){ // if this is a root halo id, we follow it.
            count_root++;
            // FILE *fp;
            // fp = fopen("test1.txt", "w");
            fprintf(fp, "# tree root id: %ld \n", (long)curr_halo->tree_root_id);

            while (curr_halo->prog != NULL){ //follow all the progenitors in this tree_root id.

                // fprintf(fp, "(%ld, %ld,  %f, %f, %d)\n",
                //      (long)curr_halo->id,
                //      (long)curr_halo->tree_root_id,
                //      curr_halo->mvir,
                //      curr_halo->scale,
                //      (int)curr_halo->mmp
                //      );

                fprintf(fp, "(%f, %f, %d)\n",
                     curr_halo->mvir,
                     curr_halo->scale,
                     (int)curr_halo->mmp
                     );

                // if(curr_halo->mmp == 0){  // check if it's in the main line progenitor.
                //     printf("Something is wrong !\n");
                //     fclose(fp);
                //     return 1;
                // }

                if (curr_halo->prog->mmp==1){
                    curr_halo = curr_halo->prog;
                }

                else{
                    fprintf(fp, "Found a mmp=0 halo progenitor!\n");
                    fprintf(fp, "(%ld, %ld, %f, %f, %d)\n",
                         (long)curr_halo->prog->id,
                         (long)curr_halo->prog->tree_root_id,
                         curr_halo->prog->mvir,
                         curr_halo->prog->scale,
                         (int)curr_halo->prog->mmp
                         );

                    if(curr_halo->prog->next_coprog != NULL){
                        fprintf(fp, "There is a coprogenitor!\n");
                        fprintf(fp, "It has mass: %f\n", curr_halo->prog->next_coprog->mvir);
                        fprintf(fp, "and mmp value: %d\n", (int)curr_halo->prog->next_coprog->mmp);
//                        fclose(fp);
//                        return 1;
                    }

                    break;
                }
            }
        fprintf(fp, "\n\n");
        fflush(fp);
      }
      count++;
  }
  fclose(fp);

  printf("Number of root nodes is: %ld\n", (long)count_root);
  printf("final count is: %ld\n", (long)count);

  return 0;
}