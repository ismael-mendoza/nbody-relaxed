#include <stdio.h>
#include <stdlib.h>

#include <inttypes.h>
#include "read_tree.h"

//find all main progenitor lines.
//Total number of halos: 24875523

int main(int argc, char **argv)
{
    char *path = argv[1];
    char *outname = argv[2];

    char *pend;
    float Mcut = strtof(argv[3], &pend);

    printf("Will read trees now from %s\n", path);
    read_tree(path);
    printf("%" PRId64 " halos found in %s!\n", all_halos.num_halos, path);

    int64_t count = 0;
    int64_t count_root = 0; // how many root ids are they?

    FILE *fp;
    fp = fopen(outname, "w");
    printf("Overwriting file %s \n", outname);
    fprintf(fp, "Order is: (id, mvir, scale, scale_of_last_MM, coprog_id, "
                "coprog_mvir, coprog_scale)\n\n");

    while (count < all_halos.num_halos)
    {
        struct halo *curr_halo = all_halos.halos + count;

        if (curr_halo->id == curr_halo->tree_root_id && curr_halo->mvir > Mcut)
        { // if this is a root halo id, we follow it.
            count_root++;
            fprintf(fp, "# tree root id: %ld #\n", (long)curr_halo->tree_root_id);

            //follow all the main line progenitors in this tree_root id.
            while (1)
            {

                fprintf(fp, "%ld,%f,%f,%f,",
                        (long)curr_halo->id,
                        curr_halo->mvir,
                        curr_halo->scale,
                        curr_halo->scale_of_last_MM);

                //also parameters for second most massive halo if it exists.
                if (curr_halo->next_coprog != NULL)
                {
                    fprintf(fp, "%ld,%f,%f\n",
                            curr_halo->next_coprog->id,
                            curr_halo->next_coprog->mvir,
                            curr_halo->next_coprog->scale);
                }
                else
                {
                    fprintf(fp, ",,\n");
                }

                // consistent trees automatically matches you to most massive progenitor
                if (curr_halo->prog == NULL)
                {
                    break;
                }

                curr_halo = curr_halo->prog;
            }
            fprintf(fp, "\n\n");
            fflush(fp);
        }
        count++;
    }

    printf("Number of root nodes is: %ld\n", (long)count_root);
    printf("final count is: %ld\n\n", (long)count);

    fprintf(fp, "Number of root nodes is: %ld\n", (long)count_root);
    fprintf(fp, "final count is: %ld\n\n", (long)count);
    fflush(fp);
    fclose(fp);

    printf("Success on writing %s!\n", outname);

    return 0;
}
