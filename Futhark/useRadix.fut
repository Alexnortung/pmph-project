import "lib/github.com/diku-dk/sorts/radix_sort"

-- ==
-- compiled random input { [10000]f32 }
-- output @ test10_000.out.gz
 



let main(xs: []f32) : []f32 = 
    --radix_sort_int i32.num_bits i32.get_bit xs
    radix_sort_float f32.num_bits f32.get_bit xs