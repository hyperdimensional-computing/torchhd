import hashmap_encoding
import hashmap_encoding_online
import hashmap_encoding_online_iterative
import hashmap_encoding_regenerative_continuous
import hashmap_encoding_regenerative_reset
import random_projection
import random_projection_online
import random_projection_online_iterative
import random_projection_regenerative_continuous
import random_projection_regenerative_reset
import sinusoid_projection
import sinusoid_projection_online
import sinusoid_projection_online_iterative
import sinusoid_projection_regenerative_contiuous
import sinusoid_projection_regenerative_reset

dimensions = [10000]
epochs = [5]
drop_rate = [0.2]
levels = [100]

files = [hashmap_encoding, hashmap_encoding_online, hashmap_encoding_online_iterative,
         hashmap_encoding_regenerative_continuous, hashmap_encoding_regenerative_reset,
         random_projection, random_projection_online, random_projection_online_iterative,
         random_projection_regenerative_continuous, random_projection_regenerative_reset,
         sinusoid_projection, sinusoid_projection_online, sinusoid_projection_online_iterative,
         sinusoid_projection_regenerative_contiuous, sinusoid_projection_regenerative_reset]

for i in files:
    i.experiment()






