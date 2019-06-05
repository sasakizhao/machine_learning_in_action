import sys
from numpy import mat, mean, power


def read_input(file):
    for line in file:
        yield line.rstrip()


input = read_input(sys.stdin)
mapper_out = [line.split('\t') for line in input]
print('mapper_out', mapper_out)
cum_val = 0.0
cum_sum_sq = 0.0
cum_N =0.0
for instance in mapper_out:
    nj = float(instance[0])
    cum_N += nj
    cum_val += nj * float(instance[1])
    cum_sum_sq += nj * float(instance[2])
mean = cum_val / cum_N
var_sum = (cum_sum_sq - 2 * mean * cum_val + cum_N * mean * mean) / cum_N
print('%d\t%f\t%f' % (cum_N, mean, var_sum))
# print(sys.stderr, 'report still alive')
