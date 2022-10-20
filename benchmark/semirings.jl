
using CUDA
using LogExpFunctions
using Printf

include("../utils.jl")

const N = 1_000_000
const MIN = -100
const MAX = 100

a = collect(MIN:((MAX - MIN) / (N - 1)):MAX)
x = collect(MIN:((MAX - MIN) / (N - 1)):MAX)
b = collect(MIN:((MAX - MIN) / (N - 1)):MAX)
y = a .* x .+ b

_a = CuArray(a)
_x = CuArray(x)
_b = CuArray(b)
_y = similar(_a)


function kernel_natural_semiring(y, a, x, b)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(y)
        y[i] = a[i]
        #y[i] = a[i] * x[i] + b[i]
    end
    return
end

function kernel_log_semiring(y, a, x, b)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(y)
        v = a[i]
        y[i] = logaddexp(a[i] + x[i], b[i])
    end
    return
end

function kernel_trop_semiring(y, a, x, b)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(y)
        y[i] = max(a[i] + x[i], b[i])
    end
    return
end

max_threads_per_block = CUDA.attribute(device(), CUDA.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
threads_per_block = min(N, max_threads_per_block)
blocks = cld(N, threads_per_block)

time_natural_semiring = @time_cukernel(
    kernel_natural_semiring(_y, _a, _x, _b),
    threads_per_block,
    blocks
)

time_natural_semiring = @time_cukernel(
    kernel_natural_semiring(_y, _a, _x, _b),
    threads_per_block,
    blocks
)

time_log_semiring = @time_cukernel(
    kernel_log_semiring(_y, _a, _x, _b),
    threads_per_block,
    blocks
)

#
#CUDA.@elapsed @cuda threads=threads_per_block blocks=blocks kernel_natural_semiring(_y, _a, _x, _b)
#CUDA.synchronize()
#time_natural_semiring = CUDA.@elapsed @cuda threads=threads_per_block blocks=blocks kernel_natural_semiring(_y, _a, _x, _b)
#
#CUDA.@elapsed CUDA.@sync @cuda threads=threads_per_block blocks=blocks kernel_log_semiring(_y, _a, _x, _b)
#CUDA.synchronize()
#time_log_semiring = CUDA.@elapsed CUDA.@sync @cuda threads=threads_per_block blocks=blocks kernel_log_semiring(_y, _a, _x, _b)
#
#CUDA.@elapsed CUDA.@sync @cuda threads=threads_per_block blocks=blocks kernel_trop_semiring(_y, _a, _x, _b)
#CUDA.synchronize()
#time_trop_semiring = CUDA.@elapsed CUDA.@sync @cuda threads=threads_per_block blocks=blocks kernel_trop_semiring(_y, _a, _x, _b)

println("Timing:")
@printf "  * natural semiring: %.3f μs\n" (time_natural_semiring * 1e6)
@printf "  * log semiring: %.3f μs\n" (time_log_semiring * 1e6)
#@printf "  * tropical semiring: %.3f μs\n" (time_trop_semiring * 1e6)

