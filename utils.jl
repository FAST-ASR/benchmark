
"""
    theoretical_bandwidth(device[; data_rate = 2])

Return the theoretical memory bandwidth of `device` in GB/s.
The result is multiplied by `data_rate` to take into account the RAM
data rate (e.g. for DDR RAM `data_rate = 2`).
"""
function theoretical_bandwidth(device; data_rate = 2)
    mem_clockrate_kHz = CUDA.attribute(device, CUDA.CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE)
    buswidth_bits = CUDA.attribute(device, CUDA.CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH)
    (data_rate * mem_clockrate_kHz * 1000 * buswidth_bits / 8) / 1e9
end

function computational_throughput(device, numops, t)
    clockrate_kHz = CUDA.attribute(device(), CUDA.CU_DEVICE_ATTRIBUTE_CLOCK_RATE)
    numproc = CUDA.attribute(device, CUDA.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    clockrate * 1000 * numproc * ops_per_cycle
end

macro time_cukernel(ex, threads=1, blocks=1)
    Meta.isexpr(ex, :call) || throw(ArgumentError("expression should be a function call"))
    f = ex.args[1]
    args = ex.args[2:end]
    @gensym kargs kargs_type kernel
    quote
        $kargs = map(cudaconvert, ($(args...),) )
        $kargs_type = Tuple{map(typeof, $kargs)...}
        $kernel = CUDA.cufunction($f, $kargs_type)
        $kernel($(args...); threads=$threads, blocks=$blocks)

        CUDA.@elapsed $kernel($(args...); threads=$threads, blocks=$blocks)
    end
end
