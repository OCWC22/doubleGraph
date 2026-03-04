# SKILLS.md — DoubleGraph: Transferable Techniques for GPU-Specific Graph Algorithm Generation

> **Audience:** Research engineers adapting DoubleGraph's approach for H100, B200, B300, and future GPU architectures.
> **Scope:** How WarpSpeed (doubleAI's AI system) generated architecture-optimized graph kernels as a drop-in replacement for NVIDIA cuGraph.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Architecture Overview](#2-architecture-overview)
3. [First Principles Walkthrough](#3-first-principles-walkthrough)
4. [A100-Specific Kernel Techniques](#4-a100-specific-kernel-techniques)
5. [Cross-Architecture Comparison](#5-cross-architecture-comparison)
6. [Challenges & Limitations](#6-challenges--limitations)
7. [Technical Constraints for Future Architectures](#7-technical-constraints-for-future-architectures)
8. [Transferable Techniques](#8-transferable-techniques)
9. [Engineering Requirements Checklist](#9-engineering-requirements-checklist)

---

## 1. Problem Statement

### The Gap

NVIDIA's cuGraph provides GPU-accelerated graph algorithms, but its kernels are **architecture-generic** — compiled to run on any CUDA GPU. This leaves significant performance on the table because:

- Different GPU architectures have different L2 cache sizes (A100: 40MB, L4: 48MB, A10G: 6MB)
- Warp scheduling, register file size, shared memory capacity, and memory bandwidth all vary per SM generation
- Optimal kernel launch configurations (block size, grid size, register pressure) are architecture-dependent
- Algorithm-level strategy (e.g., top-down vs bottom-up BFS switching thresholds) should adapt to the hardware

### The Solution

DoubleGraph replaces cuGraph's generic kernels with **per-GPU-architecture optimized implementations**. WarpSpeed generates 192 CUDA kernel files per target GPU, each hand-tuned for the specific hardware. The result is a **drop-in replacement** — same Python/C++ API, same results, 10-100x faster on 18%+ of algorithms.

### Scale of the Effort

| Metric | Count |
|--------|-------|
| GPU targets | 3 (A100/SM80, L4/SM89, A10G/SM86) |
| Algorithm categories | 8 (traversal, link_analysis, centrality, community, components, cores, link_prediction, tree) |
| Kernel files per GPU | 192 `.cu` files |
| Per-kernel flag files | 172 `.cu.flags` sidecars |
| Total kernel files | 576 `.cu` + 516 `.cu.flags` |

---

## 2. Architecture Overview

DoubleGraph has 4 architectural layers. Data flows top-to-bottom:

```
cuGraph Python API
    ↓
pylibcugraph (Cython bindings)
    ↓
libcugraph C++ (template functions)
    ↓
┌─────────────────────────────────────────────────┐
│  LAYER 4: Integration Layer                      │
│  cpp/src/aai/integration/                        │
│  Template specialization + #ifdef AAI_ROUTE_*    │
│  Routes cuGraph calls → AAI implementations      │
├─────────────────────────────────────────────────┤
│  LAYER 3: AAI Implementation (per-GPU kernels)   │
│  cpp/src/aai/impl/{a100,l4,a10g}/               │
│  192 .cu files per GPU target                    │
│  The actual optimized CUDA kernels               │
├─────────────────────────────────────────────────┤
│  LAYER 2: AAI Headers (API + types)              │
│  cpp/include/cugraph/aai/                        │
│  compact_graph_t, CachePool, result types        │
│  API declarations with 4-way dispatch            │
├─────────────────────────────────────────────────┤
│  LAYER 1: Build System                           │
│  cpp/CMakeLists.txt + .cu.flags sidecars         │
│  Compile-time GPU selection, no runtime dispatch │
└─────────────────────────────────────────────────┘
```

### Key Design Principle: Compile-Time, Not Runtime

GPU target selection happens at **build time** via `-DTARGET_GPU=A100`. There is no runtime GPU detection or kernel dispatch. Each wheel/binary is built for exactly one GPU architecture. This eliminates:
- Runtime branching overhead
- Binary bloat from shipping all GPU variants
- Complexity of runtime capability detection

---

## 3. First Principles Walkthrough

### Step 1: Define a Universal Graph Abstraction

**File:** `cpp/include/cugraph/aai/compact_graph.hpp` (151 lines)

cuGraph's `graph_view_t` is a complex template type with many features AAI doesn't need. The first step was creating `compact_graph_t` — a stripped-down CSR/CSC representation:

```cpp
template <typename vertex_t, typename edge_t>
struct compact_graph_t {
    // Core CSR/CSC data (device memory)
    const edge_t* offsets;           // Size: num_vertices + 1
    const vertex_t* indices;         // Size: num_edges
    vertex_t number_of_vertices;
    edge_t number_of_edges;

    // Graph metadata
    bool is_symmetric;               // Undirected flag
    bool is_multigraph;
    bool is_csc;                     // false=CSR, true=CSC

    // Optimization data
    std::optional<std::vector<vertex_t>> segment_offsets;  // 5 elements → 4 degree segments
    const uint32_t* edge_mask;       // Packed bitmask for edge filtering
};

using graph32_t = compact_graph_t<int32_t, int32_t>;
```

**Why this matters:**
- `from_graph_view()` factory bridges cuGraph's world to AAI's — all AAI kernels take `graph32_t`
- Segment offsets partition vertices by degree: high (≥1024), mid (32-1023), low (1-31), zero
- Edge mask enables subgraph operations without copying: bit `j` of word `i` = status of edge `j`
- Every kernel knows exactly what data it gets — no template metaprogramming in hot paths

### Step 2: Manage GPU Resources Efficiently

**File:** `cpp/include/cugraph/aai/cache_pool.hpp` (125 lines)

Repeated graph algorithm calls (common in iterative workflows) trigger repeated `cudaMalloc`/`cudaFree` cycles. CachePool eliminates this:

```cpp
class CachePool {
    static constexpr size_t DEFAULT_CAPACITY = 8;
    std::list<const void*> order_;                    // LRU tracking
    std::unordered_map<const void*, Entry> map_;      // tag → (Cacheable, iterator)

    template <typename T>
    T& acquire(const void* tag);  // Get-or-create with LRU eviction
};
```

**Usage pattern in kernels:**
```cpp
struct BfsCache : Cacheable {
    int32_t* frontier_buf = nullptr;
    BfsCache()  { cudaMalloc(&frontier_buf, ...); }
    ~BfsCache() { cudaFree(frontier_buf); }
};

void bfs(const graph32_t& graph, ...) {
    static int tag;  // Stable address per algorithm
    auto& cache = cache_pool().acquire<BfsCache>(&tag);
    // Reuse cache.frontier_buf across calls
}
```

**Key decision:** Thread-local singleton via `thread_local static`. Each thread gets its own LRU cache of 8 GPU resource sets. Eviction automatically `cudaFree`s the oldest resources.

### Step 3: Define the 4-Way Dispatch Pattern

**Files:** `cpp/include/cugraph/aai/api/*.hpp`

Every algorithm has up to 4 variants based on two binary properties of the input graph:

| Variant | Has `segment_offsets`? | Has `edge_mask`? | Suffix |
|---------|----------------------|------------------|--------|
| Base | No | No | (none) |
| Segmented | Yes | No | `_seg` |
| Masked | No | Yes | `_mask` |
| Seg+Mask | Yes | Yes | `_seg_mask` |

Each variant is a **separate function** (not overloaded), each with a **separate .cu implementation file**:

```cpp
// cpp/include/cugraph/aai/api/traversal.hpp
void bfs(const graph32_t& graph, int32_t* distances, ...);
void bfs_seg(const graph32_t& graph, int32_t* distances, ...);
void bfs_mask(const graph32_t& graph, int32_t* distances, ...);
void bfs_seg_mask(const graph32_t& graph, int32_t* distances, ...);
```

**Why separate functions, not runtime branching:**
- Each variant can use completely different kernel code optimized for its scenario
- Segmented graphs enable degree-aware work scheduling (assign warps to high-degree vertices)
- Masked graphs may skip edges cheaply via bitmask checks instead of structural modifications
- The compiler can optimize each variant independently

**Additional dispatch dimensions for some algorithms:**
- **Precision:** `float` vs `double` (e.g., `pagerank_f32.cu`, `pagerank_f64.cu`)
- **Weights:** Weighted vs unweighted variants
- **Personalization:** Standard vs personalized (PageRank)

### Step 4: Write GPU-Specific Kernel Implementations

**Directory:** `cpp/src/aai/impl/{a100,l4,a10g}/`

This is where the core optimization work lives. Each GPU target gets its own complete set of 192 `.cu` files. The implementations differ significantly between GPUs — they are not parameterized copies but fundamentally different algorithms tuned for each architecture.

**Example — BFS on A100** (`cpp/src/aai/impl/a100/traversal/bfs.cu`):

A100's BFS uses **direction-optimizing** search with **queue↔bitmap frontier conversion**:

1. **Top-down phase** (sparse frontier): Queue-based, 1 warp per frontier vertex, `ballot_sync` for collective insertion
2. **Bottom-up phase** (dense frontier): Bitmap-based, 1 thread per vertex, early termination on first parent found
3. **Switching logic**: TD→BU when `frontier_size > num_vertices/20`, BU→TD when `frontier_size < num_vertices/200`
4. **Conversion kernels**: `queue_to_bitmap_kernel` and `bitmap_to_queue_kernel` bridge the two representations

**The same algorithm on L4** (`cpp/src/aai/impl/l4/traversal/bfs.cu`) uses:
- Bitmap-only frontier (no queue representation)
- CUB library for prefix sums and reductions
- Different switching thresholds tuned for L4's memory subsystem

**The same algorithm on A10G** (`cpp/src/aai/impl/a10g/traversal/bfs.cu`) uses:
- 2-level batched top-down (processes 2 BFS levels per kernel launch)
- L2 cache pinning via `cudaCtxSetLimit` for frontier data
- Frontier rebuild capability for better cache utilization

### Step 5: Wire It Into cuGraph via Template Specialization

**Directory:** `cpp/src/aai/integration/`

The integration layer makes AAI a **drop-in replacement** by overriding cuGraph's own implementations at link time:

```cpp
// cpp/src/aai/integration/traversal/bfs.cu

#ifdef AAI_ROUTE_BFS  // Set by CMake for this file

// Override cuGraph's bfs() template instantiation
template <>
void bfs(raft::handle_t const& handle,
         graph_view_t<int32_t, int32_t, false, false> const& graph_view,
         int32_t* distances, int32_t* predecessors,
         int32_t const* sources, size_t n_sources,
         bool direction_optimizing, int32_t depth_limit,
         bool do_expensive_check) {

    handle.sync_stream();  // AAI uses default CUDA stream

    // Extract compact graph from cuGraph's view
    auto compact = graph32_t::from_graph_view<false>(graph_view);

    // 4-way dispatch based on graph properties
    if (direction_optimizing) {
        if (compact.edge_mask) {
            if (compact.segment_offsets) aai::bfs_direction_optimizing_seg_mask(compact, ...);
            else                        aai::bfs_direction_optimizing_mask(compact, ...);
        } else {
            if (compact.segment_offsets) aai::bfs_direction_optimizing_seg(compact, ...);
            else                        aai::bfs_direction_optimizing(compact, ...);
        }
    } else {
        // Same 4-way dispatch for standard BFS
    }

    cudaDeviceSynchronize();  // Sync back before returning to cuGraph
}

#else
// Fallback: use original cuGraph implementation
template void bfs<int32_t, int32_t, false>(...);
#endif
```

**Stream synchronization protocol:**
1. `handle.sync_stream()` — Ensure cuGraph's stream work completes
2. AAI kernel executes on default CUDA stream
3. `cudaDeviceSynchronize()` — Ensure AAI work completes before cuGraph resumes

**Constraint:** This only works for single-GPU, `int32_t` vertex/edge type graphs. The `#else` branch preserves original cuGraph for all other configurations.

### Step 6: Build System Ties It Together

**File:** `cpp/CMakeLists.txt`

The build system enforces compile-time GPU selection and manages the complexity:

```cmake
# 1. User selects GPU target
# cmake -DTARGET_GPU=A100 ..

# 2. Maps to implementation directory
set(_AAI_IMPL_DIR "src/aai/impl/${AAI_TARGET_GPU_SUFFIX}")  # e.g., src/aai/impl/a100

# 3. Collects all .cu files for that GPU
file(GLOB_RECURSE _all_cu_files "${_AAI_IMPL_DIR}/**/*.cu")

# 4. Reads per-file .cu.flags sidecars
foreach(_cu_file ${_all_cu_files})
    set(_flags_file "${_cu_file}.flags")
    if(EXISTS "${_flags_file}")
        file(READ "${_flags_file}" _flags)
        # Apply per-kernel compilation options
        set_source_files_properties("${_cu_file}"
            PROPERTIES COMPILE_OPTIONS "${_flags}")
    endif()
endforeach()

# 5. Separate RDC files into their own static library
# Files with --rdc=true in .cu.flags need CUDA_SEPARABLE_COMPILATION=ON
# (Required for cooperative kernels using grid-wide synchronization)

# 6. Set routing definitions on integration files
# -DAAI_ROUTE_BFS on integration/traversal/bfs.cu
# -DAAI_ROUTE_PAGERANK on integration/link_analysis/pagerank.cu
```

**Per-kernel .cu.flags examples:**
| Flag | Purpose | Used By |
|------|---------|---------|
| `--use_fast_math` | Relaxed IEEE754 for speed | Most kernels (172 files) |
| `--rdc=true` | Cooperative kernels (grid sync) | BFS mask, core_number, eigenvector |
| `--maxrregcount=48` | Limit register pressure | Leiden, clustering analysis |
| `--extra-device-vectorization` | Vectorized memory ops | WCC, cosine similarity |

**GPU target mapping** (`build_in_docker/target_gpu_map.json`):
```json
{
    "a100": { "arch": "80" },
    "l4":   { "arch": "89" },
    "a10g": { "arch": "86" }
}
```

---

## 4. A100-Specific Kernel Techniques

These are the concrete optimization strategies WarpSpeed used for A100 (SM80, Ampere):

### 4.1 Direction-Optimizing BFS with Cost Modeling

```
Alpha-Beta switching model:
  alpha = avg_degree × 0.25
  beta  = 24

  TD → BU:  frontier_degree_sum × alpha > unvisited_edges
  BU → TD:  frontier_size × beta < unvisited_vertices

  Simple fallback (symmetric, N ≥ 10000):
    TD → BU:  frontier_size > N/20
    BU → TD:  frontier_size < N/200
```

**Why A100-specific:** A100's 40MB L2 cache can hold large bitmaps, making bottom-up cheaper than on GPUs with smaller caches. The thresholds reflect this.

### 4.2 Queue↔Bitmap Frontier Conversion

A100 BFS maintains **both** representations and converts between them:
- **Queue** (array of vertex IDs): Efficient for sparse frontiers in top-down phase
- **Bitmap** (1 bit per vertex): Efficient for dense frontiers in bottom-up phase
- Conversion kernels use warp-level `__ballot_sync` to parallelize bit scanning

### 4.3 Warp-Level Primitives (No CUB Dependency)

A100 kernels use hand-rolled warp-cooperative operations:
```cuda
// Warp-level sum reduction
float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}
```
L4 uses CUB library instead — A100's hand-rolled versions allow tighter register control.

### 4.4 Shared-Memory Hash Tables (Louvain)

3-tier kernel dispatch based on graph coarsening level:
| Condition | Kernel | Strategy |
|-----------|--------|----------|
| N ≤ 200 | `local_move_serial` | Single-thread, CPU-like |
| avg_degree < 8 | `local_move_thread` | Thread-per-vertex, register hash table (64 entries) |
| avg_degree ≥ 8 | `local_move_warp` | Warp-per-vertex, shared-memory hash table (128 entries/warp) |

Hash table uses linear probing with `atomicCAS` for concurrent warp-level insertion:
```cuda
uint32_t h = (community_id * 2654435761u) % HT_SIZE_PER_WARP;
```

### 4.5 Cooperative Kernels (BFS Mask)

For low-degree masked graphs, A100 uses **persistent thread blocks** with grid-wide synchronization:
```cuda
cg::grid_group grid = cg::this_grid();
// All blocks stay resident, iterate BFS levels internally
// grid.sync() between levels — no kernel launch overhead
```
Requires `--rdc=true` compilation flag and `CUDA_SEPARABLE_COMPILATION=ON`.

### 4.6 SpMV-Based PageRank

A100 PageRank combines cuSPARSE for the matrix-vector product with custom kernels for everything else:
- `cusparseSpMV()` — CSC × vector product for rank propagation
- `dangling_sum_kernel` — Atomic reduction for dangling node contribution
- `update_and_diff_kernel` — Fused update + L1-norm convergence check (single pass)

### 4.7 Performance-First Error Handling

- **No CUDA error checking** in hot paths — relies on `CUDA_LAUNCH_BLOCKING=1` for debugging
- Callers must sync their CUDA stream before calling AAI functions
- Result memory allocated by AAI; **caller responsible for cudaFree**

---

## 5. Cross-Architecture Comparison

### 5.1 Kernel Strategy Differences

| Aspect | A100 (SM80) | L4 (SM89) | A10G (SM86) |
|--------|-------------|-----------|-------------|
| **BFS frontier** | Queue ↔ Bitmap conversion | Bitmap only | Queue + 2-level batching |
| **BFS switching** | Alpha-beta cost model | Simple threshold | Trend-based (prev frontier size) |
| **BFS primitives** | Hand-rolled warp ops | CUB library | Hand-rolled + L2 pinning |
| **PageRank** | cuSPARSE SpMV + custom kernels | Thread/warp kernel selection (degree threshold=6) | Separate launch functions + L2 pinning |
| **L2 cache strategy** | None (relies on 40MB L2) | None | Explicit `set_l2_persist()` calls |
| **Register control** | Auto (large register file) | Generic | Explicit `__launch_bounds__` |
| **Cooperative kernels** | Yes (BFS mask, core_number) | No | No |
| **Memory optimization** | Register tuning | Conservative allocation | Pinned host + L2 control |

### 5.2 Hardware-Driven Design Decisions

| GPU | L2 Cache | Strategy Implication |
|-----|----------|---------------------|
| A100 | 40 MB | Large bitmaps fit in L2; no explicit management needed |
| L4 | 48 MB | Could hold even more but kernels use simpler structures |
| A10G | 6 MB | Must explicitly pin critical data; 2-level batching reduces working set |

| GPU | SM Count | Strategy Implication |
|-----|----------|---------------------|
| A100 | 108 | High grid sizes, cooperative kernel viable |
| L4 | 58 | Moderate parallelism, CUB handles occupancy |
| A10G | 80 | Mid-range, benefits from batched processing |

### 5.3 Algorithm Coverage

All three GPUs have **identical algorithm coverage**: 192 `.cu` files each across the same 8 categories. The algorithms are:

- **Traversal:** BFS (4 variants), BFS Direction-Optimizing (4), SSSP float/double (8), K-Hop (4)
- **Link Analysis:** PageRank (10+ variants), HITS (8)
- **Centrality:** Betweenness (8), Eigenvector (20), Katz (20)
- **Community:** Louvain (4), Leiden (4), ECG (4), K-Truss (4), Triangle Count (4), Egonet (6), Spectral (4), Clustering Analysis (12)
- **Components:** WCC (4), SCC (2)
- **Cores:** Core Number (4), K-Core (4)
- **Link Prediction:** Jaccard (6), Cosine (6), Overlap (6), Sorensen (6)
- **Tree:** MST (4)

---

## 6. Challenges & Limitations

### 6.1 Fundamental Constraints

| Constraint | Impact | Root Cause |
|-----------|--------|------------|
| **Single-GPU only** | Cannot process graphs larger than device memory | `multi_gpu = false` enforced; exactly 1 edge partition required |
| **int32 vertices/edges only** | Max ~2.1B vertices/edges | `graph32_t` typedef; no int64 API surface |
| **No DCS (hypersparse) support** | Cannot handle power-law graphs with extreme degree skew | Requires exactly 5 segment offsets; DCS format not supported |
| **Default CUDA stream** | AAI kernels cannot overlap with other GPU work | All kernels use stream 0; callers must sync before/after |
| **No runtime GPU detection** | Must build separate binaries per GPU target | Compile-time `TARGET_GPU` selection |

### 6.2 Engineering Challenges Encountered

1. **Per-kernel tuning explosion**: 192 files × 3 GPUs × per-file flags = massive configuration space. Each kernel's optimal block size, register count, and algorithm strategy differs per GPU.

2. **Stream synchronization**: The integration layer must carefully sync cuGraph's stream with AAI's default stream. Missing syncs cause silent data corruption.

3. **Cooperative kernel limitations**: Only some GPUs and kernel configurations support `grid.sync()`. The build system must identify RDC files separately and compile them into a static library with special flags.

4. **4-way dispatch combinatorial growth**: Each new algorithm requires 4 variants (base/seg/mask/seg_mask) × precision variants × weight variants. PageRank alone has 10+ integration dispatch paths.

5. **cuGraph API surface tracking**: As NVIDIA updates cuGraph, template signatures change. Each integration file must match cuGraph's exact template parameters.

### 6.3 Operational Limitations

- **No incremental updates**: Changing one kernel requires rebuilding the entire AAI library for that GPU target
- **No A/B testing infrastructure**: Cannot easily compare AAI vs original cuGraph at runtime (selected at compile time)
- **No automated regression detection**: Performance regressions in generated kernels require manual benchmarking

---

## 7. Technical Constraints for Future Architectures

### 7.1 H100 (SM90, Hopper)

| Feature | Constraint | Opportunity |
|---------|-----------|-------------|
| **TMA (Tensor Memory Accelerator)** | New async copy engine; existing `cudaMemcpy` patterns won't leverage it | Bulk async data movement for large frontier copies |
| **Warp specialization** | SM90 supports persistent warp roles | Producer/consumer warp patterns for BFS (one warp expands frontier, another processes) |
| **Thread Block Clusters** | New hierarchy level; kernels need `__cluster_dims__` | Cluster-level shared memory for cross-block hash tables (Louvain) |
| **DPX instructions** | Hardware-accelerated dynamic programming | Could accelerate SSSP delta-stepping |
| **Distributed shared memory** | SMs in a cluster can access each other's SMEM | Inter-SM communication without global memory for frontier exchange |
| **50 MB L2 cache** | Larger than A100's 40MB | Bitmap thresholds may shift; more aggressive bitmap-based strategies viable |
| **FP8 support** | New precision level | Potential for low-precision PageRank variants |

**Build system change:** Add `"h100": { "arch": "90" }` to `target_gpu_map.json`, create `cpp/src/aai/impl/h100/` directory tree, add `AAI_GPU_H100 = 4` to `target_gpu.hpp`.

### 7.2 B200/B300 (SM100+, Blackwell/Next-gen)

| Feature | Constraint | Opportunity |
|---------|-----------|-------------|
| **New SM architecture** | Warp size, register file, SMEM capacity may change | All hardcoded constants (block sizes, hash table sizes) need re-evaluation |
| **NVLink 5.0** | Higher GPU-GPU bandwidth | Could enable multi-GPU AAI (currently single-GPU only) |
| **HBM3e** | Higher memory bandwidth | Memory-bound kernels (BFS bottom-up, PageRank SpMV) benefit most |
| **Unknown ISA extensions** | New instructions not yet public | Must wait for CUDA toolkit updates |
| **Compute capability ≥10.0** | New PTX features | May require `--gpu-architecture=sm_100` minimum |

### 7.3 Cross-Generation Requirements

For **any** new GPU target, the following must be addressed:

1. **Profile before optimizing**: Run existing A100 kernels on new hardware first to identify actual bottlenecks
2. **Re-tune switching thresholds**: BFS direction-switching, Louvain tier thresholds, PageRank kernel selection — all hardware-dependent
3. **Evaluate library tradeoffs**: A100 uses hand-rolled warp ops while L4 uses CUB. Each new GPU needs fresh evaluation of library vs custom code
4. **L2 cache strategy**: A10G explicitly pins data; A100 doesn't. New GPUs need profiling to determine optimal approach
5. **Register pressure**: Each SM generation has different register file size. `--maxrregcount` values and `__launch_bounds__` must be re-calibrated

---

## 8. Transferable Techniques

### 8.1 Directly Reusable (Architecture-Independent)

These patterns work on any GPU — copy them as-is:

| Technique | Location | What It Does |
|-----------|----------|--------------|
| **compact_graph_t** | `compact_graph.hpp` | Universal CSR/CSC input type; bridges cuGraph ↔ AAI |
| **CachePool** | `cache_pool.hpp` | Thread-local LRU GPU resource caching; eliminates malloc/free overhead |
| **4-way dispatch pattern** | `api/*.hpp` | base/seg/mask/seg_mask variant structure for every algorithm |
| **Integration layer** | `integration/*.cu` | Template specialization + `#ifdef` for drop-in replacement |
| **Build system structure** | `CMakeLists.txt` | TARGET_GPU selection, impl glob, .cu.flags, RDC separation |
| **Result types** | `types.hpp` | Clean return types with device pointers + metadata |
| **.cu.flags sidecars** | `impl/**/*.cu.flags` | Per-kernel compilation flag control |

### 8.2 Transferable with Re-Tuning (Architecture-Sensitive)

These strategies work on all GPUs but constants/thresholds must be re-calibrated:

| Technique | What to Re-Tune | How to Re-Tune |
|-----------|-----------------|----------------|
| **Direction-optimizing BFS** | TD↔BU switching thresholds (alpha, beta, N/20, N/200) | Profile with representative graphs; measure frontier expansion cost at different densities |
| **3-tier Louvain dispatch** | N threshold (200), degree threshold (8), hash table size (128) | Benchmark serial vs thread vs warp kernels at different graph sizes on target GPU |
| **PageRank kernel selection** | Warp vs thread threshold (avg_degree ≥ 6 for L4) | Profile SpMV at different average degrees; find crossover point |
| **Cooperative kernel viability** | Whether persistent blocks outperform iterative launches | Measure kernel launch overhead vs grid.sync() cost on target GPU |
| **Block/grid sizes** | BLOCK_SIZE (typically 256), grid calculations | Use `cudaOccupancyMaxActiveBlocksPerMultiprocessor` for target GPU |
| **Register pressure** | `--maxrregcount` values (24-64 range observed) | Profile occupancy vs register spill tradeoff per kernel |

### 8.3 Architecture-Specific (Must Re-Implement)

These are A100-specific and need new implementations for each GPU:

| Technique | A100 Approach | Why It's A100-Specific |
|-----------|--------------|----------------------|
| **Frontier representation** | Queue↔Bitmap dual | Optimal for 40MB L2; other GPUs use different strategies |
| **Warp-level reduction** | Hand-rolled `__shfl_down_sync` | CUB may be better on some architectures |
| **L2 cache management** | Implicit (large L2) | A10G needs explicit pinning; H100 may need TMA |
| **Shared memory hash tables** | 128 entries/warp | Depends on SMEM size per SM |
| **Fast math safety** | Applied globally | Accuracy requirements may differ for FP8/FP16 on newer GPUs |

### 8.4 Key Insight: The Meta-Pattern

The most transferable skill is the **process**, not any individual kernel:

1. **Characterize the hardware** — L2 size, SMEM capacity, register file, warp scheduler behavior
2. **Profile generic kernels** — Run cuGraph's original on the target to find bottlenecks
3. **Choose algorithm strategy** — Pick representation (bitmap vs queue vs hybrid) based on memory hierarchy
4. **Tune kernel parameters** — Block size, register count, shared memory allocation
5. **Validate with .cu.flags** — Use sidecars for per-kernel flag control without CMake changes
6. **Integration is automatic** — Same template specialization pattern works for any GPU

---

## 9. Engineering Requirements Checklist

### Adding a New GPU Target (e.g., H100)

#### Phase 1: Infrastructure (1-2 days)

- [ ] Add GPU to `target_gpu.hpp`:
  ```cpp
  #define AAI_GPU_H100 4
  #define AAI_IS_H100 (AAI_TARGET_GPU == AAI_GPU_H100)
  ```
- [ ] Add to `build_in_docker/target_gpu_map.json`:
  ```json
  "h100": { "arch": "90" }
  ```
- [ ] Create directory tree: `cpp/src/aai/impl/h100/{traversal,link_analysis,centrality,community,components,cores,link_prediction,tree}/`
- [ ] Update `CMakeLists.txt` `TARGET_GPU` validation to include H100
- [ ] Verify build system picks up new impl directory

#### Phase 2: Baseline Profiling (3-5 days)

- [ ] Compile A100 kernels for H100 (`-arch=sm_90`) — many will work unmodified
- [ ] Benchmark all 27 algorithm categories against cuGraph on H100
- [ ] Identify algorithms with largest performance gaps (prioritize these)
- [ ] Profile memory bandwidth utilization, occupancy, register spills
- [ ] Document H100-specific hardware features to exploit (TMA, DPX, clusters)

#### Phase 3: Kernel Generation (varies by scope)

- [ ] Start with **traversal** (BFS, SSSP) — highest impact, well-understood optimization space
- [ ] Implement base variant first, then seg, mask, seg_mask
- [ ] Create `.cu.flags` sidecars for each kernel with H100-optimal flags
- [ ] Test each kernel against cuGraph for correctness (bit-exact where applicable)
- [ ] Benchmark each kernel for performance regression/improvement

#### Phase 4: Integration & Packaging

- [ ] Verify integration layer dispatches correctly for H100 build
- [ ] Build wheel with `--wheel-target-gpu H100`
- [ ] Run full test suite via `pytest` against cuGraph's test harness
- [ ] Create performance comparison report vs A100 and vs cuGraph-on-H100

#### Phase 5: Validation

- [ ] Correctness: All algorithms produce identical results to cuGraph (within floating-point tolerance)
- [ ] Performance: Document speedup/slowdown for each algorithm
- [ ] Edge cases: Test with empty graphs, single-vertex graphs, disconnected graphs, maximum-size int32 graphs
- [ ] Memory: Verify no leaks via `compute-sanitizer --tool memcheck`
- [ ] Concurrency: Test with multiple CPU threads calling AAI simultaneously (CachePool is thread-local)

### File Count Estimate per GPU Target

| Category | Files | Variants |
|----------|-------|----------|
| Traversal | ~20 | BFS(8) + SSSP(8) + K-Hop(4) |
| Link Analysis | ~18 | PageRank(10+) + HITS(8) |
| Centrality | ~48 | Betweenness(8) + Eigenvector(20) + Katz(20) |
| Community | ~42 | Louvain(4) + Leiden(4) + ECG(4) + K-Truss(4) + Triangle(4) + Ego(6) + Spectral(4) + Clustering(12) |
| Components | ~6 | WCC(4) + SCC(2) |
| Cores | ~8 | Core Number(4) + K-Core(4) |
| Link Prediction | ~24 | Jaccard(6) + Cosine(6) + Overlap(6) + Sorensen(6) |
| Tree | ~4 | MST float/double × base/seg |
| **Total** | **~192** | **+ ~172 .cu.flags** |

---

## Appendix: Key File Reference

| File | Purpose | Lines |
|------|---------|-------|
| `cpp/include/cugraph/aai/compact_graph.hpp` | Graph abstraction + factory | 151 |
| `cpp/include/cugraph/aai/cache_pool.hpp` | Thread-local LRU GPU cache | 125 |
| `cpp/include/cugraph/aai/types.hpp` | Result type definitions | 199 |
| `cpp/include/cugraph/aai/target_gpu.hpp` | GPU target macros | 52 |
| `cpp/include/cugraph/aai/api/traversal.hpp` | BFS/SSSP/K-Hop API (4-way) | 1,070 |
| `cpp/include/cugraph/aai/api/link_analysis.hpp` | PageRank/HITS API | 2,022 |
| `cpp/src/aai/impl/a100/traversal/bfs.cu` | A100 direction-optimizing BFS | ~460 |
| `cpp/src/aai/impl/a100/community/louvain_f32.cu` | A100 3-tier Louvain | ~600 |
| `cpp/src/aai/impl/a100/link_analysis/pagerank.cu` | A100 SpMV PageRank | ~314 |
| `cpp/src/aai/integration/traversal/bfs.cu` | BFS routing dispatcher | ~200 |
| `cpp/CMakeLists.txt` | Build system (GPU selection) | 969 |
| `build_in_docker/target_gpu_map.json` | GPU → SM arch mapping | ~10 |
