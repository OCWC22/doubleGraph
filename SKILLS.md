# SKILLS.md — DoubleGraph: Complete Replication Guide

> **Purpose:** A zero-context coding agent can read this file and replicate DoubleGraph's GPU-specific graph algorithm generation for A10G, A100, H100, B200, or any future NVIDIA GPU.
>
> **What this covers:** Every architectural decision, every kernel pattern, every constant, every threshold, the complete build pipeline, what worked, what didn't, and why — distilled from the actual source code.

---

## Table of Contents

1. [The Problem and Why This Solution Works](#1-the-problem-and-why-this-solution-works)
2. [System Architecture — The 4 Layers](#2-system-architecture--the-4-layers)
3. [Layer 1: The Graph Abstraction (compact_graph_t)](#3-layer-1-the-graph-abstraction-compact_graph_t)
4. [Layer 2: GPU Resource Management (CachePool)](#4-layer-2-gpu-resource-management-cachepool)
5. [Layer 3: The 4-Way Dispatch Pattern](#5-layer-3-the-4-way-dispatch-pattern)
6. [Layer 4: Per-GPU Kernel Implementations — Complete Detail](#6-layer-4-per-gpu-kernel-implementations--complete-detail)
7. [The Integration Layer — Drop-In Replacement Mechanics](#7-the-integration-layer--drop-in-replacement-mechanics)
8. [The Build System — Complete Pipeline](#8-the-build-system--complete-pipeline)
9. [Cross-GPU Comparison — What Differs and Why](#9-cross-gpu-comparison--what-differs-and-why)
10. [What Worked and Why](#10-what-worked-and-why)
11. [What Didn't Work / Limitations](#11-what-didnt-work--limitations)
12. [Hypotheses That Were Validated](#12-hypotheses-that-were-validated)
13. [Adding a New GPU Target — Step-by-Step](#13-adding-a-new-gpu-target--step-by-step)
14. [H100 and B200 — Specific Constraints and Opportunities](#14-h100-and-b200--specific-constraints-and-opportunities)
15. [Complete File Reference](#15-complete-file-reference)

---

## 1. The Problem and Why This Solution Works

### The Problem

NVIDIA cuGraph ships architecture-generic CUDA kernels. They compile for any GPU but optimize for none. This leaves massive performance on the table because:

- **Memory hierarchy differs radically**: A100 has 40MB L2 cache, A10G has 6MB. A kernel that fits its working set in A100's L2 will thrash on A10G.
- **Optimal parallelism strategy varies**: A100 has 108 SMs, L4 has 58, A10G has 80. The right grid/block configuration differs per GPU.
- **Algorithm-level strategy is hardware-dependent**: Whether BFS should use top-down or bottom-up traversal depends on how fast bitmap scans are relative to frontier expansion — which depends on memory bandwidth, L2 size, and SM count.
- **Register files differ**: Each SM generation has different register counts. Optimal register pressure (the `--maxrregcount` flag) varies per kernel per GPU.

### The Solution

Generate **per-GPU-architecture optimized CUDA kernels** that are a **drop-in replacement** for cuGraph. Same Python API, same C++ API, same numerical results, 10-100x faster on many algorithms.

### Why It Works

1. **Compile-time GPU selection** — No runtime branching overhead. Each binary is built for exactly one GPU.
2. **Per-kernel tuning** — Each of 192 kernel files per GPU has its own `.cu.flags` sidecar controlling compilation flags.
3. **Algorithm-level redesign** — Not just parameter tuning. A100 BFS uses queue↔bitmap frontier conversion. L4 BFS uses bitmap-only. A10G uses 2-level batched processing. Fundamentally different algorithms for the same API.
4. **Template specialization for drop-in** — C++ template specialization replaces cuGraph's own implementations at link time with zero API changes.

### Scale

- 3 GPU targets: A100 (SM80), L4 (SM89), A10G (SM86)
- 27 algorithm categories across 8 domains
- 192 `.cu` kernel files per GPU (576 total)
- 172 `.cu.flags` per-kernel flag files per GPU
- Drop-in via 27 integration files with `#ifdef` routing

---

## 2. System Architecture — The 4 Layers

```
cuGraph Python API → pylibcugraph (Cython) → libcugraph C++ templates
                                                     ↓
┌──────────────────────────────────────────────────────────────────────┐
│  INTEGRATION LAYER  (cpp/src/aai/integration/)                       │
│  27 .cu files. Each has #ifdef AAI_ROUTE_<ALGO>.                     │
│  When defined: C++ template specialization overrides cuGraph impl.   │
│  When not defined: #else falls through to original cuGraph code.     │
│  Pattern: handle.sync_stream() → aai::algo() → cudaDeviceSync()     │
├──────────────────────────────────────────────────────────────────────┤
│  AAI IMPLEMENTATION  (cpp/src/aai/impl/{a100,l4,a10g}/)              │
│  192 .cu files per GPU. The actual optimized CUDA kernels.           │
│  Organized: impl/<gpu>/<category>/<algorithm>_<variant>.cu           │
│  Each file has optional .cu.flags sidecar for compilation flags.     │
├──────────────────────────────────────────────────────────────────────┤
│  AAI HEADERS  (cpp/include/cugraph/aai/)                             │
│  compact_graph.hpp — Universal CSR/CSC input type                    │
│  cache_pool.hpp    — Thread-local LRU GPU resource cache             │
│  types.hpp         — Result types (device pointers + metadata)       │
│  target_gpu.hpp    — AAI_GPU_A100/L4/A10G macros                     │
│  api/*.hpp         — Algorithm declarations with 4-way dispatch      │
├──────────────────────────────────────────────────────────────────────┤
│  BUILD SYSTEM  (cpp/CMakeLists.txt + .cu.flags + target_gpu_map)     │
│  -DTARGET_GPU=A100 → selects impl/a100/ dir → globs .cu files       │
│  Reads .cu.flags sidecars → applies per-file CUDA compile flags      │
│  --rdc=true files → separate cugraph_aai_rdc static library          │
│  Sets -DAAI_ROUTE_<ALGO> on each integration file                    │
└──────────────────────────────────────────────────────────────────────┘
```

**Data flow for a BFS call:**
1. Python: `cugraph.bfs(G, source=0)`
2. Cython: calls `cugraph::bfs<int32_t, int32_t, false>(...)`
3. Integration `bfs.cu`: `#ifdef AAI_ROUTE_BFS` is defined → template specialization activates
4. Integration: `handle.sync_stream()` → extracts `compact_graph` → 4-way dispatch → calls `aai::bfs()`
5. Impl `a100/traversal/bfs.cu`: runs direction-optimizing BFS with A100-tuned kernels
6. Integration: `cudaDeviceSynchronize()` → returns to cuGraph

---

## 3. Layer 1: The Graph Abstraction (compact_graph_t)

**File:** `cpp/include/cugraph/aai/compact_graph.hpp` (151 lines)

### Why It Exists

cuGraph's `graph_view_t` is a complex template type supporting multi-GPU, int64, DCS (hypersparse), and many features AAI doesn't need. `compact_graph_t` strips it to essentials, giving kernels a clean, predictable input.

### Complete Data Structure

```cpp
template <typename vertex_t, typename edge_t>
struct compact_graph_t {
    // === Core CSR/CSC (DEVICE memory) ===
    const edge_t* offsets;           // Size: num_vertices + 1. offsets[i] = start of vertex i's edges
    const vertex_t* indices;         // Size: num_edges. Neighbor IDs (dst if CSR, src if CSC)

    // === Counts ===
    vertex_t number_of_vertices;
    edge_t number_of_edges;          // = offsets[number_of_vertices] - offsets[0]

    // === Properties ===
    bool is_symmetric;               // True = undirected (edge u→v implies v→u)
    bool is_multigraph;              // True = duplicate edges may exist
    bool is_csc;                     // false=CSR (row=src), true=CSC (row=dst)

    // === Degree-Based Vertex Segmentation (HOST memory) ===
    // When present, vertices are sorted by degree descending.
    // 5 elements define 4 segments:
    //   [0]→[1]: high-degree (degree >= 1024)
    //   [1]→[2]: mid-degree  (32 <= degree < 1024)
    //   [2]→[3]: low-degree  (1 <= degree < 32)
    //   [3]→[4]: zero-degree (isolated)
    // segment_offsets[0] == 0, segment_offsets[4] == number_of_vertices
    std::optional<std::vector<vertex_t>> segment_offsets;

    // === Edge Mask (DEVICE memory) ===
    // Packed uint32_t bitmask. Bit j of word i = edge j status.
    // (edge_mask[j/32] >> (j%32)) & 1: 1=active, 0=masked
    // nullptr = no mask (all edges active)
    const uint32_t* edge_mask = nullptr;
};

using graph32_t = compact_graph_t<int32_t, int32_t>;
```

### The Factory Method

```cpp
template <bool is_csc_>
static compact_graph_t from_graph_view(
    cugraph::graph_view_t<vertex_t, edge_t, is_csc_, false> const& gv)
```

**Constraints enforced:**
- Single-GPU only (`gv.number_of_local_edge_partitions() != 1` → throws)
- Exactly 5 segment offsets if present (DCS/hypersparse throws: "AAI requires exactly 5 segment offsets")
- Non-null offsets pointer
- Non-null indices for non-empty graphs

**What it extracts:**
- `offsets` and `indices` from edge partition view
- `segment_offsets` moved from graph view (optional)
- `edge_mask` from mask view if present, else nullptr
- `is_csc` from compile-time template parameter `is_csc_`

### Why This Design Works

Every AAI kernel takes `const graph32_t&`. No templates in hot paths. The type alias `graph32_t = compact_graph_t<int32_t, int32_t>` enforces the int32 constraint at the type level.

---

## 4. Layer 2: GPU Resource Management (CachePool)

**File:** `cpp/include/cugraph/aai/cache_pool.hpp` (125 lines)

### Why It Exists

Graph algorithms called repeatedly (e.g., in iterative workflows) would otherwise do `cudaMalloc`/`cudaFree` on every call. CachePool eliminates this by keeping GPU buffers alive across calls.

### Complete Implementation

```cpp
struct Cacheable {
    virtual ~Cacheable() = default;  // Enables type-erased cleanup
};

class CachePool {
    static constexpr size_t DEFAULT_CAPACITY = 8;  // Max 8 caches per thread

    // LRU tracking: front = Most Recently Used, back = Least Recently Used
    std::list<const void*> order_;
    std::unordered_map<const void*, Entry> map_;  // tag → {unique_ptr<Cacheable>, iterator}

    template <typename T>
    T& acquire(const void* tag) {
        // If tag exists: promote to MRU, return reference
        // If tag missing: evict LRU if at capacity (destructor frees GPU mem), create new T()
    }
};

// Thread-local singleton — each thread gets its own pool
inline CachePool& cache_pool() {
    thread_local static CachePool pool;
    return pool;
}
```

### Usage Pattern in Every Kernel File

```cpp
struct Cache : Cacheable {
    int32_t* frontier_buf = nullptr;
    uint32_t* visited = nullptr;
    int64_t frontier_buf_capacity = 0;
    int64_t visited_capacity = 0;

    void ensure(int32_t num_vertices) {
        // Only reallocate if capacity insufficient
        if (frontier_buf_capacity < 2LL * num_vertices) {
            if (frontier_buf) cudaFree(frontier_buf);
            cudaMalloc(&frontier_buf, 2LL * num_vertices * sizeof(int32_t));
            frontier_buf_capacity = 2LL * num_vertices;
        }
        // ... same for visited bitmap
    }

    ~Cache() override {
        if (frontier_buf) cudaFree(frontier_buf);
        if (visited) cudaFree(visited);
    }
};

void bfs(const graph32_t& graph, ...) {
    static int tag;  // Static local = stable address = unique key per algorithm
    auto& cache = cache_pool().acquire<Cache>(&tag);
    cache.ensure(graph.number_of_vertices);
    // Use cache.frontier_buf, cache.visited, etc.
}
```

### Key Design Decisions

- **Thread-local** — No locking needed. Each CUDA stream typically runs on one thread.
- **LRU eviction** — When 9th algorithm runs, least-recently-used cache is destroyed, freeing GPU memory.
- **Type erasure** — `unique_ptr<Cacheable>` stores heterogeneous cache types (each algorithm defines its own Cache struct).
- **Lazy allocation** — `ensure()` only reallocates when the graph is larger than the cached buffers.

---

## 5. Layer 3: The 4-Way Dispatch Pattern

**Files:** `cpp/include/cugraph/aai/api/*.hpp`

### The Pattern

Every algorithm has up to **4 separate function implementations** based on two boolean properties:

| | No edge_mask | Has edge_mask |
|---|---|---|
| **No segment_offsets** | `bfs()` | `bfs_mask()` |
| **Has segment_offsets** | `bfs_seg()` | `bfs_seg_mask()` |

Each variant is a **separate `.cu` file** with a **separate kernel implementation**. Not runtime branching — completely different code paths.

### Why 4 Separate Implementations

- **Segmented graphs** have vertices sorted by degree. Kernels can assign warps to high-degree vertices and threads to low-degree ones. Without segments, a single strategy must handle all vertices.
- **Masked graphs** have a bitmask over edges. Kernels must check the mask before processing each edge. Without masks, this check is eliminated entirely.
- The combinations multiply: seg+mask kernels must handle both degree-aware scheduling AND edge filtering.

### Additional Dispatch Dimensions

Some algorithms have more variants:
- **Precision:** `_f32` vs `_f64` (e.g., `pagerank_f32.cu`, `pagerank_f64.cu`)
- **Weights:** Weighted vs unweighted overloads
- **Personalization:** Standard vs personalized (PageRank adds personalization vertices/values)
- **Direction optimization:** `bfs` vs `bfs_direction_optimizing` (separate file, separate algorithm)

### File Naming Convention

```
impl/<gpu>/<category>/<algorithm>[_<variant>].cu
impl/<gpu>/<category>/<algorithm>[_<variant>].cu.flags   (optional sidecar)
```

Examples:
```
impl/a100/traversal/bfs.cu                    # Base BFS
impl/a100/traversal/bfs_seg.cu                # Segmented BFS
impl/a100/traversal/bfs_mask.cu               # Masked BFS (uses cooperative kernels)
impl/a100/traversal/bfs_direction_optimizing.cu
impl/a100/link_analysis/pagerank_f32.cu       # Float PageRank
impl/a100/link_analysis/pagerank_f32_seg.cu
impl/a100/community/louvain_f32.cu
impl/a100/community/louvain_f64_seg.cu
```

### Complete Algorithm Matrix

27 algorithm categories, each with 4+ variants = 192 files per GPU:

| Category | Algorithms | Variants Per |
|----------|-----------|-------------|
| **Traversal** | BFS, BFS-DO, SSSP, K-Hop | 4 each + precision |
| **Link Analysis** | PageRank, HITS | 10+ (mask×seg×weights×personalization×precision) |
| **Centrality** | Betweenness, Eigenvector, Katz | 4-20 each |
| **Community** | Louvain, Leiden, ECG, K-Truss, TriangleCount, Egonet, Spectral, Clustering×3 | 4-12 each |
| **Components** | WCC, SCC | 4, 2 |
| **Cores** | CoreNumber, K-Core | 4 each |
| **Link Prediction** | Jaccard, Cosine, Overlap, Sorensen | 6 each (includes all-pairs) |
| **Tree** | MST | 4 (f32/f64 × base/seg) |

---

## 6. Layer 4: Per-GPU Kernel Implementations — Complete Detail

### 6.1 BFS on A100 — Direction-Optimizing with Queue↔Bitmap Conversion

**File:** `cpp/src/aai/impl/a100/traversal/bfs.cu` (459 lines)

#### Cache Structure
```cpp
struct Cache : Cacheable {
    int32_t* frontier_buf = nullptr;     // 2 × num_vertices int32s (double-buffered queue)
    uint32_t* visited = nullptr;         // ceil(num_vertices/32) uint32 words (bitmap)
    uint32_t* frontier_bmp = nullptr;    // Same size as visited (for bottom-up mode)
    uint32_t* new_frontier_bmp = nullptr;// Same size (swap target)
    int32_t* count = nullptr;            // Single int32 (frontier size counter)
};
```

#### Kernels (6 total)

1. **`bfs_init_kernel`** — Block=256, grid=min(n/256, 8192). Sets distances to `0x7fffffff`, predecessors to -1, visited bitmap to 0.

2. **`bfs_set_sources_kernel`** — Block=256, grid=ceil(n_sources/256). Sets source distances=0, adds to frontier queue, sets visited bits via `atomicOr`.

3. **`queue_to_bitmap_kernel`** — Converts queue→bitmap. Block=256, grid=min(queue_size/256, 4096). Each thread: `atomicOr(&bitmap[v >> 5], 1u << (v & 31))`.

4. **`bitmap_to_queue_kernel`** — Converts bitmap→queue. Uses **warp-level ballot** to parallelize bit scanning:
   ```cuda
   // Each warp processes one 32-bit word
   bool is_set = (v < num_vertices) && ((word >> lane) & 1u);
   unsigned ballot = __ballot_sync(0xffffffffu, is_set);
   int cnt = __popc(ballot);
   if (lane == 0) base = atomicAdd(queue_count, cnt);  // One atomic per warp, not per vertex
   base = __shfl_sync(0xffffffffu, base, 0);           // Broadcast to all lanes
   int offset = __popc(ballot & ((1u << lane) - 1u));   // Lane-local prefix count
   queue[base + offset] = v;
   ```

5. **`bfs_topdown_queue_warp_kernel`** — **The core top-down kernel.** Block=256, grid=min(frontier_size/8, 4096). One warp (32 threads) per frontier vertex:
   ```cuda
   for (int i = warp_id; i < frontier_size; i += total_warps) {
       int32_t src = cur_frontier[i];
       for (int32_t e = start + lane; e < end; e += 32) {  // Warp-stride over edges
           int32_t dst = indices[e];
           uint32_t word = __ldg(word_ptr);     // L2 cache hint
           if ((word & mask) == 0u) {
               uint32_t old = atomicOr(word_ptr, mask);
               if ((old & mask) == 0u) {        // Won the race
                   distances[dst] = new_depth;
                   // Warp-cooperative output:
                   unsigned ballot = __ballot_sync(0xffffffffu, is_new);
                   if (lane == 0) base = atomicAdd(next_count, __popc(ballot));
                   base = __shfl_sync(0xffffffffu, base, 0);
                   next_frontier[base + __popc(ballot & ((1u << lane) - 1u))] = dst;
               }
           }
       }
   }
   ```

6. **`bfs_bottomup_kernel`** — Block=256, grid=min(num_vertices/256, 65535). One thread per vertex. Scans edges for frontier membership, breaks on first hit:
   ```cuda
   for (int32_t e = start; e < end; e++) {
       int32_t u = indices[e];
       if (frontier_bitmap[u >> 5] & (1u << (u & 31))) {
           parent = u;
           break;  // Early termination — key optimization
       }
   }
   ```

#### Main BFS Flow — Step by Step

```
1. Acquire cache, ensure buffers for num_vertices
2. Init: distances=INF, visited=0, sources→frontier queue, sources→visited bitmap
3. Set thresholds:
     td_to_bu = max(num_vertices / 20, 1)    # Switch to bottom-up when frontier exceeds 5%
     bu_to_td = max(num_vertices / 200, 1)    # Switch back when frontier drops below 0.5%
     use_do = is_symmetric && (num_vertices >= 10000)

4. While frontier_size > 0 && depth < max_depth:
   IF top-down mode:
     a. Clear next_count
     b. Launch bfs_topdown_queue_warp_kernel (warp-per-source)
     c. Copy next_count to host (cudaMemcpyAsync + sync)
     d. Swap frontier buffers (cur ↔ next)
     e. Check switch: if use_do && fsize > td_to_bu && fsize >= prev_fsize → switch to bottom-up
        i.  Clear frontier_bmp
        ii. Launch queue_to_bitmap (convert current frontier)
        iii. Set topdown = false

   ELSE (bottom-up mode):
     a. Clear next count and new_frontier_bmp
     b. Launch bfs_bottomup_kernel (thread-per-vertex, bitmap frontier)
     c. Copy count to host
     d. Swap frontier bitmaps (frontier_bmp ↔ new_frontier_bmp)
     e. Check switch: if fsize > 0 && fsize < bu_to_td && fsize < prev_fsize → switch to top-down
        i.  Clear count
        ii. Launch bitmap_to_queue (convert back to queue)
        iii. Copy queue size to host
        iv. Set topdown = true
```

#### Why These Thresholds (A100-Specific)

- **td_to_bu = N/20 (5%)**: A100 has 40MB L2 cache. At 5% of vertices in the frontier, bottom-up (scanning all vertices) becomes cheaper than top-down (expanding all frontier edges), because the frontier bitmap fits in L2.
- **bu_to_td = N/200 (0.5%)**: When the frontier is very sparse, bottom-up wastes time scanning unvisited vertices with no frontier neighbors.
- **N >= 10000**: Direction-optimizing overhead (bitmap conversion, mode switching) isn't worthwhile for small graphs.
- **fsize >= prev_fsize**: Only switch if frontier is growing (monotone heuristic to avoid thrashing).

---

### 6.2 BFS on L4 — CUB-Based with Degree-Cost Model

**File:** `cpp/src/aai/impl/l4/traversal/bfs.cu` (460 lines)

#### Key Differences from A100

| Aspect | A100 | L4 |
|--------|------|-----|
| **Frontier storage** | Queue + bitmap (dual) | Queue + bitmap (but different switching) |
| **Init kernel** | Scalar loop | **Vectorized int4** writes (4 ints at once) |
| **Switching model** | Simple N/20, N/200 | **Alpha-beta cost model** with frontier degree sum |
| **Degree computation** | Not computed | **CUB BlockReduce** for frontier degree sum |
| **Grid cap (top-down)** | 4096 | 65535 |
| **Grid cap (bottom-up)** | 65535 | 2048 |
| **Template usage** | Runtime `compute_pred` flag | **Compile-time `<ComputePred>` template** |

#### L4's Alpha-Beta Cost Model

```cpp
double avg_degree = (double)num_edges / num_vertices;
double alpha = avg_degree * 0.25;   // Scaling factor for frontier edge cost
const int32_t beta = 24;             // Scaling factor for BU→TD switch

// TD → BU switch condition:
// Only when frontier > 100 AND growing:
if (frontier_size > 100 && frontier_size >= prev_frontier_size) {
    // Compute actual frontier degree sum using CUB BlockReduce:
    sum_frontier_degrees_kernel<<<grid, 256>>>(offsets, frontier, size, d_stats);
    // Copy to host
    double m_f = (double)frontier_edges;
    double unvisited_frac = 1.0 - (double)total_visited / num_vertices;
    double m_u_est = (double)num_edges * unvisited_frac;

    if (m_f * alpha > m_u_est) {  // Frontier edge cost exceeds unvisited edge scan cost
        topdown_mode = false;
    }
}

// BU → TD switch condition:
if (frontier_size * beta < unvisited_vertices && frontier_size < prev_frontier_size) {
    topdown_mode = true;
}
```

#### L4's Vectorized Init (int4 writes)

```cuda
// L4 writes 4 ints at once via int4 — 4x fewer memory transactions
int32_t vec4_count = num_vertices >> 2;
for (int32_t i = tid; i < vec4_count; i += stride) {
    reinterpret_cast<int4*>(distances)[i] = make_int4(0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF);
}
// Handle remainder
for (int32_t i = (vec4_count << 2) + tid; i < num_vertices; i += stride) {
    distances[i] = 0x7FFFFFFF;
}
```

#### L4's Bottom-Up Kernel

L4's bottom-up also outputs to a queue (not just bitmap), using `atomicAdd` for position. It updates the visited bitmap in-place (same array for read and write — relies on atomicOr semantics).

---

### 6.3 BFS on A10G — 2-Level Batching with Pinned Memory

**File:** `cpp/src/aai/impl/a10g/traversal/bfs.cu` (562 lines)

#### Key Differences from A100 and L4

| Aspect | A100 | L4 | A10G |
|--------|------|-----|------|
| **2-level batching** | No | No | **Yes** — processes 2 BFS levels per host sync |
| **Pinned host memory** | No | No | **Yes** — `cudaHostAlloc` for frontier size counter |
| **Frontier rebuild** | Not needed | Not needed | **Yes** — `build_frontier_from_distances` kernel |
| **Switching constants** | N/20, N/200 | alpha=deg×0.25, beta=24 | **ALPHA=4, BETA=24** |
| **prev_visited bitmap** | Not used | Not used | **Maintained** — for frontier bitmap construction |
| **Grid cap (top-down)** | 4096 | 65535 | **1280** |

#### A10G's 2-Level Batching

The key innovation: when in top-down mode with a reasonably sized frontier, A10G processes **two BFS levels** in a single host synchronization:

```cpp
bool do_batch = topdown &&
                current_depth + 2 <= depth_limit &&
                h_frontier_size >= 32 &&
                (!can_do_bottomup || h_frontier_size <= num_vertices / (ALPHA * 2));

if (do_batch) {
    // Level 1: frontier_a → frontier_b (counter_b)
    bfs_topdown_kernel<<<...>>>(offsets, indices, ...,
        frontier_a, frontier_b, counter_b, frontier_size, depth + 1);

    // Zero counter_a for level 2
    zero_counter<<<1, 1>>>(counter_a);

    // Level 2: frontier_b → frontier_a (counter_a)
    // This kernel reads frontier_size from device memory (counter_b), not host
    bfs_topdown_devsize_kernel<<<1280, 256>>>(offsets, indices, ...,
        frontier_b, frontier_a, counter_b, counter_a, depth + 2);

    // Single host sync for both levels
    cudaMemcpyAsync(h_pinned, counter_a, sizeof(int32_t), ...);
    cudaStreamSynchronize(stream);
    h_frontier_size = h_pinned[0];
    current_depth += 2;
}
```

**Why this helps A10G:** A10G has lower memory bandwidth than A100. The kernel launch + memcpy + sync overhead is proportionally more expensive. Batching 2 levels halves this overhead.

#### A10G's `bfs_topdown_devsize_kernel`

A special kernel that reads frontier size from **device memory** instead of a host parameter:
```cuda
template <bool COMPUTE_PRED>
__global__ void bfs_topdown_devsize_kernel(..., int32_t* frontier_size_in, ...) {
    int32_t fsize = *frontier_size_in;  // Read from device — set by previous kernel
    if (fsize <= 0) return;
    // ... same warp-per-source BFS as regular kernel
}
```

#### A10G's Frontier Rebuild on BU→TD Switch

When switching from bottom-up back to top-down, A10G doesn't have a queue — it must rebuild one from the distances array:

```cuda
__global__ void build_frontier_from_distances(
    const int32_t* distances, int32_t* frontier, int32_t* frontier_size,
    int32_t num_vertices, int32_t target_depth) {
    for (int32_t v = tid; v < num_vertices; v += stride) {
        if (distances[v] == target_depth) {
            int32_t pos = atomicAdd(frontier_size, 1);
            frontier[pos] = v;
        }
    }
}
```

#### A10G's Pinned Memory

```cpp
cudaHostAlloc(&h_pinned, 2 * sizeof(int32_t), cudaHostAllocDefault);
// Used for async frontier size transfer — avoids implicit sync of regular cudaMemcpy
cudaMemcpyAsync(cache.h_pinned, counter, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
cudaStreamSynchronize(stream);
h_frontier_size = cache.h_pinned[0];
```

---

### 6.4 Louvain Community Detection on A100 — 3-Tier Dispatch

**File:** `cpp/src/aai/impl/a100/community/louvain_f32.cu` (626 lines)

#### Constants
```
BLOCK_SIZE = 256
WARPS_PER_BLOCK = 8 (256/32)
HT_SIZE_PER_WARP = 128 (shared memory hash table entries per warp)
Thread-level HT_SIZE = 64 (register-based)
Hash function: (uint32_t)(community_id * 2654435761u) % HT_SIZE  (Knuth multiplicative hash)
Max local move iterations per level = 100
```

#### 3-Tier Kernel Dispatch Logic

```cpp
if (cur_n <= 200) {
    // TIER 1: Single thread. CPU-like serial loop.
    local_move_serial<<<1, 1>>>(offsets, indices, weights, communities,
                                sigma_tot, k, n, total_weight, resolution);
}
else if (cur_avg_deg < 8) {
    // TIER 2: One thread per vertex. Register-based hash table (64 entries).
    local_move_thread<<<grid, 256>>>(offsets, indices, weights, communities,
                                    sigma_tot, k, n, total_weight, resolution);
}
else {
    // TIER 3: One warp per vertex. Shared-memory hash table (128 entries/warp).
    local_move_warp<<<grid, 256>>>(offsets, indices, weights, communities,
                                  sigma_tot, k, n, total_weight, resolution);
}
```

#### Shared Memory Hash Table (Tier 3)

Each warp gets 128 entries in shared memory for community → weight accumulation:

```cuda
// Shared memory layout: 8 warps × 128 entries × (int32 key + float value)
__shared__ int32_t s_ht_keys[WARPS_PER_BLOCK * HT_SIZE_PER_WARP];
__shared__ float s_ht_vals[WARPS_PER_BLOCK * HT_SIZE_PER_WARP];

// Linear probing with atomicCAS for concurrent insertion
uint32_t h = (uint32_t)(community_id * 2654435761u) % HT_SIZE_PER_WARP;
for (int probe = 0; probe < HT_SIZE_PER_WARP; probe++) {
    uint32_t slot = (h + probe) % HT_SIZE_PER_WARP;
    int32_t old_key = atomicCAS(&my_keys[slot], -1, community_id);
    if (old_key == -1 || old_key == community_id) {
        atomicAdd(&my_vals[slot], edge_weight);
        break;
    }
}
```

#### Modularity Gain Formula

```cuda
// For each neighbor community c:
float gain = (w_to_c - w_to_own_community) * (2.0f / total_weight)
           + resolution * k_v * (sigma_own - k_v - sigma_c) * (2.0f / (total_weight * total_weight));

// Best community wins. Tiebreaker: lower community ID.
if (gain > best_gain || (gain == best_gain && gain > 0.0f && c < best_community)) {
    best_gain = gain;
    best_community = c;
}
```

#### Multi-Level Coarsening Flow

```
1. init_sequence_kernel: community[v] = v for all v
2. compute_weighted_degree_kernel: k[v] = sum of edge weights for v
3. compute_sigma_tot_kernel: sigma_tot[c] = sum of k[v] for all v in community c
4. Local Move Phase (up to 100 iterations):
   a. Select kernel by tier (serial / thread / warp)
   b. Count moves via changed_flag
   c. Compute modularity, check improvement > threshold
   d. If no improvement or no moves: stop
5. Graph Coarsening:
   a. mark_used_kernel: identify active communities
   b. apply_renumber_kernel: compact community IDs
   c. map_edges_kernel: create edge list with new IDs
   d. thrust::sort_by_key on (src, dst) pairs
   e. thrust::reduce_by_key to aggregate parallel edges
   f. count_edges_kernel + inclusive_scan → new CSR offsets
6. Repeat from step 2 with coarsened graph up to max_level
7. Final modularity = sum of intra-community edge weights / total_weight
```

---

### 6.5 PageRank on A100 — Custom SpMV with Warp Reduction

**File:** `cpp/src/aai/impl/a100/link_analysis/pagerank.cu` (314 lines, basic version)
**File:** `cpp/src/aai/impl/a100/link_analysis/pagerank_f32.cu` (435 lines, cuSPARSE version)

#### Two Versions Exist

1. **Basic version** (`pagerank.cu`): Custom kernel-based SpMV. No cuSPARSE dependency.
2. **cuSPARSE version** (`pagerank_f32.cu`): Uses `cusparseSpMV()` for the matrix-vector product.

#### cuSPARSE Version — Complete Flow

```
1. Compute out-weight sums: atomicAdd edge weights to source vertex totals
2. Normalize edge weights: norm_weights[e] = weight[e] / out_weight_sums[src]
   (safe division: if out_weight_sum == 0, set to 0 — dangling node)
3. Create cuSPARSE descriptors (CSC matrix + dense vectors)
4. Allocate SpMV buffer via cusparseSpMV_bufferSize()
5. Preprocess SpMV
6. Initialize PR: pr[v] = 1.0 / N

While iter < max_iterations:
  7. dangling_sum_kernel: sum PR of vertices with zero out-weight
  8. cusparseSpMV(): compute spmv_result = A^T × pr (the core PageRank step)
  9. update_and_diff_kernel (FUSED):
     - new_pr[v] = (1 - alpha) / N + spmv_result[v] + alpha * dangling_sum / N
     - thread_diff += |new_pr[v] - old_pr[v]|
     - Block reduction → atomic global L1 diff
  10. Copy diff to host, check < epsilon
  11. Swap pr buffers
```

#### Warp-Level Reduction (Used in Steps 7 and 9)

```cuda
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float block_reduce_sum(float val) {
    __shared__ float shared[32];  // One slot per warp
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;

    val = warp_reduce_sum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);
    return val;  // Only thread 0 has correct total
}
```

---

### 6.6 WCC on A100 — Union-Find with Sampling

**File:** `cpp/src/aai/impl/a100/components/weakly_connected_components.cu` (218 lines)

#### Algorithm: Label Propagation via Parallel Union-Find

```
1. Init: parent[v] = v (each vertex is its own component)
2. Hook Sample: for each vertex, link with first k=2 neighbors (fast approximate pass)
3. Compress: full path compression (parent[v] = root(v))
4. Hook Full: for remaining non-converged vertices, process ALL neighbors
5. Compress again
6. Shortcut iterations (up to 5 rounds × 100 iterations):
   - Check if parent[v] == parent[parent[v]] for all v
   - If not: apply one step of path compression
   - Use host-mapped flag (cudaHostAllocMapped) for convergence check
```

#### Core Primitives

```cuda
__device__ int find_root(int* parent, int v) {
    int p = parent[v];
    while (p != parent[p]) {
        parent[v] = parent[p];  // Path compression (path splitting)
        v = p;
        p = parent[p];
    }
    return p;
}

__device__ void link(int* parent, int u, int v) {
    int ru = find_root(parent, u);
    int rv = find_root(parent, v);
    while (ru != rv) {
        int lo = min(ru, rv), hi = max(ru, rv);
        int old = atomicCAS(&parent[hi], hi, lo);  // Smaller ID wins
        if (old == hi) break;  // CAS succeeded
        // Re-find roots if CAS failed (concurrent modification)
        ru = find_root(parent, u);
        rv = find_root(parent, v);
    }
}
```

---

### 6.7 Betweenness Centrality on A100 — Multi-Source BFS + Backward Pass

**File:** `cpp/src/aai/impl/a100/centrality/betweenness_centrality.cu` (469 lines)

#### Algorithm

```
For each source vertex (or sampled subset):
  Phase 1 — Forward BFS:
    bfs_forward_kernel: warp-per-frontier-vertex
    Track distances AND sigma (shortest path counts)
    sigma[dst] += sigma[src] for each shortest path

  Phase 2 — Backward Accumulation:
    dependency_kernel: process levels in reverse order
    delta[v] = sigma[v] * sum((1 + delta[w]) / sigma[w]) for all next-level neighbors w
    centralities[v] += delta[v] (accumulate across sources)

  Phase 3 — Endpoint Contribution (optional):
    Add reachable_count to source, 1 to each reachable vertex
```

#### Normalization

```cuda
// Normalization depends on whether all sources are used
float scale;
if (all_sources) {
    scale = normalized ? 1.0f / (k * (adj - 1)) : 2.0f / (k * adj);  // symmetric
} else {
    // Scale differs for source vs non-source vertices
}
```

---

### 6.8 Triangle Count on A100 — DAG-Based with Bilateral Intersection

**File:** `cpp/src/aai/impl/a100/community/triangle_count.cu` (379 lines)

#### Algorithm

```
1. Compute vertex degrees
2. Build DAG: orient edges toward higher-degree vertex
   Tie-breaking: if deg[u] == deg[v], orient u → v if u < v
3. Sort DAG adjacency lists (CUB DeviceSegmentedSort)
4. Count triangles via bilateral set intersection:
   - For each edge (u, v) in DAG:
   - Intersect sorted neighbor lists of u and v
   - Each intersection = one triangle
```

#### Key Kernel: `tc_dag_vertex_kernel`
- `__launch_bounds__(256, 8)` — 8 blocks per SM
- One block per vertex
- Warp-per-edge within vertex
- Binary search with `__ldg` prefetching for cache-friendly access

---

## 7. The Integration Layer — Drop-In Replacement Mechanics

**Directory:** `cpp/src/aai/integration/`

### How Template Specialization Works

Each integration file uses `#ifdef AAI_ROUTE_<ALGO>` to conditionally replace cuGraph's implementation:

```cpp
// File: cpp/src/aai/integration/traversal/bfs.cu

#ifdef AAI_ROUTE_BFS  // Set by CMake per-file compile definition

#include <cugraph/aai/algorithms.hpp>  // AAI headers
#include <cugraph/algorithms.hpp>       // cuGraph headers (for type matching)

namespace cugraph {

// Template specialization: OVERRIDES cuGraph's bfs<int32_t, int32_t, false>
template <>
void bfs<int32_t, int32_t, false>(
    raft::handle_t const& handle,
    graph_view_t<int32_t, int32_t, false, false> const& graph_view,
    int32_t* distances, int32_t* predecessors,
    int32_t const* sources, size_t n_sources,
    bool direction_optimizing, int32_t depth_limit, bool do_expensive_check)
{
    // 1. Extract compact graph
    auto compact_graph = aai::graph32_t::from_graph_view(graph_view);

    // 2. Validate preconditions (match original bfs_impl.cuh)
    CUGRAPH_EXPECTS(!compact_graph.is_csc, "BFS requires CSR format");

    // 3. Stream sync (AAI uses default stream, cuGraph uses handle's stream)
    handle.sync_stream();

    // 4. 4-way dispatch based on graph properties
    if (direction_optimizing) {
        if (compact_graph.edge_mask != nullptr) {
            if (compact_graph.segment_offsets.has_value())
                aai::bfs_direction_optimizing_seg_mask(compact_graph, ...);
            else
                aai::bfs_direction_optimizing_mask(compact_graph, ...);
        } else {
            if (compact_graph.segment_offsets.has_value())
                aai::bfs_direction_optimizing_seg(compact_graph, ...);
            else
                aai::bfs_direction_optimizing(compact_graph, ...);
        }
    } else {
        // Same 4-way dispatch for standard BFS
    }

    // 5. Sync all device work and check for CUDA errors
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) CUGRAPH_FAIL(...);
}
}

#else  // AAI_ROUTE_BFS not defined

// Fallback: instantiate original cuGraph implementation
#include "traversal/bfs_impl.cuh"
namespace cugraph {
template void bfs<int32_t, int32_t, false>(...);
}

#endif
```

### Stream Synchronization Protocol

**Critical pattern — must be replicated exactly:**
1. `handle.sync_stream()` — Ensures cuGraph's async work on its CUDA stream is complete
2. AAI kernel executes on **stream 0** (default CUDA stream)
3. `cudaDeviceSynchronize()` — Ensures AAI's work completes before cuGraph resumes
4. `cudaGetLastError()` — Check for async CUDA errors from the kernel

### PageRank Integration — 8+ Way Dispatch

PageRank has the most complex dispatch due to multiple dimensions:

```
Dispatch dimensions:
  × mask      (has edge_mask or not)
  × personalized (has personalization vertices/values or not)
  × weighted  (has edge weights or not)
  × segmented (has segment_offsets or not)
  × precision (float or double)

Total: up to 2×2×2×2×2 = 32 paths (not all exist — e.g., unweighted double is invalid)
```

The integration file handles this with nested if/else chains, calling the appropriate `aai::pagerank_*` variant.

### Louvain Integration — Weight Requirement

```cpp
CUGRAPH_EXPECTS(edge_weight_view.has_value(), "Graph must be weighted");
CUGRAPH_EXPECTS(!compact_graph.is_csc, "Louvain requires CSR format");
// Only seg dispatch (no mask support for Louvain)
if (compact_graph.segment_offsets.has_value())
    result = aai::louvain_seg(compact_graph, weights, clustering, max_level, threshold, resolution);
else
    result = aai::louvain(compact_graph, weights, clustering, max_level, threshold, resolution);
```

---

## 8. The Build System — Complete Pipeline

**File:** `cpp/CMakeLists.txt` (969 lines)

### Step 1: GPU Target Selection

```cmake
# User provides: cmake -DTARGET_GPU=A100 ..
set(TARGET_GPU "" CACHE STRING "Target GPU for AAI kernels (A100, L4, A10G)")

if(TARGET_GPU STREQUAL "A100")
    set(AAI_TARGET_GPU_VALUE "AAI_GPU_A100")   # Compile definition value
    set(AAI_TARGET_GPU_SUFFIX "a100")           # Directory suffix
elseif(TARGET_GPU STREQUAL "L4")
    set(AAI_TARGET_GPU_VALUE "AAI_GPU_L4")
    set(AAI_TARGET_GPU_SUFFIX "l4")
elseif(TARGET_GPU STREQUAL "A10G")
    set(AAI_TARGET_GPU_VALUE "AAI_GPU_A10G")
    set(AAI_TARGET_GPU_SUFFIX "a10g")
else()
    message(FATAL_ERROR "Invalid TARGET_GPU. Must be A100, L4, or A10G.")
endif()
```

### Step 2: Algorithm Routing Configuration

```cmake
# All 27 algorithm categories
set(_AAI_ALL_ALGORITHMS
    BFS SSSP K_HOP_NBRS PAGERANK HITS
    BETWEENNESS_CENTRALITY EIGENVECTOR_CENTRALITY KATZ_CENTRALITY
    LOUVAIN LEIDEN ECG TRIANGLE_COUNT K_TRUSS EGONET
    SPECTRAL_MODULARITY_MAXIMIZATION
    ANALYZE_CLUSTERING_MODULARITY ANALYZE_CLUSTERING_EDGE_CUT ANALYZE_CLUSTERING_RATIO_CUT
    WCC SCC CORE_NUMBER K_CORE COSINE JACCARD OVERLAP SORENSEN MST
)

# AAI_ROUTED_ALGORITHMS="" → all routed (default)
# AAI_ROUTED_ALGORITHMS="NONE" → all disabled
# AAI_ROUTED_ALGORITHMS="BFS;SSSP" → only those routed
```

### Step 3: Impl File Collection

```cmake
set(_AAI_IMPL_DIR "src/aai/impl/${AAI_TARGET_GPU_SUFFIX}")

# Each algorithm maps to one or more glob patterns
set(_AAI_IMPL_GLOBS_BFS "${_AAI_IMPL_DIR}/traversal/bfs*")
set(_AAI_IMPL_GLOBS_PAGERANK "${_AAI_IMPL_DIR}/link_analysis/pagerank*")
# ... 27 total mappings

# Collect only .cu files (not .cu.flags sidecars)
foreach(_algo ${_aai_route_list})
    file(GLOB _matched "${_AAI_IMPL_GLOBS_${_algo}}")
    foreach(_f ${_matched})
        if(_f MATCHES "\\.cu$")
            list(APPEND AAI_IMPL_SOURCES "${_f}")
        endif()
    endforeach()
endforeach()
```

### Step 4: Per-File .cu.flags Processing

```cmake
foreach(_cu_file ${AAI_IMPL_SOURCES})
    if(EXISTS "${_cu_file}.flags")
        file(STRINGS "${_cu_file}.flags" _extra_flags)

        # Separate RDC flag from other flags
        if("--rdc=true" IN_LIST _extra_flags)
            list(APPEND AAI_RDC_SOURCES "${_cu_file}")
            list(REMOVE_ITEM _extra_flags "--rdc=true")
        else()
            list(APPEND _AAI_NON_RDC_IMPL_SOURCES "${_cu_file}")
        endif()

        # Apply remaining flags as per-file CUDA compile options
        if(_extra_flags)
            set_source_files_properties("${_cu_file}"
                PROPERTIES COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CUDA>:${_extra_flags}>")
        endif()
    endif()
endforeach()
```

### Step 5: RDC Sub-Library

Files requiring cooperative kernels (grid-wide `__syncthreads`) need `CUDA_SEPARABLE_COMPILATION=ON`, which must be set at the **target level**. These files go into a separate static library:

```cmake
if(AAI_RDC_SOURCES)
    add_library(cugraph_aai_rdc STATIC ${AAI_RDC_SOURCES})
    set_target_properties(cugraph_aai_rdc PROPERTIES
        CUDA_SEPARABLE_COMPILATION  ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        POSITION_INDEPENDENT_CODE   ON
    )
    target_compile_definitions(cugraph_aai_rdc
        PRIVATE AAI_TARGET_GPU=${AAI_TARGET_GPU_VALUE})
endif()
```

### Step 6: Per-File Routing Definitions

```cmake
# Map algorithm names to integration file paths
set(_algo_file_map
    "BFS:traversal/bfs.cu"
    "PAGERANK:link_analysis/pagerank.cu"
    "LOUVAIN:community/louvain.cu"
    # ... 27 total mappings
)

# Set -DAAI_ROUTE_<ALGO> on each integration file
foreach(_algo ${_aai_route_list})
    # Find matching file in _algo_file_map
    set(_integration_file "src/aai/integration/${_map_file}")
    set_property(SOURCE ${_integration_file}
        APPEND PROPERTY COMPILE_DEFINITIONS "AAI_ROUTE_${_algo}")
endforeach()
```

### .cu.flags Observed in Practice

| Flag | Files Using | Purpose |
|------|------------|---------|
| `--use_fast_math` | 172 (nearly all) | Relaxed IEEE754 float ops for speed |
| `--rdc=true` | 5 (bfs_mask, core_number×2, eigenvector×2) | Cooperative kernel support |
| `--maxrregcount=48` | leiden_f32 | Limit register pressure → increase occupancy |
| `--maxrregcount=64` | sorensen_all_pairs_f32_seg | Allow more registers for compute-heavy kernel |
| `--extra-device-vectorization` | WCC, cosine_all_pairs_f64_seg | Enable NVCC vectorized memory ops |

### Docker Build Pipeline

```
build_in_docker/target_gpu_map.json:
    { "a100": {"arch": "80"}, "l4": {"arch": "89"}, "a10g": {"arch": "86"} }

build_image.py --target-gpu A100
    → Reads target_gpu_map.json
    → Passes -DTARGET_GPU=A100 -DCMAKE_CUDA_ARCHITECTURES=80

wheel_building/build_wheel.py --wheel-target-gpu A100
    → Builds wheel for single GPU target
    → Outputs to dist/arch-80/
```

---

## 9. Cross-GPU Comparison — What Differs and Why

### BFS Strategy

| | A100 (SM80) | L4 (SM89) | A10G (SM86) |
|---|---|---|---|
| **L2 Cache** | 40 MB | 48 MB | 6 MB |
| **SMs** | 108 | 58 | 80 |
| **Frontier repr** | Queue + Bitmap (dual, with conversion kernels) | Queue + Bitmap (degree-cost model switching) | Queue + Bitmap (2-level batch, frontier rebuild) |
| **Init kernel** | Scalar loops | Vectorized int4 writes | Scalar loops |
| **TD→BU trigger** | `fsize > N/20 && fsize >= prev` | `frontier_degree_sum × alpha > unvisited_edges` (alpha = avg_deg × 0.25) | `fsize > N/ALPHA (ALPHA=4) && fsize >= prev` |
| **BU→TD trigger** | `fsize < N/200 && fsize < prev` | `fsize × beta < unvisited_verts` (beta=24) | `fsize × BETA < N` (BETA=24) |
| **Extra optimization** | None needed (large L2) | CUB BlockReduce for degree sum | 2-level batching, pinned host memory, frontier rebuild |
| **Cooperative kernels** | Yes (bfs_mask.cu) | No | No |
| **Template specialization** | Runtime flag `int compute_pred` | Compile-time `<ComputePred>` template | Compile-time `<COMPUTE_PRED>` template |
| **Grid cap (TD)** | 4096 | 65535 | 1280 |
| **Grid cap (BU)** | 65535 | 2048 | 1280 |

### Why These Differences Exist

- **A100 uses queue↔bitmap conversion** because its 40MB L2 can hold the full bitmap (~128KB for 1M vertices). Converting between representations is fast when L2 holds both.
- **L4 uses a degree-cost model** because its 48MB L2 is even larger, but with fewer SMs (58 vs 108), the switching threshold should be based on actual work cost, not just frontier size.
- **A10G uses 2-level batching** because with only 6MB L2, the per-iteration overhead (kernel launch + host sync + memcpy) is proportionally larger. Processing 2 levels per sync halves this cost.
- **A10G uses pinned memory** because `cudaHostAlloc` enables truly async transfers, avoiding implicit synchronization that `cudaMemcpy` would cause.
- **A10G needs frontier rebuild** because during bottom-up mode it doesn't maintain a queue — only the visited bitmap changes. When switching back to top-down, it must scan the distances array to reconstruct the queue.

### PageRank Strategy

| | A100 | L4 | A10G |
|---|---|---|---|
| **SpMV** | cuSPARSE or custom | Custom (warp/thread kernel choice) | Custom (separate launch functions) |
| **Kernel selection** | Single kernel | Degree-based (avg_deg ≥ 6 → warp, else thread) | Fixed choice |
| **L2 management** | None (relies on 40MB L2) | None | Explicit `set_l2_persist()` on PR buffer |
| **Pinned memory** | No | No | Yes (`h_l1_norm_pinned` for convergence check) |
| **Launch bounds** | Auto | Auto | `__launch_bounds__(256, 6)` |
| **Code organization** | Single function | Inline | Separate launch functions |

---

## 10. What Worked and Why

### Strategy 1: Compile-Time GPU Selection (Worked Perfectly)

**Decision:** Build separate binaries per GPU instead of runtime detection.
**Why it worked:** Eliminates all runtime branching in kernel hot paths. The compiler can optimize each kernel independently. No binary bloat from shipping all GPU variants.
**Trade-off accepted:** Users must install the correct wheel for their GPU. Solved by the build pipeline generating per-GPU wheels.

### Strategy 2: compact_graph_t Abstraction (Worked Perfectly)

**Decision:** Strip cuGraph's complex graph_view_t to a minimal CSR/CSC struct.
**Why it worked:** Every kernel gets exactly the data it needs, no more. No template metaprogramming in hot paths. The factory method `from_graph_view()` encapsulates all complexity.
**Key insight:** The segment_offsets and edge_mask are the only "optional" features worth supporting — they enable real algorithmic improvements (degree-aware scheduling, subgraph operations).

### Strategy 3: CachePool for GPU Resources (Worked Well)

**Decision:** Thread-local LRU cache with capacity 8.
**Why it worked:** Graph algorithms are often called repeatedly (iterative workflows, batch processing). Eliminating cudaMalloc/cudaFree cycles reduced overhead significantly.
**Capacity 8 rationale:** Most workflows use 2-4 algorithms repeatedly. 8 gives headroom without excessive GPU memory usage.

### Strategy 4: 4-Way Dispatch with Separate Files (Worked Well)

**Decision:** Separate `.cu` file per variant instead of runtime branching.
**Why it worked:** Each variant can use completely different kernel code. The seg variant can use degree-aware scheduling. The mask variant can use edge-filtering logic. No runtime overhead for checking which features are active.
**Trade-off:** 4x more files per algorithm. Managed via glob patterns in CMake.

### Strategy 5: Per-File .cu.flags Sidecars (Worked Well)

**Decision:** Instead of global CUDA flags, each kernel gets its own flags file.
**Why it worked:** `--use_fast_math` is safe for most kernels but not all. `--maxrregcount` needs different values per kernel. `--rdc=true` only applies to cooperative kernels. Sidecars make this per-kernel control trivial.
**Key insight:** The sidecar pattern is simple to implement in CMake (file(STRINGS) + set_source_files_properties) and doesn't require changing the kernel code.

### Strategy 6: Direction-Optimizing BFS (Worked Very Well for A100)

**Decision:** Different frontier representations for different BFS phases.
**Why it worked:** Top-down is optimal when frontier is sparse (few vertices to expand). Bottom-up is optimal when frontier is dense (cheaper to scan unvisited vertices). A100's large L2 cache makes the bitmap representation efficient for bottom-up.

---

## 11. What Didn't Work / Limitations

### Limitation 1: Single-GPU Only

**What:** AAI requires `multi_gpu = false` and exactly 1 edge partition.
**Impact:** Cannot process graphs larger than single GPU memory (~40GB on A100, ~80GB on H100).
**Root cause:** `compact_graph_t::from_graph_view()` throws if `number_of_local_edge_partitions() != 1`. The kernel implementations assume all graph data is in one contiguous allocation.
**To overcome:** Would need distributed compact_graph_t with partition-aware kernels and inter-GPU communication.

### Limitation 2: int32 Only

**What:** `graph32_t = compact_graph_t<int32_t, int32_t>`. No int64 support.
**Impact:** Max ~2.1 billion vertices and ~2.1 billion edges.
**Root cause:** Integration layer only specializes `bfs<int32_t, int32_t, false>`. No `<int64_t, int64_t, false>` specialization exists.
**To overcome:** Add int64 variants of all 192 kernel files (or template them) + integration specializations.

### Limitation 3: No DCS (Hypersparse) Graph Support

**What:** Requires exactly 5 segment offsets. DCS format has more.
**Impact:** Cannot handle extreme power-law graphs where DCS is the efficient representation.
**Root cause:** `from_graph_view()` throws: "AAI requires exactly 5 segment offsets (4 segments). Hypersparse graphs (DCS) with N segment offsets are not supported."

### Limitation 4: Default CUDA Stream

**What:** All AAI kernels use stream 0.
**Impact:** AAI work cannot overlap with other GPU work. Integration layer must call `handle.sync_stream()` before and `cudaDeviceSynchronize()` after.
**Root cause:** CachePool is not stream-aware. Kernels don't accept stream parameters.

### Limitation 5: No CUDA Error Checking in Hot Paths

**What:** `cudaMalloc`, `cudaMemcpy`, kernel launches are not checked for errors.
**Impact:** Errors silently propagate. Debugging requires `CUDA_LAUNCH_BLOCKING=1`.
**Rationale:** Error checking adds overhead. cuGraph's pattern is to check errors at the integration boundary (the `cudaGetLastError()` after `cudaDeviceSynchronize()`).

### Limitation 6: Combinatorial Dispatch Explosion

**What:** PageRank has 8+ dispatch paths in the integration layer. Each new dimension (mask, seg, weights, personalization, precision) doubles the paths.
**Impact:** Integration files become large and error-prone.
**Root cause:** C++ template specialization requires exact type matching. Each combination needs its own call site.

---

## 12. Hypotheses That Were Validated

### H1: "Architecture-specific kernels will outperform generic ones" — VALIDATED

10-100x speedups on 18%+ of algorithms. The performance gap is real and large.

### H2: "Direction-optimizing BFS benefits from per-GPU threshold tuning" — VALIDATED

A100 uses N/20 (simple). L4 uses degree-cost model (more sophisticated). A10G uses ALPHA=4 (more aggressive switching). Each is tuned to its hardware's memory hierarchy.

### H3: "Different GPUs need fundamentally different algorithms, not just parameter tuning" — VALIDATED

A100 BFS uses queue↔bitmap conversion. A10G uses 2-level batching. L4 uses CUB-based degree computation. These are different algorithms, not the same algorithm with different constants.

### H4: "Shared-memory hash tables are effective for community detection on GPU" — VALIDATED

Louvain's 128-entry per-warp hash table with linear probing and atomicCAS achieves good performance. The 3-tier dispatch (serial/thread/warp) adapts to graph coarsening level.

### H5: "CachePool eliminates meaningful allocation overhead" — VALIDATED

Thread-local LRU with capacity 8 avoids cudaMalloc/cudaFree for repeated algorithm calls. The `ensure()` pattern only reallocates when graph size grows.

### H6: "Template specialization provides zero-overhead drop-in replacement" — VALIDATED

No API changes needed. Python users don't know AAI exists. The `#ifdef/#else` pattern cleanly falls back to original cuGraph when routing is disabled.

---

## 13. Adding a New GPU Target — Step-by-Step

### Phase 1: Infrastructure (Day 1)

1. **Add GPU macro to `target_gpu.hpp`:**
   ```cpp
   #define AAI_GPU_H100 4
   // Update validation #if to include AAI_GPU_H100
   #define AAI_IS_H100 (AAI_TARGET_GPU == AAI_GPU_H100)
   ```

2. **Add to `target_gpu_map.json`:**
   ```json
   "h100": { "arch": "90" }
   ```

3. **Add to `CMakeLists.txt` TARGET_GPU validation:**
   ```cmake
   elseif(TARGET_GPU STREQUAL "H100")
       set(AAI_TARGET_GPU_VALUE "AAI_GPU_H100")
       set(AAI_TARGET_GPU_SUFFIX "h100")
   ```

4. **Create directory tree:**
   ```
   cpp/src/aai/impl/h100/
   ├── traversal/
   ├── link_analysis/
   ├── centrality/
   ├── community/
   ├── components/
   ├── cores/
   ├── link_prediction/
   └── tree/
   ```

### Phase 2: Baseline (Days 2-5)

5. **Copy A100 kernels as starting point:**
   ```bash
   cp -r cpp/src/aai/impl/a100/* cpp/src/aai/impl/h100/
   ```

6. **Build and run cuGraph tests:**
   ```bash
   cmake -DTARGET_GPU=H100 -DCMAKE_CUDA_ARCHITECTURES=90 ..
   make -j
   pytest python/cugraph/tests/
   ```

7. **Profile with nsys/ncu:**
   - Identify memory-bound vs compute-bound kernels
   - Measure L2 hit rates, occupancy, register usage
   - Compare against cuGraph baseline on H100

### Phase 3: Kernel Optimization (Days 5-30+)

8. **Start with highest-impact algorithms** (BFS, PageRank, Louvain)

9. **For each algorithm:**
   a. Profile the A100 kernel on H100 — find bottlenecks
   b. Evaluate H100-specific features (TMA, clusters, DPX)
   c. Re-tune constants:
      - BFS switching thresholds (alpha, beta, N/X)
      - Block sizes and grid caps
      - Register pressure (`--maxrregcount`)
      - Hash table sizes (Louvain)
   d. Write new kernel or modify copied A100 kernel
   e. Create/update `.cu.flags` sidecar
   f. Test for correctness against cuGraph
   g. Benchmark for speedup

### Phase 4: Validation

10. **Correctness:** All algorithms produce identical results to cuGraph (within float tolerance)
11. **Performance:** Document speedup/slowdown per algorithm
12. **Edge cases:** Empty graphs, single vertex, disconnected, max-size int32
13. **Memory leaks:** `compute-sanitizer --tool memcheck`
14. **Thread safety:** Multiple CPU threads calling AAI (CachePool is thread-local, should be safe)

---

## 14. H100 and B200 — Specific Constraints and Opportunities

### H100 (SM90, Hopper)

| Feature | How to Exploit | Impact |
|---------|---------------|--------|
| **TMA (Tensor Memory Accelerator)** | Async bulk copies for frontier data, replacing cudaMemcpyAsync | Faster frontier transfers; frees SM compute during copy |
| **Thread Block Clusters** | Group SMs for cross-block shared memory | Louvain could share hash tables across blocks within cluster |
| **Distributed Shared Memory** | SMs in cluster access each other's SMEM | BFS frontier exchange between SMs without global memory |
| **DPX Instructions** | HW-accelerated dynamic programming | SSSP delta-stepping could use dpx for min/max operations |
| **50 MB L2** | Larger working sets fit in L2 | More aggressive bottom-up BFS, larger hash tables |
| **FP8** | New low-precision type | Possible low-precision PageRank for approximate results |
| **Warp Specialization** | Persistent producer/consumer warp roles | BFS: one warp expands frontier, another processes results |

**Recommended approach:** Start with A100 kernels (they'll work at SM90). Profile, then selectively add TMA for data movement and clusters for community detection.

### B200/B300 (SM100+, Blackwell)

| Constraint | Impact |
|-----------|--------|
| **New SM architecture** | Warp size, register file, SMEM may change — all hardcoded constants need re-evaluation |
| **NVLink 5.0** | Higher GPU-GPU bandwidth — potential for multi-GPU AAI |
| **HBM3e** | Higher memory bandwidth — memory-bound kernels (BFS BU, PageRank SpMV) benefit most |
| **Unknown ISA** | Must wait for CUDA toolkit with SM100 support |
| **Potential warp size change** | All `lane & 31`, `__ballot_sync(0xffffffff)`, `__shfl_down_sync` patterns may need updating |

**Recommended approach:** Wait for CUDA toolkit with Blackwell support. Copy H100 kernels as baseline. The main risk is warp size changes — abstract warp operations into macros/helpers.

---

## 15. Complete File Reference

### Core Infrastructure
| File | Lines | Purpose |
|------|-------|---------|
| `cpp/include/cugraph/aai/compact_graph.hpp` | 151 | Graph abstraction + `from_graph_view()` factory |
| `cpp/include/cugraph/aai/cache_pool.hpp` | 125 | Thread-local LRU GPU resource cache |
| `cpp/include/cugraph/aai/types.hpp` | 199 | Result types (device pointers + metadata) |
| `cpp/include/cugraph/aai/target_gpu.hpp` | 52 | `AAI_GPU_A100/L4/A10G` macros + validation |
| `cpp/include/cugraph/aai/api/traversal.hpp` | 1070 | BFS/SSSP/K-Hop API declarations (4-way dispatch) |
| `cpp/include/cugraph/aai/api/link_analysis.hpp` | 2022 | PageRank/HITS API declarations |

### A100 Implementation (Key Files)
| File | Lines | Algorithm |
|------|-------|-----------|
| `cpp/src/aai/impl/a100/traversal/bfs.cu` | 459 | Direction-optimizing BFS |
| `cpp/src/aai/impl/a100/traversal/bfs_mask.cu` | ~400 | Cooperative-kernel masked BFS |
| `cpp/src/aai/impl/a100/community/louvain_f32.cu` | 626 | 3-tier Louvain |
| `cpp/src/aai/impl/a100/link_analysis/pagerank.cu` | 314 | Custom SpMV PageRank |
| `cpp/src/aai/impl/a100/link_analysis/pagerank_f32.cu` | 435 | cuSPARSE PageRank |
| `cpp/src/aai/impl/a100/centrality/betweenness_centrality.cu` | 469 | Multi-source BFS + backward |
| `cpp/src/aai/impl/a100/community/triangle_count.cu` | 379 | DAG-based bilateral intersection |
| `cpp/src/aai/impl/a100/components/weakly_connected_components.cu` | 218 | Union-find with sampling |

### Integration Layer
| File | Lines | Dispatch |
|------|-------|----------|
| `cpp/src/aai/integration/traversal/bfs.cu` | 134 | 8-way (DO × mask × seg) |
| `cpp/src/aai/integration/link_analysis/pagerank.cu` | 370 | 16+ way (mask × pers × weights × seg × precision) |
| `cpp/src/aai/integration/community/louvain.cu` | 150 | 2-way (seg) × 2 precisions |

### Build System
| File | Purpose |
|------|---------|
| `cpp/CMakeLists.txt` | GPU selection, impl glob, .cu.flags, RDC, routing definitions |
| `build_in_docker/target_gpu_map.json` | GPU name → SM arch mapping |
| `build_in_docker/build_image.py` | Docker build with GPU target injection |
| `wheel_building/build_wheel.py` | Per-GPU wheel packaging |

### Cross-GPU Implementation
| File | Lines | Notes |
|------|-------|-------|
| `cpp/src/aai/impl/l4/traversal/bfs.cu` | 460 | CUB-based, int4 init, degree-cost model |
| `cpp/src/aai/impl/a10g/traversal/bfs.cu` | 562 | 2-level batching, pinned memory, frontier rebuild |
